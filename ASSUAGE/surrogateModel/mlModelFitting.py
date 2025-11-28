
"""
Machine learning model fitting and hyperparameter optimisation.

This file provides flexible hyperparameter tuning
(multi-model), evaluation strategies (holdout, nested CV, LOO), and optional PCA.

Key points
- All preprocessing (StandardScaler and optional PCA) is placed inside each
  model's sklearn Pipeline. This ensures that cross-validation and hyperparameter
  search are clean (no leakage) and that scaling is applied identically during
  training and prediction for each fold.
- General scaling (a global StandardScaler) is NOT used to transform the whole
  dataset outside the CV loop; instead the scaler is in the pipeline. The only
  reason to compute a global scaler would be purely diagnostic; it risks data
  leakage if used before CV.
- User options include choice of search ('random' or 'grid'), n_iter for random
  search, n_jobs controlling parallelism *inside* each model's search, PCA on/off,
  and which evaluation protocol to run.
"""

import json
import os
import pickle
import time
from typing import Dict, List, Optional, Union, Any, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from scipy.stats import spearmanr, pearsonr
from sklearn.base import RegressorMixin, clone
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import BayesianRidge, Lasso, LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error, make_scorer
)
from sklearn.model_selection import (
    KFold,
    LeaveOneOut,
    RandomizedSearchCV,
    GridSearchCV,
    train_test_split
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# configure plotting defaults
matplotlib.rcParams.update({"font.size": 35})


# -----------------------------
# Utility scorers and helpers
# -----------------------------
def _spearman_scorer(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return spearman correlation (or nan on failure)."""
    try:
        return spearmanr(y_true, y_pred).correlation
    except Exception:
        return float("nan")


def _pearson_safe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return pearson correlation (or nan on failure)."""
    try:
        return pearsonr(y_true, y_pred)[0]
    except Exception:
        return float("nan")


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute a small suite of regression metrics."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return {
        "r2": float(np.round(r2_score(y_true, y_pred), 6)),
        "mse": float(np.round(mean_squared_error(y_true, y_pred), 6)),
        "rmse": float(np.round(np.sqrt(mean_squared_error(y_true, y_pred)), 6)),
        "mae": float(np.round(mean_absolute_error(y_true, y_pred), 6)),
        "pearson": float(np.round(_pearson_safe(y_true, y_pred), 6)),
        "spearman": float(np.round(_spearman_scorer(y_true, y_pred), 6))
    }


# -----------------------------
# Full model dictionary & hyperparameters
# -----------------------------
DEFAULT_MODEL_DICT = {
    'Linear regression': (LinearRegression(), {
        # LinearRegression supports fit_intercept and (recent sklearn) positive
        'model__fit_intercept': [True, False],
        'model__positive': [False]  # typically False; include True only if non-negative coefficients required
    }),

    'Ridge': (Ridge(), {
        'model__alpha': [0.01, 0.1, 1.0, 10.0, 100.0],
        'model__fit_intercept': [True, False],
        # solver choices valid for ridge in sklearn (may be ignored for small problems)
        'model__solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sag', 'saga']
    }),

    'Lasso': (Lasso(max_iter=10000), {
        'model__alpha': [1e-4, 1e-3, 1e-2, 1e-1, 1.0],
        'model__fit_intercept': [True, False],
        # selection allowed for coordinate descent; 'cyclic' and 'random' are valid
        'model__selection': ['cyclic', 'random']
    }),

    'Bayesian Ridge': (BayesianRidge(), {
        # BayesianRidge accepts n_iter (number of iterations), tol (tolerance)
        # and hyperpriors lambda/alpha; tune a small subset to keep search light.
        'model__max_iter': [100, 300, 500],
        'model__tol': [1e-4, 1e-3, 1e-2],
        'model__alpha_1': [1e-6, 1e-5, 1e-4],
        'model__lambda_1': [1e-6, 1e-5, 1e-4]
    }),

    'Decision Tree': (DecisionTreeRegressor(), {
        'model__criterion': ['squared_error', 'friedman_mse', 'absolute_error'],
        'model__max_depth': [None, 3, 5, 10],
        'model__min_samples_leaf': [1, 2, 4]
    }),

    'Random Forest': (RandomForestRegressor(), {
        'model__n_estimators': [100, 200, 300],
        'model__max_depth': [None, 5, 10, 20],
        'model__min_samples_leaf': [1, 2, 4],
        'model__max_features': ['sqrt', 'log2', 0.5]  # allow fraction or aliases
    }),

    'Gradient Boosting': (GradientBoostingRegressor(), {
        'model__n_estimators': [100, 200],
        'model__learning_rate': [0.01, 0.05, 0.1],
        'model__max_depth': [3, 5],
        'model__subsample': [1.0, 0.8]  # reduce overfitting option
    }),

    'SVR': (SVR(), {
        'model__C': [0.1, 1, 10],
        'model__gamma': ['scale', 'auto'],
        'model__kernel': ['rbf', 'poly'],
        # degree only used when kernel='poly'
        'model__degree': [2, 3]
    }),

    'MLP': (MLPRegressor(max_iter=2000), {
        'model__hidden_layer_sizes': [(50,), (100,), (100, 50)],
        'model__activation': ['relu', 'tanh'],
        # solver 'lbfgs' ignores learning_rate_init; fine to include but note it may be unused
        'model__solver': ['adam', 'lbfgs'],
        'model__alpha': [1e-5, 1e-4, 1e-3],
        'model__learning_rate_init': [1e-3, 1e-2]
    }),
}

# -----------------------------
# Core class 
# -----------------------------
class mlModels:
    """
    Main class for fitting ML models and hyperparameter optimisation.

    Parameters (high-level)
    -----------------------
    input_data: path to CSV (no header) for inputs
    output_data: path to CSV (no header) for outputs (target)
    model_dict: optional dict of models -> (estimator, param_grid/distributions)
                If omitted, DEFAULT_MODEL_DICT is used.
    output_dir: directory to save results (default 'surrogateCreation')
    pca: bool or float or int
         - False or None: do not use PCA
         - True: use PCA retaining 95% variance
         - float in (0,1): treat as fraction of variance to retain (pass to PCA n_components)
         - int >0: use that number of components
    search_type: 'random' or 'grid' (RandomizedSearchCV vs GridSearchCV)
    n_iter: iterations for randomized search (ignored for grid)
    n_jobs: number of parallel workers used inside each model search
    seed: random_state for reproducibility

    NOTE on scaling and pipelines
    ----------------------------
    All preprocessing (scaling and PCA) is applied **inside** each model's sklearn
    Pipeline (StandardScaler -> optional PCA -> model). This guarantees that during
    cross-validation the scaler and PCA are fit only on training folds (avoids leakage).
    """

    def __init__(self,
                 input_data: Union[str, os.PathLike],
                 output_data: Union[str, os.PathLike],
                 model_dict: Optional[Dict[str, Tuple[RegressorMixin, Dict[str, Any]]]] = None,
                 data_label: str = "",
                 output_dir: str = "surrogateCreation",
                 pca: Union[bool, float, int, None] = True,
                 search_type: str = "random",
                 best_model_cutoff: float = 0.9,
                 n_iter: int = 50,
                 n_jobs: int = 1,
                 seed: int = 42) -> None:

        self.input_path = str(input_data)
        self.output_path = str(output_data)
        self.label = data_label or ""
        self.output_dir = os.path.join(output_dir, data_label)
        os.makedirs(self.output_dir, exist_ok=True)

        self.pca = pca
        self.search_type = search_type.lower()
        assert self.search_type in ("random", "grid"), "search_type must be 'random' or 'grid'"
        self.n_iter = n_iter
        self.n_jobs = n_jobs
        self.seed = seed
        self.best_model_cutoff = best_model_cutoff
        self.best_model_score = None

        # load data
        X = pd.read_csv(self.input_path, header=None).to_numpy()
        y = pd.read_csv(self.output_path, header=None).to_numpy().reshape(-1)

        self.X = X
        self.y = y

        # choose model dict
        self.model_dict = model_dict if model_dict is not None else DEFAULT_MODEL_DICT

        # brief logging
        with open(os.path.join(self.output_dir, f"{self.label}mlModelFitting_log.txt"), "w") as f:
            f.write(f"mlModels run: label={self.label}\n")
            f.write(f"Models: {list(self.model_dict.keys())}\n")
            f.write(f"PCA: {self.pca}\n")
            f.write(f"Search: {self.search_type}, n_iter={self.n_iter}\n")
            f.write(f"Parallel workers per model search: n_jobs={self.n_jobs}\n")
            f.write(f"Seed: {self.seed}\n")
        # ---------------------------------------------------------------------

    def _make_pipeline(self, estimator: RegressorMixin) -> Pipeline:
        steps = [("scaler", StandardScaler())]
        # Handle pca flag/float/int
        if self.pca is not None and self.pca is not False:
            if isinstance(self.pca, bool) and self.pca is True:
                # default: keep 95% variance
                steps.append(("pca", PCA(n_components=0.95, svd_solver="full", random_state=self.seed)))
            elif isinstance(self.pca, float) and 0.0 < self.pca < 1.0:
                steps.append(("pca", PCA(n_components=self.pca, svd_solver="full", random_state=self.seed)))
            elif isinstance(self.pca, int) and self.pca > 0:
                steps.append(("pca", PCA(n_components=self.pca, random_state=self.seed)))
            else:
                raise ValueError("Invalid pca parameter; expected bool, float in (0,1), or int >0.")
        steps.append(("model", estimator))
        return Pipeline(steps)

    def _param_grid_combinations(self, param_grid: Dict[str, Any]) -> Optional[int]:
        total = 1
        for v in param_grid.values():
            try:
                total *= max(1, len(v))
            except:
                return None
        return int(total)

    def _search_and_fit(self, name: str, estimator: Any, param_grid: Dict[str, Any], X_train: np.ndarray, y_train: np.ndarray, cv: int = 5):
        pipe = self._make_pipeline(estimator)
        if self.search_type == "random":
            combos = self._param_grid_combinations(param_grid)
            n_iter_effective = self.n_iter if combos is None else min(self.n_iter, combos)
            search = RandomizedSearchCV(pipe,
                                        param_distributions=param_grid,
                                        n_iter=n_iter_effective,
                                        cv=cv,
                                        scoring="r2",
                                        random_state=self.seed,
                                        n_jobs=self.n_jobs,
                                        refit=True,
                                        verbose=0)
        else:
            search = GridSearchCV(pipe,
                                  param_grid=param_grid,
                                  cv=cv,
                                  scoring="r2",
                                  n_jobs=self.n_jobs,
                                  refit=True,
                                  verbose=0)
        search.fit(X_train, y_train)
        return search

    # -----------------------
    # Evaluation methods (they update live best-model info)
    # -----------------------
    def evaluate_holdout(self, test_size: float = 0.1, inner_cv: int = 5, create_plots: bool = True) -> Dict[str, Any]:
        Xtr, Xte, ytr, yte = train_test_split(self.X, self.y, test_size=test_size, random_state=self.seed, shuffle=True)
        results = {}
        models_above_cutoff = []

        for name, (est, params) in self.model_dict.items():
            t0 = time.time()
            print(f"[holdout] Searching for model: {name}")
            search = self._search_and_fit(name, est, params, Xtr, ytr, cv=inner_cv)
            best = search.best_estimator_
            y_pred = best.predict(Xte)
            metrics = _compute_metrics(yte, y_pred)
            results[name] = {"best_params": search.best_params_, "metrics": metrics, "y_true": yte.tolist(), "y_pred": y_pred.tolist()}

            # update live best-model if this model is better
            r2 = metrics.get("r2", float("-inf"))
            if (self.best_model_score is None) or (r2 > self.best_model_score):
                self.best_model_name = name
                self.best_model_score = r2
                self.best_model_search = search
                self.best_model_info = {"mode": "holdout", "timestamp": int(time.time()), "metrics": metrics, "best_params": search.best_params_}

            if self.best_model_cutoff is not None and r2 >= self.best_model_cutoff:
                models_above_cutoff.append((name, r2))

            pd.DataFrame({"y_true": yte, "y_pred": y_pred}).to_csv(os.path.join(self.output_dir, f"{self.label}{name}_holdout_predictions.csv"), index=False)
            print(f"[holdout] {name} done in {time.time() - t0:.1f}s, R2={r2:.4f}")

        ts = int(time.time())
        json_path = os.path.join(self.output_dir, f"{self.label}holdout_results_{ts}.json")
        csv_path = os.path.join(self.output_dir, f"{self.label}holdout_summary_{ts}.csv")
        with open(json_path, "w") as fh:
            json.dump(results, fh, indent=2)
        pd.DataFrame([{"model": n, **v["metrics"]} for n, v in results.items()]).to_csv(csv_path, index=False)

        self._last_results_file = json_path

        if models_above_cutoff:
            print("[holdout] Models exceeding cutoff:")
            for n, r in models_above_cutoff:
                print(f"   {n}: R2={r:.4f}")
        
        if create_plots:
            try:
                self._plot_results(results, prefix=f"{self.label}holdout_{ts}")
            except Exception as e:
                print(f"[holdout] Warning: plotting failed: {e}")

        return results

    def evaluate_nested_cv(self, outer_splits: int = 5, inner_splits: int = 5, create_plots: bool = True) -> Dict[str, Any]:
        kf = KFold(n_splits=outer_splits, shuffle=True, random_state=self.seed)
        aggregated: Dict[str, Any] = {}
        models_above_cutoff = []

        for name, (est, params) in self.model_dict.items():
            print(f"[nested_cv] Evaluating model: {name}")
            folds = []
            for fold_idx, (train_idx, test_idx) in enumerate(kf.split(self.X, self.y)):
                Xtr, Xte = self.X[train_idx], self.X[test_idx]
                ytr, yte = self.y[train_idx], self.y[test_idx]
                search = self._search_and_fit(name, est, params, Xtr, ytr, cv=inner_splits)
                best = search.best_estimator_
                y_pred = best.predict(Xte)
                metrics = _compute_metrics(yte, y_pred)
                folds.append({"fold": int(fold_idx), "best_params": search.best_params_, "metrics": metrics, "y_true": yte.tolist(), "y_pred": y_pred.tolist()})
                print(f"  fold {fold_idx} R2={metrics['r2']:.4f}")

            metric_names = list(folds[0]["metrics"].keys())
            agg = {}
            for mname in metric_names:
                arr = np.array([f["metrics"][mname] for f in folds], dtype=float)
                agg[mname] = {"mean": float(np.nanmean(arr)), "std": float(np.nanstd(arr))}
            aggregated[name] = {"folds": folds, "aggregated": agg}

            # use aggregated mean R2 for comparison
            r2_mean = agg.get("r2", {}).get("mean", float("-inf"))
            if (self.best_model_score is None) or (r2_mean > self.best_model_score):
                # attach the last search (best of last outer fold) as representative
                self.best_model_name = name
                self.best_model_score = r2_mean
                self.best_model_search = search  # last inner search object (representative)
                self.best_model_info = {"mode": "nested_cv", "timestamp": int(time.time()), "aggregated": agg}

            if self.best_model_cutoff is not None and r2_mean >= self.best_model_cutoff:
                models_above_cutoff.append((name, r2_mean))

            # save fold-level predictions CSV
            rows = []
            for f in folds:
                for yt, yp in zip(f["y_true"], f["y_pred"]):
                    rows.append({"model": name, "fold": f["fold"], "y_true": yt, "y_pred": yp})
            pd.DataFrame(rows).to_csv(os.path.join(self.output_dir, f"{self.label}{name}_nestedcv_predictions.csv"), index=False)

        ts = int(time.time())
        json_path = os.path.join(self.output_dir, f"{self.label}nestedcv_results_{ts}.json")
        with open(json_path, "w") as fh:
            json.dump(aggregated, fh, indent=2)
        # write flattened summary
        summary_rows = []
        for name, info in aggregated.items():
            row = {"model": name}
            for k, stats in info["aggregated"].items():
                row[f"{k}_mean"] = stats["mean"]
                row[f"{k}_std"] = stats["std"]
            summary_rows.append(row)
        pd.DataFrame(summary_rows).to_csv(os.path.join(self.output_dir, f"{self.label}nestedcv_summary_{ts}.csv"), index=False)

        self._last_results_file = json_path

        if models_above_cutoff:
            print("[nested_cv] Models exceeding cutoff (mean R2):")
            for n, r in models_above_cutoff:
                print(f"   {n}: mean R2={r:.4f}")
        
        if create_plots:
            try:
                self._plot_nested_results(aggregated, prefix=f"{self.label}nestedcv_{ts}")
            except Exception as e:
                print(f"[holdout] Warning: plotting failed: {e}")

        return aggregated

    def evaluate_loo(self, inner_cv: int = 5, create_plots: bool = True) -> Dict[str, Any]:
        loo = LeaveOneOut()
        results: Dict[str, Any] = {}
        models_above_cutoff = []

        for name, (est, params) in self.model_dict.items():
            print(f"[loo] Evaluating model: {name}")
            y_true_all = []
            y_pred_all = []
            params_list = []
            for train_idx, test_idx in loo.split(self.X, self.y):
                Xtr, Xte = self.X[train_idx], self.X[test_idx]
                ytr, yte = self.y[train_idx], self.y[test_idx]
                search = self._search_and_fit(name, est, params, Xtr, ytr, cv=inner_cv)
                best = search.best_estimator_
                y_pred = best.predict(Xte)
                y_true_all.append(float(yte[0]))
                y_pred_all.append(float(y_pred[0]))
                params_list.append(search.best_params_)

            metrics = _compute_metrics(np.array(y_true_all), np.array(y_pred_all))
            results[name] = {"metrics": metrics, "y_true": y_true_all, "y_pred": y_pred_all, "params": params_list}

            r2v = metrics.get("r2", float("-inf"))
            if (self.best_model_score is None) or (r2v > self.best_model_score):
                self.best_model_name = name
                self.best_model_score = r2v
                self.best_model_search = None
                self.best_model_info = {"mode": "loo", "timestamp": int(time.time()), "metrics": metrics}

            if self.best_model_cutoff is not None and r2v >= self.best_model_cutoff:
                models_above_cutoff.append((name, r2v))

            pd.DataFrame({"y_true": y_true_all, "y_pred": y_pred_all}).to_csv(os.path.join(self.output_dir, f"{self.label}{name}_loo_predictions.csv"), index=False)
            print(f"[loo] {name} done: R2={r2v:.4f}")

        ts = int(time.time())
        json_path = os.path.join(self.output_dir, f"{self.label}loo_results_{ts}.json")
        with open(json_path, "w") as fh:
            json.dump(results, fh, indent=2)
        self._last_results_file = json_path

        if models_above_cutoff:
            print("[loo] Models exceeding cutoff:")
            for n, r in models_above_cutoff:
                print(f"   {n}: R2={r:.4f}")
        
        if create_plots:
            try:
                self._plot_results(results, prefix=f"{self.label}holdout_{ts}")
            except Exception as e:
                print(f"[holdout] Warning: plotting failed: {e}")

        return results

    # -----------------------
    # Train best pipeline using stored best-model info if no name provided
    # -----------------------
    def train_best_pipeline(self,
                            model_name: Optional[str] = None,
                            param_grid: Optional[Dict[str, Any]] = None,
                            cv: int = 5,
                            save_path: Optional[str] = None):
        """
        Train the best pipeline on the full dataset.

        If model_name is None:
          - prefer the live stored self.best_model_name (set during evaluation runs)
          - if not set, fall back to self._last_results_file parsing (rare)
        """
        # choose model
        chosen = model_name or self.best_model_name
        if chosen is None:
            # fallback: try to infer from last results file
            if self._last_results_file is None:
                raise RuntimeError("No best model known. Run an evaluation first or pass model_name.")
            # simple JSON parse as fallback
            with open(self._last_results_file, "r") as fh:
                data = json.load(fh)
            # pick best by r2
            best_name = None
            best_r2 = float("-inf")
            for name, info in data.items():
                if isinstance(info, dict) and "metrics" in info and "r2" in info["metrics"]:
                    try:
                        r2v = float(info["metrics"]["r2"])
                        if r2v > best_r2:
                            best_r2 = r2v
                            best_name = name
                    except Exception:
                        pass
            if best_name is None:
                raise RuntimeError("Could not infer best model from last results file; pass model_name.")
            chosen = best_name
            print(f"[train_best_pipeline] Fallback chose '{chosen}' from last results file.")

        if chosen not in self.model_dict:
            raise KeyError(f"Model '{chosen}' not in model_dict.")

        estimator, stored_grid = self.model_dict[chosen]
        grid_to_use = param_grid if param_grid is not None else stored_grid

        print(f"[train_best_pipeline] Running CV search on full dataset for '{chosen}' (cv={cv})")
        search = self._search_and_fit(chosen, estimator, grid_to_use, self.X, self.y, cv=cv)

        best_params = getattr(search, "best_params_", None)
        print(f"[train_best_pipeline] Best params: {best_params}")

        pipeline = self._make_pipeline(estimator)
        if best_params:
            pipeline.set_params(**best_params)
        pipeline.fit(self.X, self.y)

        if save_path is None:
            safe = chosen.replace(" ", "_").replace("/", "_")
            save_path = os.path.join(self.output_dir, f"{self.label}{safe}_best_pipeline.pkl")

        with open(save_path, "wb") as fh:
            pickle.dump(pipeline, fh, protocol=pickle.HIGHEST_PROTOCOL)

        # record best pipeline and path
        self.best_pipeline = pipeline
        self.best_pipeline_path = save_path

        # optimistic R2 on full data
        try:
            y_pred = pipeline.predict(self.X)
            r2_full = float(np.round(r2_score(self.y, y_pred), 6))
            print(f"[train_best_pipeline] Refit R2 on full dataset (optimistic): {r2_full:.4f}")
            if self.best_model_cutoff is not None and r2_full >= self.best_model_cutoff:
                print(f"[train_best_pipeline] MODEL '{chosen}' exceeds cutoff {self.best_model_cutoff} with R2={r2_full:.4f}")
        except Exception:
            pass

        return pipeline

    # ---------------------------------------------------------------------
    # plotting helpers
    # ---------------------------------------------------------------------
    def _plot_results(self, results: Dict[str, Any], prefix: str = "results") -> None:
        """
        For each model produce a true-vs-pred scatter and a combined summary figure.
        """
        # per model figure + combined grid
        n = len(results)
        ncols = min(3, n)
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(15 * ncols, 12 * nrows))
        axes_flat = np.array(axes).reshape(-1)

        for ax in axes_flat:
            ax.set_visible(False)

        for i, (name, info) in enumerate(results.items()):
            ax = axes_flat[i]
            ax.set_visible(True)
            y_true = np.array(info.get("y_true", []))
            y_pred = np.array(info.get("y_pred", []))
            ax.scatter(y_pred, y_true, s=10)
            if y_true.size:
                m = _compute_metrics(y_true, y_pred)
                ax.text(0.05, 0.95, f"R2={m['r2']:.3f}\nRMSE={m['rmse']:.3f}\nMAE={m['mae']:.3f}\nρ={m['spearman']:.3f}", transform=ax.transAxes, va="top")
                mn, mx = float(min(y_true.min(), y_pred.min())), float(max(y_true.max(), y_pred.max()))
                if mn == mx:
                    mn -= 1; mx += 1
                ax.plot([mn, mx], [mn, mx], "r--", linewidth=1)
                ax.set_xlim([mn, mx]); ax.set_ylim([mn, mx])
            ax.set_title(name)

        plt.tight_layout()
        out_png = os.path.join(self.output_dir, f"{prefix}_summary.png")
        plt.savefig(out_png, bbox_inches="tight")
        plt.close()
        print(f"Wrote summary figure to {out_png}")

    def _plot_nested_results(self, aggregated: Dict[str, Any], prefix: str = "nested") -> None:
        """
        For each model create a scatter combining predictions across folds.
        """
        for name, info in aggregated.items():
            ys = []
            yps = []
            for f in info.get("folds", []):
                ys.extend(f.get("y_true", []))
                yps.extend(f.get("y_pred", []))
            ys = np.array(ys); yps = np.array(yps)
            fig, ax = plt.subplots(figsize=(20, 15))
            ax.scatter(yps, ys, s=10)
            if ys.size:
                m = _compute_metrics(ys, yps)
                ax.text(0.05, 0.95, f"R2={m['r2']:.3f}\nρ={m['spearman']:.3f}", transform=ax.transAxes, va="top")
                mn, mx = float(min(ys.min(), yps.min())), float(max(ys.max(), yps.max()))
                if mn == mx:
                    mn -= 1; mx += 1
                ax.plot([mn, mx], [mn, mx], "r--", linewidth=1)
                ax.set_xlim([mn, mx]); ax.set_ylim([mn, mx])
            out_png = os.path.join(self.output_dir, f"{prefix}_{name}.png")
            fig.tight_layout()
            fig.savefig(out_png, bbox_inches="tight")
            plt.close(fig)
            print(f"Wrote nested figure for {name} to {out_png}")


