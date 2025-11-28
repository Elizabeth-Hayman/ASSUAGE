
"""
Primary mlModels class. Depends on `ml_helpers` for metrics and default
model dictionary. This file contains the full class implementation but
imports lightweight helpers from the helpers module so behaviour remains
unchanged while the codebase is split.
"""

import json
import os
import pickle
import time
from typing import Dict, List, Optional, Union, Any, Tuple

import numpy as np
import pandas as pd

from sklearn.base import RegressorMixin
from sklearn.decomposition import PCA
from sklearn.model_selection import (
    KFold,
    LeaveOneOut,
    RandomizedSearchCV,
    GridSearchCV,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# import helpers
import ASSUAGE.surrogateModel.ml_helpers as helpers
import ASSUAGE.surrogateModel.ml_plotter as plotter



class mlModels:
    """
    Main class for fitting ML models and hyperparameter optimisation.

    This class preserves the original functionality but keeps configuration
    (DEFAULT_MODEL_DICT) and metric helpers in a separate module to make the
    code easier to test and reuse.
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

        # load data (keep behaviour identical)
        X = pd.read_csv(self.input_path, header=None).to_numpy()
        y = pd.read_csv(self.output_path, header=None).to_numpy().reshape(-1)

        self.X = X
        self.y = y

        # choose model dict
        self.model_dict = model_dict if model_dict is not None else helpers.DEFAULT_MODEL_DICT

        # brief logging
        with open(os.path.join(self.output_dir, f"{self.label}mlModelFitting_log.txt"), "w") as f:
            f.write(f"mlModels run: label={self.label}\n")
            f.write(f"Models: {list(self.model_dict.keys())}\n")
            f.write(f"PCA: {self.pca}\n")
            f.write(f"Search: {self.search_type}, n_iter={self.n_iter}\n")
            f.write(f"Parallel workers per model search: n_jobs={self.n_jobs}\n")
            f.write(f"Seed: {self.seed}\n")

    def _make_pipeline(self, estimator: RegressorMixin) -> Pipeline:
        """Create an sklearn Pipeline with StandardScaler, optional PCA, then estimator.

        The scaler and PCA (when enabled) are deliberately placed inside the
        pipeline to avoid data leakage during cross-validation.
        """
        steps = [("scaler", StandardScaler())]
        if self.pca is not None and self.pca is not False:
            if isinstance(self.pca, bool) and self.pca is True:
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
        """Return an approximate number of combinations in the parameter grid.

        If a value is not sized, ``None`` is returned to indicate unknown cardinality.
        """
        total = 1
        for v in param_grid.values():
            try:
                total *= max(1, len(v))
            except Exception:
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
    # Evaluation methods (unchanged behaviour)
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
            metrics = helpers._compute_metrics(yte, y_pred)
            results[name] = {"best_params": search.best_params_, "metrics": metrics, "y_true": yte.tolist(), "y_pred": y_pred.tolist()}

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
                plotter.plot_results(results, output_dir=self.output_dir, prefix=f"{self.label}holdout_{ts}")
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
                metrics = helpers._compute_metrics(yte, y_pred)
                folds.append({"fold": int(fold_idx), "best_params": search.best_params_, "metrics": metrics, "y_true": yte.tolist(), "y_pred": y_pred.tolist()})
                print(f"  fold {fold_idx} R2={metrics['r2']:.4f}")

            metric_names = list(folds[0]["metrics"].keys())
            agg = {}
            for mname in metric_names:
                arr = np.array([f["metrics"][mname] for f in folds], dtype=float)
                agg[mname] = {"mean": float(np.nanmean(arr)), "std": float(np.nanstd(arr))}
            aggregated[name] = {"folds": folds, "aggregated": agg}

            r2_mean = agg.get("r2", {}).get("mean", float("-inf"))
            if (self.best_model_score is None) or (r2_mean > self.best_model_score):
                self.best_model_name = name
                self.best_model_score = r2_mean
                self.best_model_search = search
                self.best_model_info = {"mode": "nested_cv", "timestamp": int(time.time()), "aggregated": agg}

            if self.best_model_cutoff is not None and r2_mean >= self.best_model_cutoff:
                models_above_cutoff.append((name, r2_mean))

            rows = []
            for f in folds:
                for yt, yp in zip(f["y_true"], f["y_pred"]):
                    rows.append({"model": name, "fold": f["fold"], "y_true": yt, "y_pred": yp})
            pd.DataFrame(rows).to_csv(os.path.join(self.output_dir, f"{self.label}{name}_nestedcv_predictions.csv"), index=False)

        ts = int(time.time())
        json_path = os.path.join(self.output_dir, f"{self.label}nestedcv_results_{ts}.json")
        with open(json_path, "w") as fh:
            json.dump(aggregated, fh, indent=2)

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
                plotter.plot_nested_results(aggregated, output_dir=self.output_dir, prefix=f"{self.label}nestedcv_{ts}")
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

            metrics = helpers._compute_metrics(np.array(y_true_all), np.array(y_pred_all))
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
                plotter.plot_results(results, output_dir=self.output_dir, prefix=f"{self.label}holdout_{ts}")
            except Exception as e:
                print(f"[holdout] Warning: plotting failed: {e}")

        return results

    def train_best_pipeline(self,
                            model_name: Optional[str] = None,
                            param_grid: Optional[Dict[str, Any]] = None,
                            cv: int = 5,
                            save_path: Optional[str] = None):
        """Train the chosen model on the full dataset and save the fitted pipeline.

        If ``model_name`` is None the method prefers ``self.best_model_name`` (set
        during evaluation). If that is not available it attempts to parse the
        last results JSON file saved by evaluations.
        """
        chosen = model_name or getattr(self, "best_model_name", None)
        if chosen is None:
            if getattr(self, "_last_results_file", None) is None:
                raise RuntimeError("No best model known. Run an evaluation first or pass model_name.")
            with open(self._last_results_file, "r") as fh:
                data = json.load(fh)
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

        self.best_pipeline = pipeline
        self.best_pipeline_path = save_path

        try:
            y_pred = pipeline.predict(self.X)
            r2_full = float(np.round(helpers._compute_metrics(self.y, y_pred)["r2"], 6))
            print(f"[train_best_pipeline] Refit R2 on full dataset (optimistic): {r2_full:.4f}")
            if self.best_model_cutoff is not None and r2_full >= self.best_model_cutoff:
                print(f"[train_best_pipeline] MODEL '{chosen}' exceeds cutoff {self.best_model_cutoff} with R2={r2_full:.4f}")
        except Exception:
            pass

        return pipeline
