#----------------------------
# Plotting functions to use to visualise the ML training creation
#----------------------------



from typing import Dict, List, Optional, Union, Any, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import time

from sklearn.inspection import permutation_importance

# import helpers
import ASSUAGE.surrogateModel.ml_helpers as helpers

# configure plotting defaults
matplotlib.rcParams.update({"font.size": 35})


def plot_results(results: Dict[str, Any], output_dir, prefix: str = "results") -> None:
    """Produce per-model true-vs-pred scatter plots and write a combined figure.

    The function writes a PNG file into ``output_dir`` with the given
    prefix.
    """
    n = len(results)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(10 * ncols, 8 * nrows))
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
            m = helpers._compute_metrics(y_true, y_pred)
            ax.text(0.05, 0.95, f"R2={m['r2']:.3f}\nRMSE={m['rmse']:.3f}\nMAE={m['mae']:.3f}\n\u03C1={m['spearman']:.3f}", transform=ax.transAxes, va="top")
            mn, mx = float(min(y_true.min(), y_pred.min())), float(max(y_true.max(), y_pred.max()))
            if mn == mx:
                mn -= 1; mx += 1
            ax.plot([mn, mx], [mn, mx], "r--", linewidth=1)
            ax.set_xlim([mn, mx]); ax.set_ylim([mn, mx])
        ax.set_title(name)

    plt.tight_layout()
    out_png = os.path.join(output_dir, f"{prefix}_summary.png")
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()
    print(f"Wrote summary figure to {out_png}")

def plot_nested_results( aggregated: Dict[str, Any], output_dir, prefix: str = "nested") -> None:
    """Create scatter plots combining predictions across folds for each model."""
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
            m = helpers._compute_metrics(ys, yps)
            ax.text(0.05, 0.95, f"R2={m['r2']:.3f}\n\u03C1={m['spearman']:.3f}", transform=ax.transAxes, va="top")
            mn, mx = float(min(ys.min(), yps.min())), float(max(ys.max(), yps.max()))
            if mn == mx:
                mn -= 1; mx += 1
            ax.plot([mn, mx], [mn, mx], "r--", linewidth=1)
            ax.set_xlim([mn, mx]); ax.set_ylim([mn, mx])
        out_png = os.path.join(output_dir, f"{prefix}_{name}.png")
        fig.tight_layout()
        fig.savefig(out_png, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote nested figure for {name} to {out_png}")

def feature_importance( pipeline,
                        output_dir: str,
                        X: np.ndarray,
                        y: Optional[np.ndarray] = None,
                        label: str = "",
                        feature_names: Optional[list] = None,
                        use_permutation: bool = False,
                        n_repeats: int = 30,
                        top_k: int = 20,
                        seed: int = 42,
                        n_jobs: int = 1,
                        create_plots: bool = True,
                        csv_name: Optional[str] = None,
                        png_name: Optional[str] = None) -> pd.DataFrame:
    """Save and plot feature importances for the given pipeline (or best pipeline).

    Behaviour mirrors the original implementation: prefer ``feature_importances_``
    or ``coef_`` when present; otherwise compute permutation importance.

    The function will attempt to load ``pipeline`` from ``best_pipeline`` or
    ``best_pipeline_path`` if ``pipeline`` is None. It uses ``X`` and
    ``y`` as default data if the user does not provide ``X`` explicitly.
    """

    n_features = X.shape[1]
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(n_features)]
    if len(feature_names) != n_features:
        raise ValueError("feature_names length must equal number of columns in X")

    # ---- unwrap model and detect PCA ----
    if "model" in pipeline.named_steps:
        model = pipeline.named_steps["model"]
    else:
        model = pipeline.steps[-1][1]

    pca = pipeline.named_steps.get("pca", None)

    importances = None
    reason = None

    if (not use_permutation) and hasattr(model, "feature_importances_"):
        importances = np.array(getattr(model, "feature_importances_")).ravel()
        reason = "feature_importances_"
    elif (not use_permutation) and hasattr(model, "coef_"):
        coef = getattr(model, "coef_")
        if coef.ndim == 1:
            importances = np.abs(coef).ravel()
        else:
            importances = np.mean(np.abs(coef), axis=0).ravel()
        reason = "coef_ (abs or mean)"
    else:
        use_permutation = True

    if use_permutation:
        if y is None:
            raise RuntimeError("y is not set as input. y is required for permutation calculation.")
        print(f"[feature_importance] Computing permutation importance (n_repeats={n_repeats}). This may be slow.")
        res = permutation_importance(pipeline, X, y,
                                    n_repeats=n_repeats, random_state=seed, n_jobs=n_jobs, scoring="r2")
        importances = res.importances_mean
        reason = f"permutation_importance (n_repeats={n_repeats})"

    if importances is None:
        raise RuntimeError("Could not determine importances (unexpected).")

    # ---- map back through PCA if needed ----
    if pca is not None:
        try:
            comp = pca.components_
            if comp.shape[0] != importances.shape[0]:
                if importances.shape[0] == comp.shape[1]:
                    mapped_importances = importances
                else:
                    raise RuntimeError("Dimension mismatch between PCA components and importances.")
            else:
                mapped_importances = np.abs(comp.T.dot(importances))
            final_importances = mapped_importances
            mapping_note = "mapped_from_pca_components"
        except Exception as e:
            final_importances = importances
            feature_names = [f"PC{i}" for i in range(len(importances))]
            mapping_note = f"mapping_failed:{e}"
    else:
        final_importances = importances
        mapping_note = "direct"

    final_importances = np.asarray(final_importances).ravel()
    final_importances = np.nan_to_num(final_importances, nan=0.0, posinf=0.0, neginf=0.0)

    df = pd.DataFrame({"feature": feature_names, "importance": final_importances})
    df = df.sort_values("importance", ascending=False).reset_index(drop=True)

    ts = int(time.time())
    if csv_name is None:
        csv_name = f"{label}feature_importances_{ts}.csv"
    csv_path = os.path.join(output_dir, csv_name)
    df.to_csv(csv_path, index=False)

    if create_plots:
        top = df.head(top_k)
        plt.figure(figsize=(max(12, top_k*0.6), 12))
        plt.barh(range(len(top)), top["importance"].values[::-1], align="center")
        plt.yticks(range(len(top)), top["feature"].values[::-1])
        plt.xlabel("Importance (arbitrary scale)")
        plt.title(f"Feature importances ({reason}; mapping={mapping_note})")
        plt.tight_layout()
        if png_name is None:
            png_name = f"{label}feature_importances_{ts}.png"
        png_path = os.path.join(output_dir, png_name)
        plt.savefig(png_path, dpi=150)
        plt.close()
        print(f"[feature_importance] Saved plot to {png_path}")

    print(f"[feature_importance] Saved CSV to {csv_path}  (reason={reason}; mapping={mapping_note})")
    return df
