# -------------------------------------
# Data exploration utilities for surrogate ML layer
#
# Single canonical input DataFrame (self.input_df).
# Scaling is performed once and stored in self.input_data_scaled for
# PCA / ML tasks. The user-facing flag `scaled_vis` controls whether
# visualisations use the scaled data or the raw input DataFrame.
# Local seaborn/pandas plotting warnings are suppressed during plotting.
# -------------------------------------

import os
from contextlib import contextmanager
import warnings
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

os.makedirs("surrogateCreation", exist_ok=True)


class DataExploration:
    """
    Perform common exploratory data analysis (EDA) tasks for surrogate modelling.

    Behaviour notes
    --------------
    - self.input_df: cleaned, unscaled DataFrame (canonical stored inputs).
    - self.input_data_scaled: numpy array; StandardScaler fitted on input_df.
    - scaled_vis (bool): if True, visualisation methods use the scaled data;
      if False, they use self.input_df (unscaled).
    - PCA and model-based diagnostics always use the scaled data for
      numerical stability.
    """

    def __init__(self,
                 input_data_path: str,
                 output_data_path: str,
                 label: str = "",
                 scaled_vis: bool = True) -> None:
        """
        Load CSVs, remove constant columns, fit scaler.

        Parameters
        ----------
        input_data_path : str
            Path to CSV of input parameters (no header).
        output_data_path : str
            Path to CSV of output values (no header).
        label : str
            Prefix for saved files.
        scaled_vis : bool
            If True, plots will visualise the scaled data. If False, plots
            will use the original unscaled input dataframe.
        """
        self.input_path = input_data_path
        self.output_path = output_data_path
        self.label = label or ""
        self.scaled_vis = bool(scaled_vis)
        self.log_file = os.path.join("surrogateCreation", f"{self.label}logFile.txt")

        # Load data
        try:
            self.input_df = pd.read_csv(self.input_path, header=None)
            self.output_data = pd.read_csv(self.output_path, header=None)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Input/output files missing: {e}. Ensure CSV paths are correct.")

        # Clean: remove constant columns
        nan_count = int(self.input_df.isna().sum().sum())
        constant_cols = self.input_df.nunique() == 1
        constant_indices = constant_cols[constant_cols].index.tolist()
        if constant_indices:
            self.input_df.drop(columns=constant_indices, inplace=True)

        # Persist cleaned unscaled inputs (same file) so downstream steps see the cleaned version
        self.input_df.to_csv(self.input_path, index=False, header=None)

        # Fit scaler once (used for PCA / ML). Keep scaled as a numpy array.
        self.scaler = StandardScaler()
        self.input_data_scaled = self.scaler.fit_transform(self.input_df)

        # Ensure output folder exists
        os.makedirs("surrogateCreation", exist_ok=True)

        # Log summary
        with open(self.log_file, "w") as f:
            f.write(f"Input data summary for '{self.label}'\n")
            f.write(f"{self.input_df.shape[1]} features after cleaning.\n")
            f.write(f"{self.input_df.shape[0]} total samples.\n")
            f.write(f"Total NaN values found: {nan_count}\n")
            if constant_indices:
                f.write(f"Constant columns removed at indices: {constant_indices}\n")
            else:
                f.write("No constant columns found.\n")
            f.write(f"Visualisations will use {'scaled' if self.scaled_vis else 'unscaled'} data.\n")

        print("Surrogate data loaded correctly.")
        print(f"Total NaN values: {nan_count}")
        print("Removed constant columns:", constant_indices)
        print(f"Visualisations will use {'scaled' if self.scaled_vis else 'unscaled'} data.")

    # ---------------------------------------------------------------------

    @contextmanager
    def _suppress_seaborn_warnings(self):
        """
        Context manager that locally suppresses noisy seaborn/pandas plotting
        warnings (UserWarning, FutureWarning) to avoid clutter from older
        seaborn/pandas versions.
        """
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module=r"seaborn.*")
            warnings.filterwarnings("ignore", category=FutureWarning, module=r"seaborn.*")
            warnings.filterwarnings("ignore", category=FutureWarning, module=r"pandas.*")
            yield

    # ---------------------------------------------------------------------

    def _df_for_visualisation(self) -> pd.DataFrame:
        """
        Return the DataFrame to be used for visualisations depending on scaled_vis.
        Replace +/- inf with NaN.
        """
        if self.scaled_vis:
            return pd.DataFrame(self.input_data_scaled).replace([np.inf, -np.inf], np.nan)
        return self.input_df.replace([np.inf, -np.inf], np.nan)

    # ---------------------------------------------------------------------

    def correlation_matrix(self) -> None:
        """
        Plot and save a correlation matrix heatmap for visualised data (scaled/unscaled).
        """
        df = self._df_for_visualisation()
        corr = df.corr()

        plt.figure(figsize=(12, 10))
        with self._suppress_seaborn_warnings():
            sns.heatmap(corr, cmap="coolwarm", annot=False)
        plt.title("Correlation Matrix")
        plt.tight_layout()

        out_file = os.path.join("surrogateCreation", f"{self.label}correlation_matrix.png")
        plt.savefig(out_file)
        plt.close()

        with open(self.log_file, "a") as f:
            f.write(f"Correlation matrix saved to {out_file}\n")

    # ---------------------------------------------------------------------

    def explanatory_dimension(self, explained_proportion: float = 0.95) -> np.ndarray:
        """
        Run PCA on the scaled data and return the reduced projection that
        explains 'explained_proportion' of variance.

        PCA is always performed on the scaled data for numerical stability.
        """
        pca = PCA()
        pca.fit(self.input_data_scaled)
        cum = np.cumsum(pca.explained_variance_ratio_)
        n = int(np.searchsorted(cum, explained_proportion) + 1)

        out_plot = os.path.join("surrogateCreation", f"{self.label}explained_variance.png")

        plt.figure(figsize=(10, 6))
        plt.plot(cum, marker="o")
        plt.axvline(n, color="r", linestyle="--")
        plt.title("Cumulative Explained Variance")
        plt.xlabel("Number of Components")
        plt.ylabel("Cumulative Variance")
        plt.tight_layout()
        plt.savefig(out_plot)
        plt.close()

        with open(self.log_file, "a") as f:
            f.write(f"PCA: Retaining {n} components for {explained_proportion:.2f} variance.\n")
            f.write(f"Saved explained variance plot to {out_plot}\n")

        return PCA(n_components=n).fit_transform(self.input_data_scaled)

    # ---------------------------------------------------------------------

    def feature_histograms(self,
                           cols: Optional[Sequence[int]] = None,
                           bins: int = 30,
                           sample: Optional[int] = 5000) -> None:
        """
        Plot histograms + KDE for selected features (visualised on scaled or unscaled).
        """
        df = self._df_for_visualisation()
        if sample is not None and df.shape[0] > sample:
            df = df.sample(sample, random_state=0)

        cols = cols or list(range(df.shape[1]))
        n = len(cols)
        ncols = 3
        nrows = int(np.ceil(n / ncols))

        plt.figure(figsize=(4 * ncols, 3 * nrows))
        for i, col in enumerate(cols):
            ax = plt.subplot(nrows, ncols, i + 1)
            with self._suppress_seaborn_warnings():
                sns.histplot(df[col], bins=bins, kde=True, stat="density")
            ax.set_title(f"Feature {col}")
        plt.tight_layout()

        out_file = os.path.join("surrogateCreation", f"{self.label}feature_histograms.png")
        plt.savefig(out_file)
        plt.close()

        with open(self.log_file, "a") as f:
            f.write(f"Feature histograms saved to {out_file}\n")

    # ---------------------------------------------------------------------

    def missing_value_report(self) -> pd.DataFrame:
        """
        Create a summary table of missing values for the unscaled input DataFrame.
        """
        df = self.input_df
        miss = df.isna().sum()
        pct = miss / len(df) * 100

        report = pd.DataFrame({
            "feature": miss.index,
            "n_missing": miss.values,
            "pct_missing": pct.values
        })

        out_file = os.path.join("surrogateCreation", f"{self.label}missing_value_report.csv")
        report.to_csv(out_file, index=False)

        with open(self.log_file, "a") as f:
            f.write(f"Missing value report saved to {out_file}\n")

        return report

    # ---------------------------------------------------------------------

    def pairplot(self, sample: int = 500, hue: Optional[int] = None) -> None:
        """
        Generate a seaborn pairplot for a random subset of features.
        Visualisation uses scaled data if scaled_vis is True, else unscaled.
        """
        df = self._df_for_visualisation()

        nrows = df.shape[0]
        sample_n = min(sample, nrows)
        sample_idx = df.sample(sample_n, random_state=0).index
        df_sample = df.loc[sample_idx].reset_index(drop=True)

        if hue is not None:
            if hue >= self.output_data.shape[1]:
                raise IndexError("Selected hue column is out of range for output_data")
            hue_series = pd.Series(self.output_data.iloc[:, hue]).iloc[sample_idx].reset_index(drop=True)
            df_sample["hue"] = hue_series
            with self._suppress_seaborn_warnings():
                grid = sns.pairplot(df_sample, hue="hue", corner=True)
        else:
            with self._suppress_seaborn_warnings():
                grid = sns.pairplot(df_sample, corner=True)

        out_file = os.path.join("surrogateCreation", f"{self.label}pairplot.png")
        try:
            grid.savefig(out_file)
        except Exception:
            plt.savefig(out_file)
        plt.close()

        with open(self.log_file, "a") as f:
            f.write(f"Wrote pairplot to {out_file}\n")

    # ---------------------------------------------------------------------

    def outlier_detection(self, method: str = "zscore", thresh: float = 3.0) -> np.ndarray:
        """
        Identify outlier rows. This method uses the scaled data (recommended).
        Returns a boolean mask where True indicates an outlier row.
        """
        df = pd.DataFrame(self.input_data_scaled)

        if method == "zscore":
            z = np.abs((df - df.mean()) / df.std(ddof=0))
            mask = (z > thresh).any(axis=1).values
        elif method == "iqr":
            Q1, Q3 = df.quantile(0.25), df.quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - thresh * IQR
            upper = Q3 + thresh * IQR
            mask = ((df < lower) | (df > upper)).any(axis=1).values
        else:
            raise ValueError("method must be 'zscore' or 'iqr'")

        idx = np.where(mask)[0]
        out_file = os.path.join("surrogateCreation", f"{self.label}outlier_indices.csv")
        pd.DataFrame({"outlier_index": idx}).to_csv(out_file, index=False)

        with open(self.log_file, "a") as f:
            f.write(f"Outliers detected ({len(idx)} rows). Saved to {out_file}\n")

        return mask

    # ---------------------------------------------------------------------

    def feature_importance_proxy(self,
                                 target_column: int = 0,
                                 n_estimators: int = 100,
                                 random_state: int = 0) -> pd.Series:
        """
        Quick proxy feature importance using RandomForest. Uses scaled data.
        """
        if target_column >= self.output_data.shape[1]:
            raise IndexError("target_column out of range.")

        X = pd.DataFrame(self.input_data_scaled)
        y = self.output_data.iloc[:, target_column].values

        n = min(len(X), len(y))
        X, y = X.iloc[:n, :], y[:n]

        model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
        model.fit(X, y)

        imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)

        out_file = os.path.join("surrogateCreation", f"{self.label}feature_importances.csv")
        imp.to_csv(out_file, header=["importance"])

        with open(self.log_file, "a") as f:
            f.write(f"Feature importance proxy saved to {out_file}\n")

        return imp

    # ---------------------------------------------------------------------

    def explain_variance_table(self) -> pd.DataFrame:
        """
        Return a detailed PCA variance breakdown table (computed on scaled data).
        """
        pca = PCA()
        pca.fit(self.input_data_scaled)
        ev = pca.explained_variance_ratio_
        cum = np.cumsum(ev)

        df = pd.DataFrame({
            "component": np.arange(1, len(ev) + 1),
            "explained_variance": ev,
            "cumulative_variance": cum
        })

        out_file = os.path.join("surrogateCreation", f"{self.label}explained_variance_table.csv")
        df.to_csv(out_file, index=False)

        with open(self.log_file, "a") as f:
            f.write(f"PCA variance table saved to {out_file}\n")

        return df
