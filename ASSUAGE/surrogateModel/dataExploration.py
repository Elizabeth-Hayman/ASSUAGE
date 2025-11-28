# -------------------------------------
# Code to investigate data and give some summary statistics
# -------------------------------------

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from typing import Union


class DataExploration():
    """
    Class for performing basic data exploration tasks on input/output data,
    including correlation plotting and PCA dimensionality reduction.
    """

    def __init__(self, input_data_path: str, output_data_path: str, label: str = "") -> None:
        """
        Initialize by loading input/output data from CSV files and applying standard scaling.
        """
        self.label = label
        self.log_file = os.path.join("surrogateCreation", f"{label}logFile.txt")

        try:
            # Read input and output data
            self.input_data = pd.read_csv(input_data_path, header=None)
            self.output_data = pd.read_csv(output_data_path, header=None)
            print(f"Surrogate data loaded correctly from 'surrogateCreation' folder.")
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Cannot find input or output data files: {e}. Check that the paths exist."
            )

        nan_count = self.input_data.isna().sum().sum()
        constant_cols = self.input_data.nunique() == 1
        constant_col_indices = constant_cols[constant_cols].index.tolist()
        self.input_data.drop(columns=constant_col_indices, inplace=True)
        self.input_data.to_csv(input_data_path, index=False, header=None)

        # Apply standard scaling
        print(f"Applying StandardScaler to input data '{label}'")
        self.scaler = StandardScaler()
        self.input_data = self.scaler.fit_transform(self.input_data)

        # Prepare log directory and write logs
        os.makedirs("surrogateCreation", exist_ok=True)
        with open(self.log_file, "w") as f:
            f.write(f"Looking at input data {label}\n")
            f.write(f"{self.input_data.shape[1]} features and {self.input_data.shape[0]} datapoints.\n")
            f.write(f"Total NaN values in input data: {nan_count}\n")
            if constant_col_indices:
                f.write(f"Constant columns found at indices: {constant_col_indices}\n")
            else:
                f.write("No constant columns found in input data.\n")

        # Also print checks to console
        print(f"Total NaN values in input data: {nan_count}")
        if len(constant_col_indices) < 50:
            print(f"Constant columns found at indices: {constant_col_indices}")
        elif constant_col_indices: 
            print("Many constant columns found and removed")
        else:
            print("No constant columns found in input data.")

    def correlation_matrix(self) -> None:
        """
        Plot and save a correlation matrix heatmap for the scaled input data.
        """
        df = pd.DataFrame(self.input_data)  # Convert back to DataFrame to use .corr()
        corr_matrix = df.corr()

        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, cmap='coolwarm', annot=False)
        plt.title("Correlation Matrix")
        plt.tight_layout()
        plt.savefig(os.path.join("surrogateCreation", f"{self.label}correlation_matrix.png"))
        plt.close()

    def explanatory_dimension(self, explained_proportion: float = 0.95) -> np.ndarray:
        """
        Perform PCA to determine how many components are required to explain a target
        proportion of variance, and return the reduced dataset.
        """
        pca = PCA()
        pca.fit(self.input_data)
        explained_variance = np.cumsum(pca.explained_variance_ratio_)

        n_components = np.searchsorted(explained_variance, explained_proportion) + 1

        print(f"Input data '{self.label}': Components to retain {explained_proportion:.0%} variance: {n_components}")
        with open(self.log_file, "a") as f:
            f.write(f"Number of components to retain {explained_proportion:.0%} variance: {n_components}\n")

        # Plot explained variance
        plt.figure(figsize=(10, 6))
        plt.plot(explained_variance, marker='o', label='Cumulative Explained Variance')
        plt.axvline(n_components, color='r', linestyle='--',
                    label=f'{n_components} components ({int(explained_proportion * 100)}% variance)')
        plt.xlabel('Number of Principal Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title('Explained Variance by Principal Components')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join("surrogateCreation", f"{self.label}explained_variance.png"))
        plt.close()
        print("Figure written to folder surrogateCreation")

    
