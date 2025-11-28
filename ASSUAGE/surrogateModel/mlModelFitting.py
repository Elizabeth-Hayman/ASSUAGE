import json
import os
import time
from typing import Dict, List, Optional, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from scipy.stats import spearmanr
from sklearn.base import RegressorMixin
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import BayesianRidge, Lasso, LinearRegression, Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    make_scorer,
    r2_score
)
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_validate, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

# Set consistent plot style
matplotlib.rcParams.update({"font.size": 25})


class mlModels:
    def __init__(self, input_data: Union[str, os.PathLike],
        output_data: Union[str, os.PathLike], model_dict: Optional[Dict[str, RegressorMixin]] = None,
        data_label: str = "", pca = False, seed: int = 42) -> None:
        """
        Initialize the mlModels class.

        This sets up input/output data, normalizes input features, and stores 
        the model dictionary for use in later steps. If no model dictionary is provided, 
        a default suite of regression models is used. The model dictionary should have 
        entries of the form model_name : model_instance

        Parameters:
            input_data (Union[str, os.PathLike]): Path to input feature CSV.
            output_data (Union[str, os.PathLike]): Path to output target CSV.
            model_dict (Optional[Dict[str, RegressorMixin]]): Optional custom model dictionary.
            data_label (str): Label for identifying outputs from this model run.
            pca (bool): Whether to apply pca for dimensionality reduction (retain 95% variance).
            seed (int): Random seed for reproducibility across models and pca.
        """
        self.label = data_label
        self.seed = seed
        self.output_dir = "surrogateCreation"
        self.pca=pca

        # Default model dictionary if none is given
        self.model_dict = model_dict or {
            'Linear Regression': LinearRegression(),
            'Support Vector Regressor': SVR(),
            'Ridge': Ridge(),
            'Lasso': Lasso(),
            'Bayesian Ridge': BayesianRidge(),
            'Decision Tree': DecisionTreeRegressor(),
            'Random Forest': RandomForestRegressor(),
            'XGBoost': XGBRegressor(objective='reg:squarederror'),
            'Gradient Boost': GradientBoostingRegressor()
        }

        # Load and scale input data
        X = pd.read_csv(input_data, header=None).to_numpy()
        self.y = pd.read_csv(output_data, header=None).to_numpy().reshape(-1)
        self.X = StandardScaler().fit_transform(X)

    def select_promising_ml_models(self, accuracy_cutoff: float = 0.6) -> Optional[List[str]]:
        """
        Evaluate a suite of ML regression models and identify those with strong performance.

        This method applies cross-validation to each model and calculates R^2, MAE, MSE,
        and Spearman correlation. It optionally applies pca to reduce input dimensionality.
        A heatmap of scores is saved, and well-performing models (based on R^2) are selected.

        Parameters:
            accuracy_cutoff (float): Minimum R^2 value required for a model to be selected.

        Returns:
            Optional[List[str]]: List of model names passing the R^2 cutoff; None if none qualify.
        """
        print("\n----------------------------\n")
        print(f"Evaluating models for input data: {self.label}")

        results_path = os.path.join(self.output_dir, "model_performance.txt")

        with open(results_path, "w") as f:
            f.write("model_name,ave_r2,ave_mae,ave_mse,ave_rho,"
                    "std_r2,std_mae,std_mse,std_rho,time\n")

        # Apply pca if requested
        X_processed = self.X
        if self.pca:
            full_pca = PCA(random_state=self.seed)
            explained = np.cumsum(full_pca.fit(self.X).explained_variance_ratio_)
            n_components = np.argmax(explained >= 0.95) + 1
            X_processed = PCA(n_components=n_components, random_state=self.seed).fit_transform(self.X)

        # Custom Spearman scorer
        def spearman_scorer(y_true, y_pred) -> float:
            return spearmanr(y_true, y_pred).correlation

        # Define cross-validation and scoring metrics
        kf = KFold(n_splits=5, shuffle=True, random_state=self.seed)
        scoring = {
            'r2': 'r2',
            'mae': 'neg_mean_absolute_error',
            'mse': 'neg_mean_squared_error',
            'sr': make_scorer(spearman_scorer, greater_is_better=True)
        }

        good_models = []
        num_metrics = len(scoring.keys())
        grid_scores = np.zeros((num_metrics, len(self.model_dict)))

        for idx, (name, model) in enumerate(self.model_dict.items()):
            print(f"Fitting model: {name}")
            start = time.time()

            results = pd.DataFrame(cross_validate(model, X_processed, self.y, cv=kf, scoring=scoring))
            runtime = time.time() - start

            # Convert scores where necessary
            results["test_mae"] *= -1
            results["test_mse"] *= -1

            means = results.mean()[2:].values
            stds = results.std()[2:].values

            if means[0] > accuracy_cutoff:
                good_models.append(name)

            with open(results_path, "a") as f:
                f.write(f"{name}," +
                        ",".join([f"{v:.3f}" for v in means]) + "," +
                        ",".join([f"{v:.3f}" for v in stds]) + f",{runtime:.2f}\n")

            grid_scores[:, idx] = np.clip(means, 0, 1)

        # Generate heatmap
        fig, ax = plt.subplots(figsize=(20, 15))
        im = ax.imshow(grid_scores, cmap='viridis', interpolation='nearest')
        plt.colorbar(im, ax=ax, label='Score')

        ax.set_xticks(np.arange(len(self.model_dict)))
        ax.set_xticklabels(self.model_dict.keys(), rotation=45, ha='right')
        ax.set_yticks(np.arange(num_metrics))
        ax.set_yticklabels([r"$R^2$", "MAE", "MSE", r"$\rho$"])

        for i in range(num_metrics):
            for j in range(len(self.model_dict)):
                ax.text(j, i, f"{grid_scores[i, j]:.2f}",
                        ha='center', va='center', color='white')

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "ML_model_accuracy_score.png"))

        if not good_models:
            print(f"No models exceeded R^2 > {accuracy_cutoff}")
            return None

        print("\nModels exceeding R^2 threshold:")
        for model in good_models:
            print(f"- {model}")
        return good_models

    def hyperparameter_optimisation(self, good_models: List[str]) -> None:
        """
        Optimize hyperparameters for a set of previously selected models.

        Performs randomized search cross-validation using predefined parameter grids
        for each model. Each model is evaluated on a held-out validation set, and
        its performance metrics (R^2, MAE, MSE) are plotted and saved.

        Parameters:
            good_models (List[str]): List of model names selected from prior evaluation.

        Returns:
            None. Outputs are saved to disk including performance plots and a JSON of results.
        """
        if not self.model_dict:
            raise ValueError("Model dictionary not found.")


        print("\n----------------------------\n")
        # Reproducible train/test split
        kf = KFold(n_splits=10, shuffle=True, random_state=self.seed)
        train_index, val_index = list(kf.split(self.X, self.y))[0]
        X_train, X_val = self.X[train_index], self.X[val_index]
        y_train, y_val = self.y[train_index], self.y[val_index]

        # Prefix model parameters for pipeline compatibility
        def prefixed(params: Dict[str, List]) -> Dict[str, List]:
            return {f"model__{k}": v for k, v in params.items()}

        # Define hyperparameter grid for each model
        param_grid = {
            'Linear Regression': prefixed({'fit_intercept': [True, False], 'positive': [True, False]}),
            'Support Vector Regressor': prefixed({
                'kernel': ['rbf', 'linear', 'poly'],
                'C': [0.1, 1, 10, 100],
                'epsilon': [0.01, 0.1, 0.5],
                'gamma': ['scale', 'auto']
            }),
            'Ridge': prefixed({
                'alpha': [0.01, 0.1, 1.0, 10.0, 100.0],
                'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sag'],
                'fit_intercept': [True, False]
            }),
            'Lasso': prefixed({
                'alpha': [0.001, 0.01, 0.1, 1.0, 10.0],
                'selection': ['cyclic', 'random'],
                'fit_intercept': [True, False],
                'max_iter': [1000, 5000, 10000]
            }),
            'Bayesian Ridge': prefixed({
                'alpha_1': [1e-6, 1e-4, 1e-2],
                'alpha_2': [1e-6, 1e-4, 1e-2],
                'lambda_1': [1e-6, 1e-4, 1e-2],
                'lambda_2': [1e-6, 1e-4, 1e-2],
                'fit_intercept': [True, False]
            }),
            'Decision Tree': prefixed({
                'max_depth': [None, 5, 10, 20],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None]
            }),
            'Random Forest': prefixed({
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2']
            }),
            'XGBoost': prefixed({
                'n_estimators': [100, 200],
                'max_depth': [3, 6, 10],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.6, 0.8, 1.0],
                'colsample_bytree': [0.6, 0.8, 1.0]
            }),
            'Gradient Boost': prefixed({
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 10],
                'subsample': [0.6, 0.8, 1.0],
                'max_features': ['sqrt', 'log2']
            })
        }

        results = {}
        fig, axes = plt.subplots(len(good_models), 1, figsize=(10, 5 * len(good_models)))
        if len(good_models) == 1:
            axes = [axes]

        for j, model_name in enumerate(good_models):
            print(f"Optimising hyperparameters for {model_name}")
            model = self.model_dict[model_name]
            params = param_grid[model_name]

            steps = [('scaler', StandardScaler())]
            if self.pca:
                steps.append(('pca', PCA(n_components=0.95, random_state=self.seed)))
            steps.append(('model', model))
            pipe = Pipeline(steps)

            search = GridSearchCV(estimator=pipe,param_grid=params,cv=5,scoring='r2',n_jobs=-1)
            search.fit(X_train, y_train)
            final_model = search.best_estimator_
            y_pred = final_model.predict(X_val)

            # Metrics
            r2 = round(r2_score(y_val, y_pred), 3)
            mse = round(mean_squared_error(y_val, y_pred), 3)
            mae = round(mean_absolute_error(y_val, y_pred), 3)
            rho = round(spearmanr(y_val, y_pred).correlation, 3)

            results[model_name] = {
                'best_params': {k.replace("model__", ""): v for k, v in search.best_params_.items()},
                'test_r2': r2,
                'test_mse': mse,
                'test_mae': mae,
                'test_rho': rho
            }

            # Plotting
            ax = axes[j]
            ax.scatter(y_pred, y_val, label=f"$R^2$ = {r2}", s=60)
            ax.plot([0, 1], [0, 1], "r--", linewidth=2)
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            ax.set_title(model_name)
            ax.legend()

            print(f"{model_name}: R2 = {r2}, MSE = {mse}, MAE = {mae}, Rho = {rho}")

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"optimal_hyperparams{self.label}.png"))

        self.opt_hyp_file = os.path.join(self.output_dir, f"optimised_hyperparameter_results{self.label}.json")


        results = {k.replace("model__", ""): v for k, v in results.items()}
        with open(self.opt_hyp_file, "w") as f:
            json.dump(results, f, indent=2)

    def train_best_model(self, optimal_hyperparams_file: str = None):
        """Take the best hyperparameter combination and train the final model
        """
        try:
            opt_hyper_results = json.loads(open(optimal_hyperparams_file, "r").read())
        except:
            try:
                opt_hyper_results = json.loads(open(self.opt_hyp_file, "r").read())
            except:
                print(f"No optimal hyperparam file found, check file paths exists.")
        

        print("\n----------------------------\n")
        best_model_name = max(opt_hyper_results, key=lambda k: opt_hyper_results[k]["test_r2"])
        best_info = opt_hyper_results[best_model_name]
        best_params = best_info["best_params"]

        print(f"Best model: {best_model_name} (R² = {best_info['test_r2']})")

        # Extract raw model kwargs (remove 'model__' prefix)
        #model_kwargs = {k.replace("model__", ""): v for k, v in best_params.items()}
        model = self.model_dict[best_model_name].set_params(**best_params)

        steps = [('scaler', StandardScaler())]
        if self.pca:
            steps.append(('pca', PCA(n_components=0.95, random_state=self.seed)))
        steps.append(('model', model))
        pipe = Pipeline(steps)
        pipe.fit(self.X, self.y)
        with open(os.path.join(self.output_dir,"best_model_pipeline.pkl"), "wb") as f: pickle.dump(pipe, f)


        y_pred = pipe.predict(self.X)
        r2 = round(r2_score(self.y, y_pred), 3)
        mse = round(mean_squared_error(self.y, y_pred), 3)
        mae = round(mean_absolute_error(self.y, y_pred), 3)
        rho = round(spearmanr(self.y, y_pred).correlation, 3)
        print(f"Best model: R2 = {r2}, MSE = {mse}, MAE = {mae}, Rho = {rho}")
        ## Plot a figure with
        fig, ax = plt.subplots(figsize = (20, 15))
        ax.scatter(pipe.predict(self.X), self.y, s=60)
        ax.plot([0, 1], [0, 1], "r--", linewidth=2)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        fig.tight_layout()
        fig.savefig(os.path.join(self.output_dir, f"best_model{self.label}.png"))
        return pipe



if __name__=='__main__':

    m = mlModels("surrogateCreation/trainingInput.csv","surrogateCreation/trainingOutput.csv" )
    good_models = m.select_promising_ml_models( accuracy_cutoff=0)
    m.hyperparameter_optimisation(good_models)



