import os
import pickle
import subprocess
from typing import Callable, List, Optional, Tuple

import numpy as np
from lurtis_eoe.Fitness.FitnessFunction import StatefullFitnessFunction, Solution



class Fitness(StatefullFitnessFunction):
    def __init__(
        self,
        parameter_to_model: Callable,
        extract_surrogate_func: Callable,
        surrogate_template_folder: str,
        bounds: Tuple[List[float], List[float]],
        parameter_names: List[str],
        preprocess_func: Optional[Callable] = None,
        surrogate_file: Optional[str] = None,
        simulation_folder: Optional[str] = None,
        log_folder: str = "."
    ):
        """
        A fitness function wrapper for use with surrogate-based evaluation.

        Args:
            parameter_to_model: Function to transform input parameters into model setup.
            extract_surrogate_func: Function to extract features from simulation folder.
            surrogate_template_folder: Folder containing simulation template.
            bounds: Tuple of (lower_bounds, upper_bounds).
            parameter_names: List of parameter names.
            preprocess_func: Optional preprocessing function (discard_flag, values) = func(values).
            surrogate_file: Optional path to a pickled surrogate model.
            simulation_folder: Where to store simulations.
            log_folder: Where to save logs.
        """
        self.template_folder = surrogate_template_folder
        self.parameter_to_model = parameter_to_model
        self.extract_surrogate_func = extract_surrogate_func
        self.sim_folder = simulation_folder or os.path.join(os.getcwd(), "simulations")
        self.parameter_names = parameter_names
        self.preprocess_func = preprocess_func or (lambda values: (False, values))

        os.makedirs(self.sim_folder, exist_ok=True)

        surrogate_file = surrogate_file or os.path.join(
            os.getcwd(), "surrogateCreation", "best_model_pipeline.pkl"
        )
        if not os.path.exists(surrogate_file):
            raise FileNotFoundError(f"No pickle file found at {surrogate_file}")

        with open(surrogate_file, "rb") as f:
            self.surrogate = pickle.load(f)

        # Call base class constructor
        lower_bounds, upper_bounds = bounds
        super().__init__(np.array(lower_bounds), np.array(upper_bounds), len(lower_bounds))

        # Log files
        self.disp_file = os.path.join(log_folder, "fitness.csv")
        self.discard_params_file = os.path.join(log_folder, "discarded_parameters.csv")

        for file_path in [self.disp_file, self.discard_params_file]:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                with open(file_path, "w") as f:
                    if "fitness" in file_path:
                        f.write("runID,fitness," +",".join(parameter_names) + "\n")
                    else:
                        f.write("runID," + ",".join(parameter_names) + "\n")
            except Exception as e:
                raise IOError(f"Error setting up log file {file_path}: {e}")

    def fitness(self, values: np.ndarray, id: int) -> Solution:
        """
        Compute fitness for given parameter values.

        Args:
            values: The parameter values to evaluate.
            run_id: Unique run identifier.

        Returns:
            A Solution object with prediction.
        """
        print(f"Starting fitness evaluation at id = {id}")
        original_values = values.copy()

        run_folder = os.path.join(self.sim_folder, f"sim{id}")
        try:
            subprocess.run(["cp", "-r", self.template_folder, run_folder], check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Error copying simulation folder: {e}")

        # Save input parameters
        param_file = os.path.join(run_folder, "design_parameters.csv")
        np.savetxt(param_file, values.reshape(1, -1), delimiter=',', fmt='%.3f')

        # Run model and extract surrogate features
        self.parameter_to_model(values, run_folder, alpha=0)
        print(self.extract_surrogate_func(run_folder))
        extracted_data = self.extract_surrogate_func(run_folder).reshape(1, -1)
        prediction = self.surrogate.predict(extracted_data)

        # Clean up simulation directory
        try:
            subprocess.run(["rm", "-rf", run_folder], check=True)
        except subprocess.CalledProcessError:
            print(f"Warning: failed to remove folder {run_folder}")

        # Log output
        with open(self.disp_file, "a") as f:
            f.write(f"{id},{prediction[0]}," + ",".join(map(str, values)) + "\n")

        return Solution(id, prediction, proposed_genome=original_values, canonical_genome=np.array(values))

    def preprocess(self, values: np.ndarray) -> Tuple[bool, np.ndarray]:
        """
        Preprocess input parameter set before evaluation.

        Args:
            values: Raw parameter values.

        Returns:
            discard (bool), possibly transformed values (np.ndarray).
        """
        discard, processed_values = self.preprocess_func(values)
        return discard, processed_values

