import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from typing import Callable, List, Optional, Tuple

import os
import numpy as np
import json
import time
import datetime

from distributed import Client, LocalCluster

from lurtis_eoe.Optimisers.Algorithms.SHADE import SHADE
from lurtis_eoe.Optimisers.Algorithms.MTS_LS import MTS
from lurtis_eoe.Optimisers.Algorithms.Operators.Elitism.PairwiseElitism import PairwiseElitism
from lurtis_eoe.Surrogates.SurrogateManager import SurrogateManager
from lurtis_eoe.Surrogates.Models.Classifiers import DecisionTreeClassifier
from lurtis_eoe.Surrogates.Models.Regressors import XGBoostRegressor
from lurtis_eoe.OptimisationProcess import OptimisationProcess
from lurtis_eoe.Optimisers.MOS.MOS import MOSOptimiser
from lurtis_eoe.Fitness.BudgetCounter import FFEBudgetCounter
from lurtis_eoe.Fitness.FitnessModule import DaskFitnessModule

from ASSUAGE.optimisation.optimisation_fitness import Fitness


class runOptimisation():
    def __init__(self, 
                
                parameter_to_model: Callable,
                extract_surrogate_func: Callable,
                surrogate_template_folder: str,
                bounds: Tuple[List[float], List[float]],
                parameter_names: List[str],
                preprocess_func: Optional[Callable] = None,
                surrogate_file: Optional[str] = None
                ):
        self.parameter_to_model = parameter_to_model
        self.preprocess_func = preprocess_func
        self.extract_surrogate_func = extract_surrogate_func
        self.surrogate_template_folder = surrogate_template_folder
        self.bounds = bounds
        self.parameter_names = parameter_names
        self.surrogate_file = surrogate_file



    def run_opt_realisation(self, seed,
                            simulation_folder,
                            clean_dir: bool = True,
                            n_jobs: int = 1,
                            num_steps = 1,
                            budget: int = 1000, 
                            pop_size = 15):

        timeNow = datetime.datetime.now()
        log_folder = f'{timeNow.strftime("%m-%d-%H:%M")}-Logs'
        os.system(f"mkdir {log_folder}; rm -rf {simulation_folder}; mkdir {simulation_folder}")
        problem = Fitness(self.parameter_to_model, self.extract_surrogate_func, self.surrogate_template_folder,
                            bounds=self.bounds, parameter_names=self.parameter_names, preprocess_func=self.preprocess_func,
                            simulation_folder=simulation_folder, clean_dir=clean_dir, log_folder=log_folder, surrogate_file=self.surrogate_file)
        np.random.seed(seed)
        cluster = LocalCluster(threads_per_worker=n_jobs, processes=False)
        cluster.scale(1) # leave this number

        with Client(cluster) as client:

            optimiser = MOSOptimiser(
                    starting_policy={SHADE('SHADE_1', elitism_operator=PairwiseElitism(use_surrogate_models=True)): 0.5,
                                    MTS('MTS_1', elitism_operator=PairwiseElitism(use_surrogate_models=True)): 0.5},
                    num_steps=num_steps
                )

            surrogate_manager = SurrogateManager(
                warm_up=30, 
                trail_size=45,
                models=[XGBoostRegressor(), DecisionTreeClassifier()],
                strategies = [],
                training_schedule= 'step',
                dask_client = client
            )

            op = OptimisationProcess(
                fitness_function=problem,
                fitness_executor=DaskFitnessModule(client, FFEBudgetCounter(budget)),
                optimiser=optimiser,
                surrogates_manager=surrogate_manager,
                output_folder=Path(log_folder),
                seed = seed
            )

            result = op.solve(population_size=pop_size)
            algorithm_info = op.optimiser.tracking_info

        time.sleep(.2)
        print("SUCCESSFUL OPTIMISATION FINISH")
        result = json.load(open(f"{log_folder}/Result.json",))
        values = result["values"]
        print(values)

        print("Run in time: ", str((datetime.datetime.now() - timeNow)) )