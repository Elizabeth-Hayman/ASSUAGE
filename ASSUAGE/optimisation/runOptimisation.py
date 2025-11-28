import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import os
import numpy as np
import json
import time
import datetime

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

from fitness import stressCatheter
timeNow = datetime.datetime.now()
seed = 30
np.random.seed(seed)

from distributed import Client, LocalCluster

cluster = LocalCluster(threads_per_worker=2, # change this number
                       processes=False)
cluster.scale(1) # leave this number

with Client(cluster) as client:

    optimiser = MOSOptimiser(
            #starting_policy={SHADE('SHADE_1', elitism_operator=PairwiseElitism(use_surrogate_models=True)): 1.0},
            starting_policy={MTS('MTS_1', elitism_operator=PairwiseElitism(use_surrogate_models=True)): 1.0},
            num_steps=1
        )

    logFolder = f'{timeNow.strftime("%m-%d-%H:%M")}-Logs'
    os.system(f"mkdir {logFolder}")
    problem = stressCatheter(numRings=6, logFolder=logFolder)
    problem.configure(dimension=24)

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
        fitness_executor=DaskFitnessModule(client, FFEBudgetCounter(1000)),
        optimiser=optimiser,
        surrogates_manager=surrogate_manager,
        output_folder=Path(logFolder),
        seed = seed
    )

    result = op.solve(population_size=15)
    algorithm_info = op.optimiser.tracking_info

time.sleep(.2)
print("SUCCESSFUL OPTIMISATION FINISH")
result = json.load(open(f"{logFolder}/Result.json",))
values = result["values"]


print("Run in time: ", str((datetime.datetime.now() - timeNow)) )