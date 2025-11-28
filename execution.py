# Import all necessary packages

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pickle
import numpy as np
import os
from ASSUAGE.surrogateModel.ml_plotter import feature_importance
from encoding import *

lowerBounds = [0.05] * 6 + [0.2] * 6 + [-0.5] * 6 + [0] * 6
upperBounds = [0.5] * 6 + [10] * 6 + [10.5] * 6 + [360] * 6
paramNames = [f"radius{i}" for i in range(6)]+[f"zPos{i}" for i in range(6)]+ [f"number{i}" for i in range(6)]+[f"offset{i}" for i in range(6)]

numNewRuns = 150
FSItemplateFolder = "exampleTemplates/fullModelTemplate"


numCoresPerSim = 1
numCores = 10
assert len(lowerBounds) == len(upperBounds), "Upper and lower bound lists must have the same length."

templateFolder = os.path.join(os.getcwd(),FSItemplateFolder)

try:
    from encoding import preprocess_parameters as preprocess_func
except:
    print("No preprocess_parameters function found in encoding file")
    preprocess_func = None

try:
    from encoding import parameter_to_model
except:
    print("No parameter_to_model function found in encoding file. This is a necessary function!!")

"""
## Create ground truth data set
from ASSUAGE.create_ground_truth import start_new_runs, extractData
os.system(f"rm -rf groundTruth surrogateCreation")
start_new_runs(numNewRuns, FSItemplateFolder, lowerBounds, upperBounds, numCoresPerSim, numCores, parameter_to_model, preprocess_func)
from ASSUAGE.create_ground_truth import extractData
extractData(extract_surrogate_inputs, extract_fitness)"""

"""
from ASSUAGE.surrogateModel.data_exploration import data_exploration
explorer = data_exploration("surrogateCreation/trainingInput.csv","surrogateCreation/trainingOutput.csv", scaled_vis=False)
explorer.correlation_matrix()
explorer.feature_histograms()
missing_report = explorer.missing_value_report()
reduced_data = explorer.explanatory_dimension(0.95) # PCA dimensionality reduction to preserve 95% variance
variance_table = explorer.explain_variance_table()
explorer.pairplot(sample=150)
explorer.pairplot(sample=150, hue=0)
outlier_mask = explorer.outlier_detection(method="zscore", thresh=3.0)
importances = explorer.feature_importance_proxy(target_column=0)
print("\nAll demonstration outputs written to 'surrogateCreation/'.")




from ASSUAGE.surrogateModel.ml_models import mlModels

modeler = mlModels(
    input_data="surrogateCreation/trainingInput.csv",
    output_data="surrogateCreation/trainingOutput.csv",
    data_label="demo",
    output_dir="surrogateCreation",
    pca=False,            # no PCA to keep things simple
    search_type="random",
    n_iter=5,            # small n_iter for quick demo
    n_jobs=10,            # 1 to avoid parallel issues in demo
    seed=42,
    best_model_cutoff=0.95
)

# --- Run a quick holdout evaluation ---
print("\nRunning evaluate_holdout (quick demo)...")
results = modeler.evaluate_holdout(test_size=0.1, inner_cv=5, create_plots=True)
#results = modeler.evaluate_nested_cv(outer_splits=5, inner_splits=5, create_plots=True)

print("\Best model recorded in the object:")
print(" best_model_name:", modeler.best_model_name)
print(" best_model_score:", modeler.best_model_score)
print(" best_model_info:")
print(modeler.best_model_info)

# --- Train best pipeline (auto-selected) and save with pickle (the class uses pickle) ---
print("\nRunning train_best_pipeline() (uses recorded best model)...")
pipeline = modeler.train_best_pipeline(cv=5)  # will save pipeline to output_dir

print("\nPipeline saved to:", modeler.best_pipeline_path)
print("Best pipeline object type:", type(pipeline))

# --- Load the saved pipeline via pickle and predict on first 5 rows ---
with open(modeler.best_pipeline_path, "rb") as fh:
    loaded_pipe = pickle.load(fh)

sample_X = pd.read_csv("surrogateCreation/trainingInput.csv", header=None).iloc[:5].to_numpy()
preds = loaded_pipe.predict(sample_X)
print("\nPredictions on first 5 samples (loaded pipeline):")
print(preds)

feature_importance(loaded_pipe, output_dir=modeler.output_dir, X=modeler.X, y= modeler.y)

print("\nDemo finished. All outputs written into 'surrogateCreation' folder.")



from ASSUAGE.optimisation.optimisation_fitness import Fitness
f = Fitness(parameter_to_model, extract_surrogate_inputs, "exampleTemplates/reducedModelTemplate",
        bounds=  (lowerBounds, upperBounds), parameter_names=paramNames, preprocess_func=preprocess_parameters,
        simulation_folder="SimulationTest", clean_dir=False)
print(f.fitness(np.random.random(size=(24)), id=0).fitness)
print(f.fitness(np.random.random(size=(24)), id=1).fitness)
"""


from ASSUAGE.optimisation.run_optimisation import runOptimisation

ro = runOptimisation(parameter_to_model=parameter_to_model, extract_surrogate_func=extract_surrogate_inputs, 
                     surrogate_template_folder="exampleTemplates/reducedModelTemplate", surrogate_file="surrogateCreation/best_pipeline.pkl",
                     bounds=  (lowerBounds, upperBounds), parameter_names=paramNames, preprocess_func=preprocess_parameters)

ro.run_opt_realisation(seed=42, budget=100, n_jobs=10, simulation_folder="run_optimisation_test", clean_dir=False)
