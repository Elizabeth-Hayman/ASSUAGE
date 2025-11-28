# Import all necessary packages
# Parameters introduced by radius, zPos, number, offset.
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pickle
import numpy as np
import os

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
from ASSUAGE.createGroundTruth import start_new_runs, extractData
os.system(f"rm -rf groundTruth surrogateCreation")
start_new_runs(numNewRuns, FSItemplateFolder, lowerBounds, upperBounds, numCoresPerSim, numCores, parameter_to_model, preprocess_func)
from ASSUAGE.createGroundTruth import extractData
extractData(extract_surrogate_inputs, extract_fitness)"""

"""
from ASSUAGE.surrogateModel.dataExploration import DataExploration
explorer = DataExploration("surrogateCreation/trainingInput.csv","surrogateCreation/trainingOutput.csv", scaled_vis=False)
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
"""



from ASSUAGE.surrogateModel.mlModelFitting import mlModels


modeler = mlModels(
    input_data="surrogateCreation/trainingInput.csv",
    output_data="surrogateCreation/trainingOutput.csv",
    data_label="demo",
    output_dir="surrogateCreation",
    pca=True,            # no PCA to keep things simple
    search_type="random",
    n_iter=15,            # small n_iter for quick demo
    n_jobs=10,            # 1 to avoid parallel issues in demo
    seed=42,
    best_model_cutoff=0.95
)

# --- Run a quick holdout evaluation ---
print("\nRunning evaluate_holdout (quick demo)...")
results = modeler.evaluate_nested_cv(outer_splits=5, inner_splits=5, create_plots=True)

print("\nLive best model recorded in the object:")
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

print("\nDemo finished. All outputs written into 'surrogateCreation' folder.")

"""
from ASSUAGE.optimisation.optFitness import Fitness
f = Fitness(parameter_to_model, extract_surrogate_inputs, "exampleTemplates/reducedModelTemplate",
        bounds=  (lowerBounds, upperBounds), parameter_names=paramNames, preprocess_func=preprocess_parameters,
        simulation_folder="SimulationTest")


print(f.fitness(np.random.random(size=(24)), id=0).fitness)"""

"""
ML_model_pickle = ""
assert os.path.exists(ML_model_pickle), f"No pickle file found at {ML_model_pickle}, check file paths"
with open(ML_model_pickle, 'rb') as f: 
    ml_model = pickle.load(f)
imFile = os.path.join("surrogateCreation", f"initialML")


fig, ax = plt.subplots(4,1,figsize=(40,40))
for i, a in enumerate(ax.flatten()):
    sc = a.imshow(ml_model.feature_importances_[1:].reshape(4,99,9)[i,::-1,::-1].T, "Greys", vmax=0.015)
    divider = make_axes_locatable(a)
    cax = divider.append_axes("bottom", size="30%", pad=2.)  
    cb = fig.colorbar(sc,orientation='horizontal', cax=cax)
    a.set_xticks(np.linspace(0,100,7))
    a.set_xticklabels(np.linspace(0,30,7))
    a.set_yticks(np.linspace(0,10,8))
    a.set_yticklabels([3.5,3,"",2,"",1,"",0])
    a.set_ylabel("$y$ (mm)")
    a.set_xlabel("$z$ (mm)")
ax.flatten()[0].set_title("Pressure")
ax.flatten()[1].set_title("Stress vector, $x$ component")
ax.flatten()[2].set_title("Stress vector, $y$ component")
ax.flatten()[3].set_title("Stress vector, $z$ component")
fig.tight_layout()
fig.savefig(os.path.join(imFile, "featureImportance.png"))
plt.close()
"""


