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

os.system(f"rm -rf groundTruth surrogateCreation")

numNewRuns = 100
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

## Create ground truth data set
from ASSUAGE.createGroundTruth import start_new_runs, extractData
start_new_runs(numNewRuns, FSItemplateFolder, lowerBounds, upperBounds, numCoresPerSim, numCores, parameter_to_model, preprocess_func)
from ASSUAGE.createGroundTruth import extractData
extractData(extract_surrogate_inputs, extract_fitness)

"""
## Just for the test remove all the zero columns from the input data
df = pd.read_csv("surrogateCreation/trainingInput.csv", low_memory=False)
df = df.loc[:, (df != 0).any(axis=0)]
#df.to_csv("surrogateCreation/trainingInput.csv", index=False)

plt.scatter(df.iloc[:,0], pd.read_csv("surrogateCreation/trainingOutput.csv").iloc[:,0], label="Ground truth data")
xx = np.sort(df.iloc[:,0].values)
plt.plot(xx, 2*xx*xx+3/2*xx + 0.05, "r",label="True relation")
plt.xlabel("First radius")
plt.legend()
plt.ylabel("Quadratic fitness relation")
plt.savefig("surrogateCreation/trueTestDataRelation.png")


from ASSUAGE.surrogateModel.mlModelFitting import mlModels
from ASSUAGE.surrogateModel.dataExploration import DataExploration

#d = DataExploration("surrogateCreation/trainingInput.csv","surrogateCreation/trainingOutput.csv")
#d.correlation_matrix()
#d.explanatory_dimension()

#print(d.input_data.shape)
m = mlModels("surrogateCreation/trainingInput.csv","surrogateCreation/trainingOutput.csv" )
good_models = m.select_promising_ml_models(accuracy_cutoff=0.9)
m.hyperparameter_optimisation(good_models)
m.train_best_model()

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


