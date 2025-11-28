
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score
from scipy.stats import sem, spearmanr
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.preprocessing import StandardScaler
from typing import Optional, Dict, Union
import os
import matplotlib
matplotlib.rcParams.update({"font.size": 55})

def plot_feature_importance(ML_model_pickle: str, data_label: str = "", pca_model_pickle = None):
    """
    ML_model_pickle: pickle file name 
    """
    
    assert os.path.exists(ML_model_pickle), f"No pickle file found at {pca_model_pickle}, check file paths"
    with open(ML_model_pickle, 'rb') as f: 
        ml_model = pickle.load(f)
    imFile = os.path.join("surrogateCreation", f"initialML{data_label}")


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
    stress = pd.read_csv("allStress.csv", header=None)
    overfit = ""


    stress, disp =  pd.read_csv("allStress.csv", header=None),  pd.read_csv("maxDisp.csv", header=None)
    X, y = stress.to_numpy(), disp.to_numpy().reshape(-1)
    print(X.shape, y.shape)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    np.random.seed(40)

    #train_index, val_index = list(KFold(n_splits=10, shuffle=True, random_state=40).split(X, y))[0]
    #_, X, _, y = X[train_index], X[val_index], y[train_index], y[val_index]
    rf = RandomForestRegressor(random_state=40, n_estimators= 200, min_samples_split= 3, min_samples_leaf= 4, max_features= 'sqrt', max_depth= 5)
    #rf = RandomForestRegressor(random_state=42, max_depth= 10, min_samples_leaf= 1, min_samples_split=2, n_estimators= 300)
    if len(overfit)>0:
        rf = XGBRegressor(subsample= 1.0, n_estimators= 100, max_depth= 2, learning_rate= 0.1, colsample_bytree= 0.8)
    rf.fit(X, y)
    y_pred = rf.predict(X)
    mse = mean_squared_error(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    print("Full data", r2, mae, mse)
    print("spearman ", spearmanr(y, y_pred))
    plt.figure(figsize=(20,20))
    plt.scatter(y_pred, y, s=50)
    plt.plot([0,1],[0,1], "r--", linewidth=3)
    plt.xlabel("Predicted value")
    plt.ylabel("True value")
    plt.tight_layout()
    plt.savefig(f"fullData{overfit}.png")
    plt.close()"""

