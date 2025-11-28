import time
import numpy as np
import pickle
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score
from sklearn.preprocessing import StandardScaler
import os
from datetime import datetime
import matplotlib.pyplot as plt

from lurtis_eoe.Fitness.FitnessFunction import StatefullFitnessFunction, Solution
from fitnessUtils import *


class stressCatheter(StatefullFitnessFunction):
    def __init__(self, numRings=6, logFolder = ".", writeOutput = True):
        ## These first lines are very specific to the problem. Need to be user set.
        self.writeOutput = writeOutput
        self.directory = os.getcwd()
        self.templateFolder = os.path.join(self.directory, "fluidTemplate")
        self.now = datetime.now().strftime("%m-%d-%H:%M") 
        os.system(f"mkdir simulations{self.now}")
        # number of rings of holes
        self.Q = 6
        self.L = 30
        self.numOrientation = 1 # existing fitting parameter
        ## for each ring of holes need the z spacing, hole radius, offset angle, number of holes. 
        self.order, lb, ub = [], [], []
        for param, l, u in zip(["radius", "zPos", "number", "offset"], [0.05, 0.2, -0.5+1e-8, 0], [0.5, 10, 10.5, 360]):
            self.order += [f"{param}{i}" for i in range(numRings)]
            lb += [l for _ in range(numRings)]
            ub += [u for _ in range(numRings)]
        
        StatefullFitnessFunction.__init__(self, np.array(lb), np.array(ub), len(lb))

        if writeOutput:
            ## Set up log files, labelled by start time
            self.dispFile = os.path.join(self.directory, logFolder, "disps.csv")
            self.discardParams = os.path.join(self.directory, logFolder, "discardParams.csv")
            for file in [self.dispFile, self.discardParams]: os.system(f'rm -rf {file};touch {file}')
            with open(self.dispFile, "a") as f: f.write("runID,fitness,"+ ",".join([f"fitness{i}" for i in range(self.numOrientation)])+ ",".join(self.order)+"\n")
            with open(self.discardParams, "a") as f: f.write("runID," + ",".join(self.order)+"\n")
        

        ## Set up and train the ML models in existing data 
        stress, disp =  pd.read_csv("trainStress.csv", header=None),  pd.read_csv("trainDisp.csv", header=None)
        X, y = stress.to_numpy(), disp.to_numpy().reshape(-1)
        print(X.shape, y.shape)
        print(np.any(np.isnan(X)), np.any(np.isinf(X)))
        print(np.any(np.isnan(y)), np.any(np.isinf(y)))
        if os.path.exists('scaler.pkl') and os.path.exists('rfr.pkl'): 
            print("both ml models found")
            with open('scaler.pkl', 'rb') as f: self.scaler = pickle.load(f)
            with open('rfr.pkl', 'rb') as f: self.rfr = pickle.load(f)
            X = self.scaler.transform(X)
            y_pred = self.rfr.predict(X)   
        else:
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)
            with open('scaler.pkl', 'wb') as f: pickle.dump(self.scaler, f)
            self.rfr = RandomForestRegressor(random_state=30, n_estimators= 200, min_samples_split= 3, min_samples_leaf= 4, max_features= 'sqrt', max_depth= 5)
            self.rfr.fit(X, y)
            with open("rfr.pkl", 'wb') as file: pickle.dump(self.rfr, file)
            y_pred = self.rfr.predict(X)
        mse = mean_squared_error(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        print(f"Random forest regressor - MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
        

    def fitness(self, values, id):
        ## Split up and label parameter set.
        print(f"starting fitness id = {id}")
        valuesSave = values
        n = int(len(values) / 4)
        values = [round(v, 3) for v in values]
        radius, zPos, number, offset = values[:n], values[n:2*n],values[2*n:3*n],values[3*n:]
        number = [round(n) for n in number]
        radius = [r if r > 0.1 else 0 for r in radius]
        values = radius + zPos + number + offset
        print("Run " + str(id) + ", with parameters "+', '.join([str(d) for d in values]),datetime.now().strftime("%H:%M:%S"))

        zList = [zPos[0] + radius[0]] + [radius[i]+zPos[i+1]+radius[i+1] for i in range(len(radius)-1)]
        zList  = np.cumsum(np.asarray(zList))
        paramDict = {"radius[]": radius, "zPos[]": zList, "number[]": number, "offset[]": offset, "alpha": 0}
        #print(paramDict)
        maxDeformation = []
        
        for run in range(self.numOrientation):
            # Set up new file structure
            runID = str(id) + "-" + str(run)
            alpha = run * 360 / self.numOrientation
            paramDict["alpha"] = alpha
            print(f"Starting simulation {runID}, rotation angle $\alpha$ ", alpha,datetime.now().strftime("%H:%M:%S") )
            runFolder = os.path.join(self.directory, f"simulations{self.now}", 'sim'+(runID))
            Log = runFolder+"/Outfile.txt" 
            print("Created folder " + str(runID)+"\n")
            # Copy template folder over
            os.system(f"cp -r {self.templateFolder} {self.directory}/sim{str(runID)}")
            os.system(f"mv {self.directory}/sim{str(runID)} {self.directory}/simulations{self.now}")

            configureMesh(runFolder, paramDict)
            print("Configured mesh in run "+runID,datetime.now().strftime("%H:%M:%S"))
            # Run simulations  
            exitCode = os.system(f"timeout 1500s {runFolder}/run.sh {runFolder} >> {Log}")
            print('Completed simulation ' + runID+" with exit code "+str(exitCode),datetime.now().strftime("%H:%M:%S") )

            # If initial run fails, try ajusting alpha by 1 degree and trying again.
            if not exitCode==0:
                os.system(f"cd {runFolder} \n ./clean.sh")
                paramDict["alpha"] = paramDict["alpha"] + 1
                configureMesh(runFolder, paramDict)
                exitCode = os.system(f"timeout 1500s {runFolder}/run.sh {runFolder} >> {Log}")
                if not exitCode==0:
                    maxDeformation.append(np.nan)
                    continue

            stressFieldFlap(runFolder)
            stress =  pd.read_csv(f"{runFolder}/stress.csv", header=None)
            newX = stress.to_numpy().reshape(1, -1)
            newX = self.scaler.transform(newX)
            rf_prediction = self.rfr.predict(newX)

            # Output the predictions
            print(f"Random Forest Prediction: {rf_prediction[0]}")
            with open(f"{runFolder}/MLpred.txt", "w") as f: f.write(f"Random Forest Prediction: {rf_prediction[0]} \n ")

            maxDeformation.append(rf_prediction[0])
        fitness = np.nanmax(maxDeformation)
        
        #os.system(f"rm -rf {self.directory}/simulations{self.now}/sim{id}-*")
        with open(self.dispFile, "a") as f: f.write(f"{id},{fitness},"+",".join([str(x) for x in maxDeformation+list(values)]) + "\n")
        return Solution(id, fitness, proposed_genome=valuesSave, canonical_genome=np.array(values))
    
    def preprocess(self, values):
        '''
        Receives the input and pre-process the sample.
        :param values: input data
        :return:
        - True/False: Whether discard or not the sample
        - values to use in the optimization. Can be modified.
            Returns unmodified from input.

        Want a non-negligible inter-hole difference to make the
        mesh stable
        '''
        discard = False
        valuesSave = values  
        n = round(len(values) / 4)
        values = [round(v, 3) for v in values]
        radius, zPos, number, offset = values[:n], values[n:2*n],values[2*n:3*n],values[3*n:]
        number = [round(n) for n in number]
        if sum(number) == 0:
            discard = True
        zList = [zPos[0] + radius[0]] + [radius[i]+zPos[i+1]+radius[i+1] for i in range(len(radius)-1)]
        zList  =np.cumsum(np.asarray(zList))

        if zList[-1] > 15: discard = True
        for r, n in zip(radius, number):
            if 2*r*n > 2*np.pi*1.25: 
                discard=True
                return discard, valuesSave

        if discard: 
            with open(self.discardParams, "a") as f: f.write(",".join([str(x) for x in values]) + "\n")
    
        return discard, valuesSave

if __name__=='__main__':

    # Bounds of the discrete vars adjusted by 0.5 as later will round
    lb = [0.035, 0.5, 0.5]
    ub = [0.5, 8.5, 40.5]

    bounds = np.array([lb,ub]) #[Bounds]
    numRings = 6
    number = [4]*numRings #[3]*numRings # 
    offset = [0]*numRings # [0,60,0,60,0,60,0,60] #
    zPos = [1.2, 1.0, 1.0750, 1.1556, 1.2423,1.3355] #[5.455, 2.7, 2.7, 2.7, 2.7, 2.7, 2.7, 2.7] #
    exteriorRad = np.array([0.8496, 0.7732, 0.6874, 0.6180, 0.5650, 0.5236])/2 # [0.455]*numRings # 
    problem = stressCatheter(numRings=numRings)
    print(problem.preprocess(values=list(exteriorRad)+zPos+number+offset))
    fitness = problem.fitness(values=list(exteriorRad)+zPos+number+offset, id=1).fitness
