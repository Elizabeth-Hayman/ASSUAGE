#----------------------------
# File to create the ground truth dataset
#----------------------------

import numpy as np
import os
import pandas as pd
import time
from typing import Callable, Optional, Tuple, Sequence
import subprocess



def start_new_runs(numNewRuns: int, FSItemplateFolder: str,lowerBounds: Sequence[float],upperBounds: Sequence[float],
    numCoresPerSim: int,numCores: int, parameter_to_model: Callable[[np.ndarray, str], None],
    preprocess_func: Optional[Callable[[np.ndarray], Tuple[bool, np.ndarray]]] = None, seed: int = 42) -> None:
    """
    Launch a number of new ground truth simulations in parallel, respecting user-set core limits.

    Args:
        numNewRuns: Number of simulations to start.
        FSItemplateFolder: Path to the folder to be copied for each simulation.
        lowerBounds: Lower bounds for random parameter generation.
        upperBounds: Upper bounds for random parameter generation.
        numCoresPerSim: Number of cores each simulation uses.
        numCores: Total number of available cores.
        parameter_to_model: Function to generate simulation files based on parameters.
        preprocess_func: Optional function to transform or discard parameter sets.
        seed: Seed for reproducible randomness.
    """
    np.random.seed(seed)
    GTFolder = os.path.join(os.getcwd(), "groundTruth")
    os.makedirs(GTFolder, exist_ok=True)

    # Get next available run ID
    existing = [d for d in os.listdir(GTFolder) if d.startswith("run") and d[3:].isdigit()]
    nextId = max([int(d[3:]) for d in existing], default=-1) + 1
    
    print("Starting new set of runs with first id ",nextId)
    runsLaunched = 0
    while runsLaunched < numNewRuns:
        # Generate new, random parameter vector
        values = np.random.uniform(low=lowerBounds, high=upperBounds)

        # Optional preprocessing
        if preprocess_func:
            discard, values = preprocess_func(values)
            if discard:
                continue

        print(f"Starting simulation in run{nextId}")
        runFolder = os.path.join(GTFolder, f"run{nextId}")

        try:
            subprocess.run(["cp", "-r", FSItemplateFolder, runFolder], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error copying folder: {e}")
            break

        # Model setup and execution
        os.system(f"touch {os.path.join(runFolder, 'design_parameters.csv')}")
        np.savetxt(os.path.join(runFolder, "design_parameters.csv"), values.reshape(1, -1), delimiter=',', fmt='%.3f')
        parameter_to_model(values, runFolder, alpha=0)
        with open(os.path.join(runFolder, "nohup.out"), "ab") as out:
            subprocess.Popen(
                ["./run.sh", "."],
                cwd=runFolder,
                stdout=out,
                stderr=subprocess.STDOUT
            )
        
        def numCoresRemaining() -> int:
            """
            In-built function to calculate the number of cores not currently in use, and judge whether
            further runs can be started.
            """
            folders = [f for f in os.listdir(GTFolder) if os.path.isdir(os.path.join(GTFolder, f))]
            activeFolders = sum(not os.path.exists(os.path.join(GTFolder, f, "endedSim.txt")) for f in folders)
            return numCores - numCoresPerSim * activeFolders
        
        # Wait until enough cores are available
        time.sleep(1)
        while numCoresRemaining() < numCoresPerSim:
            time.sleep(10)

        runsLaunched += 1
        nextId += 1

def extractData(input_func: Optional[Callable[[str], np.ndarray]],
                 output_func: Optional[Callable[[str], np.ndarray]]) -> None:
    """
    Processes all folders in the 'groundTruth' directory and applies input_func and output_func
    to extract and save input and output data as CSV files.

    Args:
        input_func: A callable that takes a folder path and returns a numpy array of input data.
        output_func: A callable that takes a folder path and returns a numpy array of output data.
    """
    gt_folder = os.path.join(os.getcwd(), "groundTruth")
    surrogate_folder = os.path.join(os.getcwd(), "surrogateCreation")
    os.makedirs(surrogate_folder, exist_ok=True)

    
    input_arr, output_arr = [], []

    for folder_name in sorted([f for f in os.listdir(gt_folder) if os.path.isdir(os.path.join(gt_folder, f))]):
        folder_path = os.path.join(gt_folder, folder_name)
        print(f"Starting to extract data from {folder_name}")
        try:
            input_arr.append(input_func(folder_path))
            output_arr.append(output_func(folder_path))
            print(f"Successfully extracted data from {folder_name}")
        except Exception as e:
            print(f"Warning: Skipping folder {folder_name} due to error: {e}")

    input_np = np.array(input_arr)
    output_np = np.array(output_arr)

    print(f"Input shape: {input_np.shape}")
    pd.DataFrame(input_np).to_csv(os.path.join(surrogate_folder, "trainingInput.csv"), index=False, header=False)
    pd.DataFrame(output_np).to_csv(os.path.join(surrogate_folder, "trainingOutput.csv"), index=False, header=False)

