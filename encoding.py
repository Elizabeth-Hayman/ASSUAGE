#-----------------------------
## User set class structure to interpret a vector of decimals (the genotype) used by internal ASSUAGE methods into the 
# design to be incorporated into the reduced and full models (the phenotype).
#
# Critical functions are:
# - parameter_to_model: needs to work in the same way for the full and reduced models
# - preprocess_parameters : can be None. Will give a discard boolean based on input parameters.  
# - extract_fitness : output of a full model. Desired fitness. Will be the output of the surrogate layer 
# - extract_surrogate_input : numerical data taken from the output of a reduced order model which 
#-----------------------------

import os
import numpy as np
import pandas as pd
import subprocess
from typing import Union, Sequence, Tuple, Optional, Any

def parameter_to_model(values: Union[Sequence[float], np.ndarray],
                       runFolder: str,
                       alpha: float = 0.0) -> None:
    """
    Interpret a sequence of design parameters and write them into the model folder.

    The function expects `values` to contain groups of parameters arranged in four
    equal-length blocks (radius, zPos, number, offset). It formats those values,
    copies a template `input.geo` to `Input-orig.geo`, updates specified variables
    inside the file, and writes the resulting `input.geo` in `runFolder`.

    Parameters
    ----------
    values :
        Sequence or array of design parameter floats.
    runFolder :
        Path to the folder containing the model template files.
    alpha :
        Design-specific variable used when writing parameters.
    """
    values = [round(v, 3) for v in values]

    n = int(len(values) / 4)
    radius, zPos, number, offset = values[:n], values[n:2*n], values[2*n:3*n], values[3*n:]
    number = [round(n) for n in number]
    radius = [r if r > 0.1 else 0 for r in radius]
    zList = [zPos[0] + radius[0]] + [radius[i] + zPos[i+1] + radius[i+1] for i in range(len(radius)-1)]
    zList = list(np.cumsum(np.asarray(zList)))
    paramDict = {"radius[]": radius, "zPos[]": zList, "number[]": number, "offset[]": offset, "alpha": alpha}

    os.system(f"cp {runFolder}/input.geo {runFolder}/Input-orig.geo")
    infileInput = os.path.join(runFolder, "Input-orig.geo")
    outfileInput = os.path.join(runFolder, "input.geo")
    update_variable, new_values = list(paramDict.keys()), [paramDict[k] for k in list(paramDict.keys())]

    assert len(update_variable) == len(new_values), 'Need same number of values as variables'

    with open(infileInput) as f:
        with open(outfileInput, "w") as f1:
            for line in f:
                for i, var in enumerate(update_variable):
                    if line.startswith(var):
                        ind1, ind2 = line.find('='), line.find(';')
                        string = "{"
                        if "[]" in var:
                            for v in new_values[i]:
                                string += str(v)+","
                            newline = line[:ind1+2] + string[:-1] + "}" + line[ind2:]
                        else:
                            string += str(new_values[i])
                            newline = line[:ind1+2] + string + "}" + line[ind2:]
                        line = newline
                f1.write(line)
    os.system(f"rm {runFolder}/Input-orig.geo")


def preprocess_parameters(values: Union[Sequence[float], np.ndarray]
                          ) -> Tuple[bool, Union[Sequence[float], np.ndarray]]:
    """
    Pre-process the input parameter vector and decide whether to discard it.

    Returns a tuple (discard, values). The `discard` boolean indicates whether
    the sample should be discarded based on simple geometric and stability checks.
    The returned `values` are the (possibly) processed parameters to be used
    downstream; currently they are returned unmodified.

    Parameters
    ----------
    values :
        Sequence or array of design parameter floats.

    Returns
    -------
    tuple
        (discard_flag, values). `discard_flag` is True when the sample should be
        discarded; `values` is returned unchanged in current implementation.
    """
    discard = False
    
    n = int(len(values)/4)

    # Physical measurements are only accurate to 3 d.p.
    valuesRound = [round(v, 3) for v in values]
    
    # Discard if sum of values is outside physical ranges.
    if sum(valuesRound) <= 0:
        discard = True
    if sum(valuesRound[n:3*n]) > 100:
        discard = True

    for r, n in zip(valuesRound[:n], valuesRound[2*n:3*n]):
        if 2*r*n > 100:
            discard = True

    return discard, values


def extract_fitness(runFolder: str) -> Optional[float]:
    """
    Extract a scalar fitness value from a full model run folder.

    The function reads `output.txt` (CSV) in `runFolder` and returns the
    `maxDisp` value from the last iteration row. It will print a warning
    if the run appears insufficiently converged.

    Parameters
    ----------
    runFolder :
        Path to the folder containing the full model results.

    Returns
    -------
    float or None
        The extracted fitness value (maxDisp) or None if the file cannot be read.
    """
    try:
        iterFile = pd.read_csv(f"{runFolder}/output.txt")
        row = iterFile.iloc[-1]
        if row["Iteration"] <= 1 and row["errorDisp"] > 0.1:
            print(f"Folder {runFolder} is not sufficiently converged")
        return row["maxDisp"]

    except Exception:
        print(f"Couldn't read output file of {runFolder}")
        return None


def extract_surrogate_inputs(runFolder: str) -> np.ndarray:
    """
    Extract surrogate input features from (or create) a reduced-model run in runFolder.

    If the surrogate input CSV does not exist, the function copies an example reduced
    model template into the folder, writes parameters with `parameter_to_model`,
    and runs the reduced model script (with a timeout). It then loads the surrogate
    input CSV and computes a feature vector consisting of mean pressure and binned
    pressure / stress differences.

    Parameters
    ----------
    runFolder :
        Path to the folder containing the reduced model template or results.

    Returns
    -------
    numpy.ndarray
        A 1D array of computed features for the surrogate.
    """
    Input_folder = os.path.join(runFolder, "reducedModelTemplate")
    surrogate_input_file = os.path.join(Input_folder, "surrogate_input_data.csv")

    # If results do not exist, create a reduced model instance run the simulation
    if not os.path.exists(surrogate_input_file):
        subprocess.run(["cp", "-r", "exampleTemplates/reducedModelTemplate", runFolder], check=True)

        values = pd.read_csv(os.path.join(runFolder, "design_parameters.csv"), header=None).iloc[0].values
        # print(f"Values found in folder {runFolder}")
        parameter_to_model(values, Input_folder, alpha=0)

        # print(f"chmod a+x {Input_folder}/run.sh \n ./run.sh {Input_folder}")
        subprocess.run(["chmod", "a+x", os.path.join(Input_folder, "run.sh")], check=True)

        # Run the script with timeout
        try:
            subprocess.run(["timeout", "15m", "./run.sh", "."],
                           cwd=Input_folder, check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Simulation failed or timed out in {Input_folder}") from e

    # Load and filter pressure data
    inputDf = pd.read_csv(surrogate_input_file)
    features = inputDf.iloc[-1]


    return np.array(features)


if __name__ == '__main__':

    values = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 1, 1, 1, 1, 1, 1, 4, 4, 4, 4, 4, 4, 0, 0, 0, 0, 0, 0]

    runFolder = "groundTruth/run18"
    # print(extract_fitness(runFolder))
    # print(extract_surrogate_inputs(runFolder))
