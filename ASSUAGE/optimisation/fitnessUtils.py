import numpy as np
import os
import pandas as pd

## UTILS FOR EXTRACTING STRESS FIELD

def extractFluidData(fluidFile, csvFileName, varNames: list, patchKeyName="flap"):
    if os.path.exists(csvFileName):df = pd.read_csv(csvFileName)
    else: df = pd.DataFrame({})
    f = open(fluidFile, 'r').readlines()
    for i, line in enumerate(f):
        if patchKeyName in line: ind = i
    try:        numCells = int(f[ind+4])
    except:        print(f[ind+4])
    if len(varNames) == 1:
        varList = []
        for i in range(numCells):
            try: varList.append(float(f[ind+6+i]))
            except:
                print(f[ind+6+i])
                continue
        try: df[varNames[0]] = varList
        except: print(varList)
    elif len(varNames) == 3:
        varX, varY, varZ = [],[],[]
        for i in range(numCells):
            line = list(filter(lambda s: s.strip(), f[ind+6+i].split()))
            try:  
                varX.append(float(line[0][1:]))
                varY.append(float(line[1]))
                varZ.append(float(line[2][:-1]))
            except: 
                print(f[ind+6+i])
                continue
        try:
            df[varNames[0]] = varX
            df[varNames[1]] = varY
            df[varNames[2]] = varZ
        except: print(varX, varY, varZ)
    elif len(varNames) == 6:
        varXX, varXY, varXZ, varYY, varYZ, varZZ = [],[],[],[],[],[]
        for i in range(numCells):
            line = list(filter(lambda s: s.strip(), f[ind+6+i].split()))
            try:  
                varXX.append(float(line[0][1:]))
                varXY.append(float(line[1]))
                varXZ.append(float(line[2]))
                varYY.append(float(line[4]))
                varYZ.append(float(line[5]))
                varZZ.append(float(line[8][:-1]))
            except: 
                print(f[ind+6+i])
                continue
        try:
            df[varNames[0]],df[varNames[1]],df[varNames[2]],df[varNames[3]],df[varNames[4]],df[varNames[5]] = varXX, varXY, varXZ, varYY, varYZ, varZZ 
        except: print(varXX,varXY,varYY)
    else: print(varNames)
    df.drop(df.filter(regex='Unnamed').columns, axis=1, inplace=True) 
    df.to_csv(csvFileName)

def stressFieldFlap(runFolder):
    fullCSV = os.path.join(runFolder,"surrogate_input_data.csv")
    stressCSV = os.path.join(runFolder,"stress.csv")
    fluidFile = os.path.join(runFolder,str(np.max([int(f) for f in os.listdir(runFolder) if f.isdigit()])))
    for file, var in zip(["C", "static(p)", "wallShearStress"],
                        [["C_0", "C_1", "C_2"], ["p"], ["wallShearStress_0","wallShearStress_1","wallShearStress_2"]]):
        extractFluidData(f"{fluidFile}/{file}", fullCSV, var)
    os.system(f"rm -rf {stressCSV} \n touch {stressCSV}")
    stressFile = pd.read_csv(fullCSV)
    stressFile = stressFile[stressFile["C_0"] > 0]
    pAve = np.mean(stressFile["p"])
    pList, zPosList = [], np.linspace(0, 30, 100)
    yPosList = np.linspace(0, 3.6, 10)
    pList.append(pAve)
    ## Need the extra code as stress points aren't ordered
    
    for key in ["p","wallShearStress_0","wallShearStress_1","wallShearStress_2"]:
        for i, zPos in enumerate(zPosList): 
            for j, yPos in enumerate(yPosList):
                if j==0: continue
                if i==0: continue
                lineStresses = stressFile[(stressFile["C_1"] < yPos) & (stressFile["C_1"] > yPosList[j-1])]
                lineStresses = lineStresses[(lineStresses["C_2"] < zPos) & (lineStresses["C_2"] > zPosList[i-1])]
                if len(lineStresses)==0: print(zPos, yPos)
                if key=="p": pList.append(np.mean(lineStresses[key]) - pAve) 
                else: pList.append(np.mean(lineStresses[key]))

    with open(stressCSV, "a") as f: f.write(",".join([str(p) for p in pList])+"\n")

## UTILS TO OVERWRITE THE MESH FILE
def writeNewline(line, update_variable: str, new_values):
        ind1, ind2 = line.find('='), line.find(';')
        string = "{"
        if "[]" in update_variable:
            for v in new_values: string += str(v)+","
            newline = line[:ind1+2]+ string[:-1]+ "}" + line[ind2:]
        else: 
            string += str(new_values)
            newline = line[:ind1+2]+ string+ "}" + line[ind2:]
        return newline

def changeVariables(infile: str, outfile: str, update_variable: list, new_values: list):
    if type(update_variable) == str and (type(new_values) in [int, float]):
        # Catch bad syntax if just one variable being changed.
        update_variable = [update_variable]
        new_values = [new_values]
    assert len(update_variable) == len(new_values), 'Need same number of values as variables'

    with open(infile) as f:
        with open(outfile, "w") as f1:
            for line in f:
                for i, var in enumerate(update_variable):
                    if line.startswith(var):
                        newline = writeNewline(line, var, new_values[i])
                        line = newline
                f1.write(line)

def configureMesh(runFolder, paramDict):
    os.system("cp "+runFolder+ "/fluid.geo "+runFolder+ "/fluid-orig.geo")
    infileFluid = os.path.join(runFolder, "fluid-orig.geo")
    outfileFluid = os.path.join(runFolder, "fluid.geo")
    changeVariables(infileFluid, outfileFluid, list(paramDict.keys()), [paramDict[k] for k in list(paramDict.keys())])

# UTILS FOR PARAMETER ENCODING
def sortData(values, numAxialHoles = 5):
    """
    Takes data of the form [t1 t2 ... p1 p2 ...]
    and sorts so p values are in ascending order, 
    and t values stay with their respective pvalue.
    Return list tValues+pValues.
    """
    if len(values) > 2*numAxialHoles: params = normaliseAngle(values[:2*numAxialHoles])
    else: params = normaliseAngle(values)
    #print("normalised:")
    #print(params)
    tValues = params[:numAxialHoles]
    pValues = params[numAxialHoles:2*numAxialHoles]
    tValues = [t for _, t in sorted(zip(pValues, tValues))]
    pValues = sorted(pValues)
    return tValues, pValues

def removeOverlapData(values, numAxialHoles = 5, k=0):
    """
    Takes data of the form [t1 t2 ... p1 p2 ...]
    where p has been sorted into ascending order
    and works out any overlaps. Returns 
    reduce data of the form [t'1 t'2 ... p'1 p'2]
    which corresponds to the original parametisation
    and is restricted to the range 0, 360. 
    May be less holes as overlaps will have merged.
    """
    #print("starting overlap")
    #print(values)
    tValues, pValues = sortData(values, numAxialHoles = numAxialHoles)
    #print("after sort")
    #print(tValues, pValues)
    xLeft = [p - t for p, t in zip(pValues, tValues)]
    xRight = [p + t for p, t in zip(pValues, tValues)]
    #print("boundaries")
    #print(xLeft, xRight)
    i=0
    while i < len(xLeft)-1:
        if (xLeft[i+1]==0 and xRight[i]==0): i+=1
        elif xLeft[i+1] < xRight[i] + k:
            # holes merge so drop this value of xLeft and xRight
            xLeft.remove(xLeft[i+1])
            xRight.remove(xRight[i])
        else: i+=1

    if xLeft[0] < xRight[-1]-360 + k:
        xLeft = xLeft[1:]
        xRight = xRight[:-1]
        xRight.append(xRight[0]+360)
        xRight = xRight[1:]
    
    tEnd = [(r-l)/2 for r, l in zip(xRight, xLeft)]
    pEnd = [(r+l)/2 for r, l in zip(xRight, xLeft)]

    return tEnd, pEnd

def normaliseAngle(angleList: list):
    # Puts any angle into the range (0, 360) by
    # subtracting off multiples of 360.
    normalisedList = []
    for angle in angleList:
        while (angle > 360): angle -= 360
        while (angle < 0): angle += 360
        normalisedList.append(angle)
    return normalisedList



if __name__=='__main__':
    id = 0
    #values = np.array([20, 11, 10, 10, 10, 270, 360, 68, 90, 180, 0, 90, 4, 4])
    stressFieldFlap("simulations01-03-16:21/sim1-0")