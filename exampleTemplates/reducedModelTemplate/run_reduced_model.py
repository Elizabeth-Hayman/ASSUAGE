
import os

## Code to extract the input data for the surrogate. Here the input data is the first radius value.
directory = os.path.dirname(os.path.abspath(__file__))

def geoToParams(line):
    ind1, ind2 = line.find("{"), line.find("}")
    if ind1==-1: # catch variables not defined by lists
        ind1, ind2 = line.find("="), line.find(";")
    line = line[ind1+1:ind2]
    line = [round(float(l), 3) for l in line.split(",")]
    if len(line)==1: line = line[0]
    return line


with open(os.path.join(directory, "input.geo"), "r") as f:
    for line in f.readlines():
        if line.startswith("radius[]"): rLine = geoToParams(line)

first_radius, second_radius, third_radius = rLine[0], rLine[1], rLine[2]
writefile = os.path.join(directory, "surrogate_input_data.csv")
os.system(f"touch {writefile}")
with open(writefile, "a") as f: 
    f.write("C_0,C_1,C_2\n")
    f.write(f"{first_radius},{second_radius},{third_radius}\n")


