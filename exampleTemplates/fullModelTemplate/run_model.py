import os
import json
from datetime import datetime
import numpy as np
import time

directory = os.path.dirname(os.path.abspath(__file__))
print("running full model in folder ", os.getcwd())
time.sleep(.1)

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

first_radius, second_radius, third_var = rLine[0], rLine[1], rLine[2]

## in this test the ``true value'' is quadratic in the first radius value and independent of others.
true_val = 2 * first_radius ** 2 + 3/2 * second_radius - 4* third_var ** (3/2) + np.random.random()/10
print(f"First radius { first_radius}, true fitness {true_val}")

with open("output.txt", "a") as f: f.write(f"\n4,{true_val}")



