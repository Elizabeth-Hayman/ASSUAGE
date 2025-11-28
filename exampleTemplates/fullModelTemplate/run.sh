#!/bin/sh
set -e
cd $1

python3 run_model.py
touch endedSim.txt
