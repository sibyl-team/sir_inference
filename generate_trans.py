import sys
sys.path.insert(0,'../sib/')
import sib
import csv
import os
import numpy as np
import pandas as pd
import sklearn.metrics as mm
import sys
import argparse
# sir_inference imports
from sir_model import FastProximityModel, patient_zeros_states
from ranking import csr_to_list
import os.path
from os import path
factor=1
N=int(500000/factor)
#N=50000
## new try with 100 spreaders
N_patient_zero = int(200/factor);
#N_patient_zero = 5;
lamb = 0.05;
mu = 0.02;
scale=1.0; # Easy Case
T=100;

parser = argparse.ArgumentParser(description="Run a simulation and don't ask.")
parser.add_argument('-s', type=int, default=1, dest="seed", help='seed')
args = parser.parse_args()
print(f"arguments {args}")


seed=args.seed

print("Generate network with N=%d T=%d scale=%.1f default lambda=%.2f seed=%d..."%(N,T,scale,lamb,seed), flush=True)
initial_states = patient_zeros_states(N, N_patient_zero)
model = FastProximityModel(N, scale, mu, lamb, initial_states)
location="networks"
if path.isdir(location) : print("Will save in "+location)
else : 
    print("log file not found. was looking for: \n "+location+"\n Bye Bye")
    sys.exit()
model.run(T=T, print_every=2)
# model.get_counts().plot(
#     title=f"N_patient_zero={N_patient_zero} lamb={lamb:.2f}  mu={mu:.2f}"
# );
print("Saving transmissions...", flush=True)
logfile="interactions_proximity_N%dK_s%.1f_T%d_lamb%.2f_s%d.csv"%(N/1000,scale,T,lamb,seed)
with open(location+"/"+logfile, 'w', newline='') as csvfile:
    fieldnames = ['t', 'i', 'j', 'lamb']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for t, A in enumerate(model.transmissions):
        for i, j, lamb in csr_to_list(A):
            writer.writerow(dict(t=t, i=i, j=j, lamb=lamb))
print("Bye-Bye")