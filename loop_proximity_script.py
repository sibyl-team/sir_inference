from pathlib import Path
import csv
import os
import numpy as np
import pandas as pd
import argparse
import sklearn.metrics as mm
import sys
# sir_inference imports
from sir_model import FastProximityModel, patient_zeros_states
from ranking import csr_to_list
import os.path
from os.path import join
from os import path

from time import time
from scenario import Scenario
from sir_model import EpidemicModel, patient_zeros_states, symptomatic_individuals
import ranking
from ranking import RANKINGS

# READ ARGUMENTS
parser = argparse.ArgumentParser(description="Run another simulation and don't ask.")
parser.add_argument('--N', type=int, default=10000, dest="N", help='network size')
parser.add_argument('--T', type=int, default=100, dest="T", help='total time')
parser.add_argument('--nsi', type=int, default=5, dest="N_patient_zero", help='number of initial seeds')
#network arguments
parser.add_argument('--gseed', type=int, default=1, dest="graph_seed", help='seed of graph constructor')
parser.add_argument('--gdir', type=str, default="networks", dest="location", help='output directory of graphs')
parser.add_argument('--scale', type=float, default=1.0, dest="scale", help='scale of graph constructor')

parser.add_argument('--seed', type=int, default=1, dest="seed", help='seed of dynamics')
parser.add_argument('--mu', type=float, default=0.05, dest="mu", help='recovery rate')
parser.add_argument('--lamb', type=float, default=0.03, dest="lamb", help='infection rate')
parser.add_argument('-o', type=int, default=100, dest="num_test_algo", help='number of observations algo')
parser.add_argument('--outdir', type=str, default="output", dest="out_dir", help='output directory of results')
parser.add_argument('-i', type=int, default=0, dest="initial_steps", help='initial_steps')

parser.add_argument('--or', type=int, default=0, dest="num_test_random", help='number of observations random')
parser.add_argument('--fss', type=float, default=1, dest="fraction_SS_obs", help='fraction of observed SS')
parser.add_argument('--fsm', type=float, default=0.3, dest="fraction_SM_obs", help='fraction of observed SM')
parser.add_argument('--af', type=float, default=1, dest="adoption_fraction", help='adoption fraction')
#parser.add_argument('--pai', type=float, default=1e-10, dest="pautoinf", help='auto-infection probability')
#parser.add_argument('--fp_rate', type=float, default=0, dest="fp_rate", help='false positive rate')
#parser.add_argument('--fn_rate', type=float, default=0, dest="fn_rate", help='false negative rate')
#parser.add_argument('--window', type=int, default=14, dest="window_length", help="window_length")
#parser.add_argument('--threads', type=int, default=None, dest="num_threads", help='num threads')
#parser.add_argument('--smart_users', type = int, default = 0, dest="smartphone_users_abm", help = "smartphone_users_abm")

args = parser.parse_args()


#if args.num_threads is not None:
    #sib.set_num_threads(args.num_threads)
    #print(f"using {args.num_threads} threads")
# set parameters of the openABM foward simulation


N = args.N
T = args.T
gseed = args.graph_seed
seed = args.seed
location=args.location
N_patient_zero = args.N_patient_zero
mu = args.mu
lamb = args.lamb
scale = args.scale
out_dir = args.out_dir
initial_steps = args.initial_steps
num_test_algo = args.num_test_algo
########################################################################
##################### generate network #################################
########################################################################
print("Generate network with N=%d T=%d scale=%.1f default lambda=%.2f seed=%d..."%(N,T,scale,lamb,gseed), flush=True)

fold_location = Path(args.location)
if not fold_location.exists():
    fold_location.mkdir(parents=True)

fold_out = Path(out_dir)
if not fold_out.exists():
    fold_out.mkdir(parents=True)

if path.isdir(location) : print("Will save in "+location)
else : 
    print("log file not found. was looking for: \n "+location+"\n Bye Bye")
    sys.exit()
    
logfile="interactions_proximity_N%dK_scale%.1f_T%d_seed%d.csv"%(N/1000,scale,T,gseed)
if os.path.isfile(join(location,logfile)):
    print("Network already there, skipping generation")
else:
    model.run(T=T, print_every=10)

    print("Saving transmissions...", flush=True)
    logfile="interactions_proximity_N%dK_scale%.1f_T%d_seed%d.csv"%(N/1000,scale,T,gseed)
    with open(location+"/"+logfile, 'w', newline='') as csvfile:
        fieldnames = ['t', 'i', 'j', 'lamb']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for t, A in enumerate(model.transmissions):
            for i, j, lamb in csr_to_list(A):
                writer.writerow(dict(t=t, i=i, j=j, lamb=lamb))
    print("Bye-Bye")
    


np.random.seed(seed);    
##### LOOP
flag = f"Proximity_{int(N/1000)}k_T_{T}_seed_{seed}_gseed_{seed}_lamb_{lamb}_mu_{mu}"
print("Load Proximity model", flush=True)
tic = time()
model = EpidemicModel(initial_states=np.zeros(N), x_pos=np.zeros(N), y_pos=np.zeros(N))
model.load_transmissions(join(location,logfile), new_lambda = lamb)
model.recover_probas = mu*np.ones(N)
print(f"Loading took {time()-tic:.1f}s",flush=True)
model.initial_states = patient_zeros_states(N, N_patient_zero)
model.time_evolution(model.recover_probas, model.transmissions, print_every=50)
t_max = len(model.transmissions)
print("Save plain dynamics", flush=True)
db = pd.DataFrame()
db["S"] = np.sum(model.states==0,axis=1)
db["I"] = np.sum(model.states==1,axis=1)
db["R"] = np.sum(model.states==2,axis=1)
db.to_csv(join(out_dir,flag)+ "_freedyn.csv",index=False, sep="\t")
del db
model.initial_states = model.states[initial_steps]
model.states = model.states[initial_steps:]
model.transmissions = model.transmissions[initial_steps:]

################ PARAMETERS ################################
# trac parameters
trac_tau = 5;
# MF parameters
MF_tau = 5;
MF_delta = 15;
# observation parameters
n_ranking = num_test_algo
p_untracked=0
#seed = int(sys.argv[2]);
#seeds for running [32,123,456]
#seed=int(sys.argv[1]);
################################################

intervention_options=dict(quarantine_time=T-initial_steps)
observation_options=dict(n_random=0,n_infected=0,n_ranking=n_ranking, p_symptomatic=0.5, tau=5, p_untracked=p_untracked)

################## MF #############################
scenario_MF = Scenario(
    model, seed=seed+1, 
    ranking_options=dict(ranking=RANKINGS["backtrack"], 
                         algo="MF", init="all_S", tau=MF_tau, delta=MF_delta), 
    observation_options=observation_options,
    intervention_options=intervention_options,
)
scenario_MF.run(t_max-initial_steps, print_every = 1)
print("Save MF strategy", flush=True)
scenario_MF.counts.to_csv(join(out_dir,flag)+ "_MF_res.csv",index=False, sep="\t")
del scenario_MF
# 1h01min per round
print("End seed", flush=True)




##### RANDOM SCENARIO #######
scenario_rnd = Scenario(
    model, seed=seed+1, 
    ranking_options=dict(ranking=RANKINGS["random"]),
    observation_options=observation_options,
    intervention_options=intervention_options
)

scenario_rnd.run(t_max-initial_steps,  print_every = 1)
print("Save random strategy", flush=True)
#scenario_rnd.counts.to_csv("csv/Proximity_N%dK_T%d_s1_ti%d_pz%d_mu%.2f_l%.2f_seed%d_obs%d_rnd.csv"%(N/1000,T,initial_steps,N_patient_zero,mu,lamb,seed,n_ranking),
 #         index=False, sep="\t")
scenario_rnd.counts.to_csv(join(out_dir,flag)+ "_random_res.csv",index=False, sep="\t")
del scenario_rnd




##### TRACING SCENARIO #######
scenario_trac = Scenario( model, seed=seed+1, 
    ranking_options=dict(ranking=RANKINGS["tracing"], tau=trac_tau),
    observation_options=observation_options,
    intervention_options=intervention_options
)
scenario_trac.run(t_max-initial_steps, print_every = 1)
print("Save tracing strategy", flush=True)
#scenario_trac.counts.to_csv("csv/Proximity_N%dK_T%d_s1_ti%d_pz%d_mu%.2f_l%.2f_seed%d_obs%d_trac_t%d.csv"%(N/1000,T,t1,N_patient_zero,mu,lamb,seed,n_ranking,trac_tau),
#                  index=False, sep="\t")
scenario_trac.counts.to_csv(join(out_dir,flag)+ "_tracing_res.csv",index=False, sep="\t")
del scenario_trac



