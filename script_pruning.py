import sys

sys.path.insert(0,'../sib/')
import csv
import sib
import os
import numpy as np
import pandas as pd
import sklearn.metrics as mm
# sir_inference imports
from sir_model import FastProximityModel, patient_zeros_states
from ranking import csr_to_list
import os.path
from os import path
N=500000
N_patient_zero = 200;
lamb_load = 0.05;
lamb = 0.05
mu = 0.02;
scale=1.0; # Easy Case
T=100;
seed=2;
location="networks"
# SIR parameters
t1 = 10;

##################### pruning fraction, per day
pru = 0.81 # ~ 0.90 AF
#pru = 0.64 # ~ 0.80 AF
#pru = 0.49 # ~ 0.70 AF
#pru = 0.36 # ~ 0.60 AF

## import sys
from time import time
import numpy as np
import pandas as pd
import os.path
from os import path
from scenario import Scenario
from sir_model import EpidemicModel, patient_zeros_states, symptomatic_individuals
from ranking import RANKINGS
log_fname="interactions_proximity_N%dK_s%.1f_T%d_lamb%.2f_s%d.csv"%(N/1000,scale,T,lamb_load,seed)
csv_file=location+"/"+log_fname
print("seed = %d"%seed, flush=True)
np.random.seed(seed);
if path.exists(csv_file): print("Loading model from %s"%csv_file)
else:
    print("Could not find the file! Was looking for \n "+csv_file+"\n Bye-Bye")
    sys.exit()
print("Load Proximity model", flush=True)
tic = time()
model = EpidemicModel(initial_states=np.zeros(N), x_pos=np.zeros(N), y_pos=np.zeros(N))
model.load_transmissions(csv_file, new_lambda = lamb)
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
db.to_csv("csv/Proximity_N%dK_T%d_s1_pz%d_mu%.2f_l%.2f_seed%d.csv"%(N/1000,T,N_patient_zero,mu,lamb,seed),
          index=False, sep="\t")
del db
model.initial_states = model.states[t1]
model.states = model.states[t1:]
model.transmissions = model.transmissions[t1:]


from scipy import sparse
model.pruned_transmissions = []
for t in range(0,len(model.transmissions)):
    A = model.transmissions[t].nonzero()
    row = A[0]
    col = A[1]
    keep = np.random.permutation(len(row))
    keep = keep[0:int(pru*len(row))]
    print(len(row), len(keep), len(keep)/len(row))    
    B = sparse.csr_matrix((lamb * np.ones(len(keep),), (row[keep], col[keep])), shape = (N,N))
    model.pruned_transmissions.append(B)
    
    
tau = 5
# trac parameters
trac_tau = tau;
trac_lambd = lamb;
# MF parameters
MF_tau = tau;
MF_delta = 15;
# observation parameters
n_ranking = 1500
p_untracked=0

################################################

intervention_options=dict(quarantine_time=T)
observation_options=dict(n_random=0,n_infected=0,n_ranking=n_ranking, p_symptomatic=0.5, tau=5, p_untracked=p_untracked)

import imp
import scenario
imp.reload(scenario)
from scenario import Scenario
import sib
import bp_ranking
imp.reload(bp_ranking)
from bp_ranking import bp_ranker_class

mu_r = np.log(1 + mu)
win = 21
#bp_tau = 10
name_csv = "csv/Proximity_N%dK_T%d_s1_ti%d_pz%d_mu%.2f_l%.2f_seed%d_obs%d_bp_no_tau_win%d_pru%.2lf.csv"%(N/1000, T,t1,N_patient_zero,mu,lamb,seed,n_ranking,win,pru)
prob_seed=1/N
prob_sus = 0.55
pseed = prob_seed / (2 - prob_seed)
psus = prob_sus * (1 - pseed)
pautoinf = 1e-10

bp_ranker = bp_ranker_class(params = sib.Params(
                                 prob_r = sib.Exponential(mu=mu_r),
                                 prob_i = sib.Uniform(p=lamb),
                                 pseed = pseed,
                                 psus = psus,
                                 pautoinf = pautoinf,
                                 fn_rate = 0.001,
                                 fp_rate = 0.001),
                                 
                 maxit0 = 30,
                 maxit1 = 30,
                 tol = 1e-3,
                 memory_decay = 0.1,
                 window_length = win
                            #tau=bp_tau
                           )
bp_ranker.init(N, T)
bp_ranker.__name__ = "bp"
scenario_bp = Scenario( model, seed=seed+1, 
    ranking_options=dict(ranking=bp_ranker.step_scenario, lamb = 1.0),
    observation_options=observation_options,
    intervention_options=intervention_options,
                         save_csv = name_csv
)
scenario_bp.run(t_max-t1, print_every = 1)
print("Save bp strategy", flush=True)
scenario_bp.counts.to_csv(name_csv,
          index=False, sep="\t")


import imp
import scenario
imp.reload(scenario)
from scenario import Scenario
##### TRACING UP TO SECOND NN #######
scenario_trac_2nd = Scenario( model, seed=seed+1, 
    ranking_options=dict(ranking=RANKINGS["tracing2nd"], tau=trac_tau, lamb=trac_lambd),
    observation_options=observation_options,
    intervention_options=intervention_options
)
scenario_trac_2nd.run(t_max-t1, print_every = 1)
print("Save tracing sec NN strategy", flush=True)
scenario_trac_2nd.counts.to_csv("csv/Proximity_N%dK_T%d_s1_ti%d_pz%d_mu%.2f_l%.2f_seed%d_obs%d_trac2nd_t%d_pru%.2lf.csv"%(N/1000,T,t1,N_patient_zero,mu,lamb,seed,n_ranking,trac_tau,pru),
                  index=False, sep="\t")
#scenario_trac_2nd.save("csv/Proximity_N%dK_T%d_s1_ti%d_pz%d_mu%.2f_l%.2f_seed%d_obs%d_trac2nd_t%d_all"%(N/1000,T,t1,N_patient_zero,mu,lamb,seed,n_ranking,trac_tau))

del scenario_trac_2nd
print("End seed", flush=True)


##### MF SCENARIO #######
import imp
import scenario
imp.reload(scenario)
from scenario import Scenario
import ranking
imp.reload(ranking)
from ranking import RANKINGS
scenario_MF = Scenario(
    model, seed=seed+1, 
    ranking_options=dict(ranking=RANKINGS["backtrack"], 
                         algo="MF", init="all_S", tau=MF_tau, delta=MF_delta,mu=mu,lamb=lamb), 
    observation_options=observation_options,
    intervention_options=intervention_options,
)
scenario_MF.run(t_max-t1, print_every = 1)
print("Save MF strategy", flush=True)
scenario_MF.counts.to_csv("csv/Proximity_N%dK_T%d_s1_ti%d_pz%d_mu%.2f_l%.2f_seed%d_obs%d_MF_t%d_d%d_pru%.2lf.csv"%(N/1000,T,t1,N_patient_zero,mu,lamb,seed,n_ranking,MF_tau,MF_delta,pru),
                  index=False, sep="\t")
#scenario_MF.save("csv/Proximity_N%dK_T%d_s1_ti%d_pz%d_mu%.2f_l%.2f_seed%d_obs%d_MF_t%d_d%d_all"%(N/1000,T,t1,N_patient_zero,mu,lamb,seed,n_ranking,MF_tau,MF_delta))

del scenario_MF
# 1h01min per round
print("End seed", flush=True)


import imp
import scenario
imp.reload(scenario)
from scenario import Scenario
##### RANDOM SCENARIO #######
scenario_rnd = Scenario(
    model, seed=seed+1, 
    ranking_options=dict(ranking=RANKINGS["random"]),
    observation_options=observation_options,
    intervention_options=intervention_options
)
scenario_rnd.run(t_max-t1,  print_every = 1)
print("Save random strategy", flush=True)
scenario_rnd.counts.to_csv("csv/Proximity_N%dK_T%d_s1_ti%d_pz%d_mu%.2f_l%.2f_seed%d_obs%d_rnd_pru%.2lf.csv"%(N/1000,T,t1,N_patient_zero,mu,lamb,seed,n_ranking,pru),
          index=False, sep="\t")
del scenario_rnd


##### TRACING SCENARIO #######
scenario_trac = Scenario( model, seed=seed+1, 
    ranking_options=dict(ranking=RANKINGS["tracing"], tau=trac_tau),
    observation_options=observation_options,
    intervention_options=intervention_options
)
scenario_trac.run(t_max-t1, print_every = 1)
print("Save tracing strategy", flush=True)
scenario_trac.counts.to_csv("csv/Proximity_N%dK_T%d_s1_ti%d_pz%d_mu%.2f_l%.2f_seed%d_obs%d_trac_t%d_pru%.2lf.csv"%(N/1000,T,t1,N_patient_zero,mu,lamb,seed,n_ranking,trac_tau,pru),
                  index=False, sep="\t")
del scenario_trac
