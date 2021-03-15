import sys

sys.path.insert(0,'../sib/')
import csv
import sib
import os
import numpy as np
import pandas as pd
from time import time
import argparse
import sklearn.metrics as mm
import os.path
from os import path
# sir_inference imports
from sir_model import FastProximityModel, patient_zeros_states
from ranking import csr_to_list
from scenario import Scenario
from sir_model import EpidemicModel, patient_zeros_states, symptomatic_individuals
from ranking import RANKINGS
import sib
import bp_ranking
from bp_ranking import bp_ranker_class

parser = argparse.ArgumentParser(description="Run a simulation and don't ask.")
parser.add_argument('-s', type=int, default=1, dest="seed", help='seed')
args = parser.parse_args()
print(f"arguments {args}")

############################### Network Params ############################
factor=5
N=int(500000/factor)
seed=args.seed;
#N = 50000
## new try with 100 spreaders
N_patient_zero = int(200/factor);
lamb_load = 0.05;
lamb_true = 0.05;
mu_true = 0.02;
scale=1.0; # Easy Case
T=50;
location="networks"
# SIR parameters
t1 = 10;
############################################################################

################# parameters used in the inference proc. ###################
#lamb_infs = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10];
#lamb_infs = [0.0001, 0.001,0.003, 0.006, 0.15, 0.2, 0.5]
#lamb_infs = [0.001, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1];
lamb_infs = [0.09];
mu_infs = [0.02]; 
############################################################################

threads=int(48/2)
sib.set_num_threads(threads)

############################### Load network ################################

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
model.load_transmissions(csv_file, new_lambda = lamb_true)
model.recover_probas = mu_true*np.ones(N)
print(f"Loading took {time()-tic:.1f}s",flush=True)
model.initial_states = patient_zeros_states(N, N_patient_zero)
model.time_evolution(model.recover_probas, model.transmissions, print_every=50)
t_max = len(model.transmissions)
print("Save plain dynamics", flush=True)
db = pd.DataFrame()
db["S"] = np.sum(model.states==0,axis=1)
db["I"] = np.sum(model.states==1,axis=1)
db["R"] = np.sum(model.states==2,axis=1)
db.to_csv("csv/Proximity_N%dK_T%d_s1_pz%d_mu%.2f_l%.4f_seed%d.csv"%(N/1000,T,N_patient_zero,mu_true,lamb_true,seed),
          index=False, sep="\t")
del db

model.initial_states = model.states[t1]
model.states = model.states[t1:]
model.transmissions = model.transmissions[t1:]

#######################  Inference parameters ############################

n_ranking = int(1500/factor) # observation parameters
#n_ranking = 500
p_untracked=0
p_symptomatic=0.5
tau_obs_symp=5
intervention_options=dict(quarantine_time=T-t1)
observation_options=dict(n_random=0,n_infected=0,n_ranking=n_ranking, p_symptomatic=p_symptomatic, tau=tau_obs_symp, p_untracked=p_untracked)

############################### Starting inference loops ################################

for lamb_inf in lamb_infs:
    for mu_inf in mu_infs:

        ##### RANDOM SCENARIO #######

        scenario_rnd = Scenario(
            model, seed=seed+1, 
            ranking_options=dict(ranking=RANKINGS["random"]),
            observation_options=observation_options,
            intervention_options=intervention_options
        )
        scenario_rnd.run(t_max-t1,  print_every = 1)
        print("Save random strategy", flush=True)
        scenario_rnd.counts.to_csv("csv/Proximity_N%dK_T%d_s1_ti%d_pz%d_mu%.2f_l%.4f_seed%d_obs%d_rnd.csv"%(N/1000,T,t1,N_patient_zero,mu_inf,lamb_inf,seed,n_ranking),
                  index=False, sep="\t")
        del scenario_rnd

        ##### TRACING SCENARIO #######
        trac_tau = 5

        scenario_trac = Scenario( model, seed=seed+1, 
            ranking_options=dict(ranking=RANKINGS["tracing"], tau=trac_tau),
            observation_options=observation_options,
            intervention_options=intervention_options
        )
        scenario_trac.run(t_max-t1, print_every = 1)
        print("Save tracing strategy", flush=True)
        scenario_trac.counts.to_csv("csv/Proximity_N%dK_T%d_s1_ti%d_pz%d_mu%.2f_l%.4f_seed%d_obs%d_trac_t%d.csv"%(N/1000,T,t1,N_patient_zero,mu_inf,lamb_inf,seed,n_ranking,trac_tau),
                          index=False, sep="\t")
        del scenario_trac

        ##### TRACING UP TO SECOND NN #######

        trac_tau = 5

        scenario_trac_2nd = Scenario( model, seed=seed+1, 
            ranking_options=dict(ranking=RANKINGS["tracing2nd"], tau=trac_tau, lamb=lamb_inf),
            observation_options=observation_options,
            intervention_options=intervention_options
        )

        scenario_trac_2nd.run(t_max-t1, print_every = 1)
        print("Save tracing sec NN strategy", flush=True)
        scenario_trac_2nd.counts.to_csv("csv/Proximity_N%dK_T%d_s1_ti%d_pz%d_mu%.2f_l%.4f_seed%d_obs%d_trac2nd_t%d_mism.csv"%(N/1000,T,t1,N_patient_zero,mu_inf,lamb_inf,seed,n_ranking,trac_tau),
                          index=False, sep="\t")
        #scenario_trac_2nd.save("csv/Proximity_N%dK_T%d_s1_ti%d_pz%d_mu%.2f_l%.2f_seed%d_obs%d_trac2nd_t%d_all"%(N/1000,T,t1,N_patient_zero,mu,lamb,seed,n_ranking,trac_tau))

        del scenario_trac_2nd
        print("End seed", flush=True)

        ##### MF SCENARIO #######

        MF_tau = 5;
        MF_delta = 15;

        scenario_MF = Scenario(
            model, seed=seed+1, 
            ranking_options=dict(ranking=RANKINGS["backtrack"], 
                                 algo="MF", init="all_S", tau=MF_tau, delta=MF_delta, mu=mu_inf, lamb=lamb_inf), 
            observation_options=observation_options,
            intervention_options=intervention_options,
        )
        scenario_MF.run(t_max-t1, print_every = 1)
        print("Save MF strategy", flush=True)
        scenario_MF.counts.to_csv("csv/Proximity_N%dK_T%d_s1_ti%d_pz%d_mu%.2f_l%.4f_seed%d_obs%d_MF_t%d_d%d_mism.csv"%(N/1000,T,t1,N_patient_zero,mu_inf,lamb_inf,seed,n_ranking,MF_tau,MF_delta),
                          index=False, sep="\t")
        #scenario_MF.save("csv/Proximity_N%dK_T%d_s1_ti%d_pz%d_mu%.2f_l%.2f_seed%d_obs%d_MF_t%d_d%d_all"%(N/1000,T,t1,N_patient_zero,mu_inf,lamb_inf,seed,n_ranking,MF_tau,MF_delta))

        del scenario_MF
        # 1h01min per round
        print("End seed", flush=True)

        ##### BP SCENARIO #######


        mu_r = np.log(1 + mu_inf)
        win = 21
        #bp_tau = 10
        name_csv = "csv/Proximity_N%dK_T%d_s1_ti%d_pz%d_mu%.2f_l%.4f_seed%d_obs%d_bp_win%d_mism.csv"%(N/1000, T,t1,N_patient_zero,mu_inf,lamb_inf,seed,n_ranking,win)
        prob_seed=1/N
        prob_sus = 0.55
        pseed = prob_seed / (2 - prob_seed)
        psus = prob_sus * (1 - pseed)
        pautoinf = 1e-10

        bp_ranker = bp_ranker_class(params = sib.Params(
                                         prob_r = sib.Exponential(mu=mu_r),
                                         prob_i = sib.Uniform(p=lamb_inf),
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




