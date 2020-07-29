import sys
sys.path.insert(0,'../loop_abm/src/')
import pandas as pd
from ranking import csr_to_list
import sib
from loop_ranker import winbp_prob0_rank


class dummy_logger():
    def info(self, x):
        print(x)
        
class bp_ranker_class(winbp_prob0_rank.WinBPProb0Ranker):
    
    def set_tau_obs(self, tau = 0):
        self.tau_obs = tau
    
    def step_scenario(self, t, model, obs, params):
        tau = 0
        if hasattr(self, "tau_obs"):
            #print("tau", tau)
            tau = self.tau_obs
        else:
            tau = 0
        data = {}
        data["logger"] = dummy_logger()
        daily_contacts = []
        if t>0:
            daily_contacts = [(i, j, t-1, l) for i, j, l in list(csr_to_list(model.transmissions[t-1]))]
        daily_obs = []
        print("SCENARIO BP")
        if len(obs)!=0:
            obs = pd.DataFrame(obs)
            #print(observations)
            obs = obs[obs["t_test"] == t-1]
            print("t", t, tau)
            obs["t_test"] = max(t-1-tau, 0)
            #print(t, observations)
            daily_obs = list(obs[["i", "s", "t_test"]].itertuples(index=False, name=None))  
            print(t, "obs", len(obs))
        #print(t, daily_contacts[0])
        #print(t, daily_obs)
        rank_algo = self.rank(t, daily_contacts, daily_obs, data)
        rank = sorted(rank_algo, key= lambda tup: tup[1], reverse=True)
        rank = [(int(tup[0]), tup[1])  for tup in rank]
        rank_pd = pd.DataFrame(rank, columns=["i", "score"])
        rank_pd["rank"] = range(model.N)
        #print(rank_pd)
        return rank_pd