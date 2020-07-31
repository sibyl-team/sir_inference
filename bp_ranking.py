import sys
sys.path.insert(0,'../loop_abm/src/')
import pandas as pd
from ranking import csr_to_list
import sib
import numpy as np

class AbstractRanker():

    def __init__(self):
        self.description = "This shape has not been described yet"
        self.author = "Nobody has claimed to make this shape yet"
 
    def init(self, N, T):
        raise NotImplementedError
        
    def rank(self, t, daily_contacts, daily_obs, data):
        '''
        Order the individuals by the probability to be infected
        
        input
        ------
        t: int - 
            day of rank
        
        daily_contacts: list (i, j, t, value)
            list of daily contacts
        daily_obs: list (i, state, t)
            list of daily observations

        return
        ------
        list -- [(index, value), ...]
        '''
        raise NotImplementedError



class WinBPProb0Ranker_old(AbstractRanker):
    def __init__(self,
                params = sib.Params(),
                window_length = 14,
                maxit0 = 10,
                maxit1 = 10,
                damp0 = 0,
                damp1 = 0.5,
                tol = 1e-3,
                memory_decay = 1.0,
                tau = None,
                print_callback = lambda t,err,f: print(t,err)
                ):
        
        self.description = "class for BP inference of openABM loop"
        self.authors = "Indaco Biazzo, Alessandro Ingrosso, Alfredo Braunstein"
        self.pseed = params.pseed
        self.prob_sus = params.psus / (1 - self.pseed)
        self.params = params
        self.window_length = window_length
        self.maxit0 = maxit0
        self.maxit1 = maxit1
        self.damp0 = damp0
        self.damp1 = damp1
        self.tol = tol
        self.window_length = window_length
        self.memory_decay = memory_decay
        self.print_callback = print_callback
        self.tau = tau


    def init(self, N, T):
        self.T = T
        self.N = N
        vec_i = [self.params.prob_i(t) for t in range(T+2)]
        vec_r = [self.params.prob_r(t) for t in range(T+2)]
        #prob_i and prob_r must be PriorDiscrete for this to work
        pi = lambda : sib.PriorDiscrete(vec_i)
        pr = lambda : sib.PriorDiscrete(vec_r) 
        prob_i = pi()
        prob_r = pr()
        self.f = sib.FactorGraph(params=self.params, individuals=[(i, prob_i, prob_r, pi(), pr()) for i in range(N)])
        self.contacts = []
        self.bi = np.zeros((N, self.T + 2))
        self.br = np.zeros((N, self.T + 2))
        self.bpSs = np.full(T, np.nan)
        self.bpIs = np.full(T, np.nan)
        self.bpRs = np.full(T, np.nan)
        self.bpseeds = np.full(T, np.nan)
        self.lls = np.full(T, np.nan)
        self.all_obs = [[] for t in range(T + 1)]
        self.pi = np.array([vec_i for i in range(N)])
        self.pr = np.array([vec_r for i in range(N)])

    def rank(self, t_day, daily_contacts, daily_obs, data):

        for obs in daily_obs:
            self.f.append_observation(obs[0],obs[1],obs[2])
            self.all_obs[obs[2]] += [obs]

        
        for c in daily_contacts:
            self.f.append_contact(*c)

        ### add fake obs
        for i in range(self.N):
            self.f.append_observation(i,-1,t_day)

            
        if t_day >= self.window_length:
            t_start = t_day - self.window_length
            print("...adjust prob_i0 and prob_r0")
            nodes = self.f.nodes
            for i in range(self.N):
                n = nodes[i]
                p1 = n.bt[0]
                p2 = n.bt[1]
                p1i = p1 / (p1 + p2 + 1e-10)
                p2i = p2 / (p1 + p2 + 1e-10)
                p1r = p1 / (p1 * self.pr[i,1] + p2 * self.pr[i,0] + 1e-10)
                p2r = p2 / (p1 * self.pr[i,1] + p2 * self.pr[i,0] + 1e-10)
                for t in range(self.T):
                    self.pi[i,t] = p1i * self.pi[i,t+1] + p2i * self.pi[i,t]
                    self.pr[i,t] = p1r * self.pr[i,t+1] + p2r * self.pr[i,t]
                n.prob_i0.p = sib.VectorReal(self.pi[i,:])
                n.prob_r0.p = sib.VectorReal(self.pr[i,:])

            print("...drop first time and reset observations")
            self.f.drop_time(t_start)
            self.f.reset_observations(sum(self.all_obs[t_start+1:], []))

            if self.memory_decay < 1:
                self.f.params.pseed = 1/3
                self.f.params.psus = 2/3*self.prob_sus
                print(f"pI at intial time: {sum(self.bi[:,t_start])}")
                for i in range(self.N):
                    self.f.nodes[i].ht[0] *= self.bi[i,t_start]
                    self.f.nodes[i].hg[0] *= self.br[i,t_start+1]
            for i in range(self.N):
                self.f.nodes[i].ht[0] = max(self.f.nodes[i].ht[0], self.pseed)
                self.f.nodes[i].hg[0] = max(self.f.nodes[i].hg[0], self.pseed)
                                

        sib.iterate(self.f, maxit=self.maxit0, damping=self.damp0, tol=self.tol, 
                    callback=lambda t,e,f : print(f"sib.iterate(damp={self.damp0}):  {t}/{self.maxit0} {e:1.3e}/{self.tol}", end='    \r', flush=True))
        sib.iterate(self.f, maxit=self.maxit1, damping=self.damp1, tol=self.tol, 
                    callback=lambda t,e,f : print(f"sib.iterate(damp={self.damp1}):  {t}/{self.maxit1} {e:1.3e}/{self.tol}", end='    \r', flush=True))

        marg = np.array([sib.marginal_t(n,t_day) for n in self.f.nodes])

        for i in range(self.N):
            self.bi[i,t_day] = (1-self.memory_decay) * marg[i][1] + self.memory_decay * self.pseed
            self.br[i,t_day] = (1-self.memory_decay) * marg[i][2]

        bpS, bpI, bpR = sum(m[0] for m in marg), sum(m[1] for m in marg), sum(m[2] for m in marg)
        nseed = sum(n.bt[0] for n in self.f.nodes[:self.N])
        ll = self.f.loglikelihood()

        data["logger"].info(f"winBP: (S,I,R): ({bpS:.1f}, {bpI:.1f}, {bpR:.1f}), seeds: {nseed:.1f}, ll: {ll:.1f}")

        self.bpSs[t_day] = bpS
        self.bpIs[t_day] = bpI
        self.bpRs[t_day] = bpR
        self.bpseeds[t_day] = nseed
        self.lls[t_day] = ll

        data["<I>"] = self.bpIs
        data["<IR>"] = self.bpRs + self.bpIs
        data["<seeds>"] = self.bpseeds
        data["lls"] = self.lls
        ###### warning
        
        # inf_prob = [[i, 1-self.f.nodes[i].bt[-1]] for i in range(self.N)]
        if self.tau:
            day_start = lambda idx: max(idx - self.tau, 0)
            idx_day = lambda n, t: list(n.times).index(t)
            inf_prob = np.array([[i_n, sum(n.bt[day_start(idx_day(n, t_day)):idx_day(n, t_day)])] for i_n, n in enumerate(self.f.nodes)])
        else:
            inf_prob = [[i, marg[i,1]] for i in range(self.N)]
            
        rank = list(sorted(inf_prob, key=lambda tup: tup[1], reverse=True))
        return rank




class WinBPProb0Ranker(AbstractRanker):
    def __init__(self,
                params = sib.Params(),
                window_length = 14,
                maxit0 = 10,
                maxit1 = 10,
                damp0 = 0,
                damp1 = 0.5,
                tol = 1e-3,
                memory_decay = 1.0,
                tau = None,
                print_callback = lambda t,err,f: print(t,err)
                ):
        
        self.description = "class for BP inference of openABM loop"
        self.authors = "Indaco Biazzo, Alessandro Ingrosso, Alfredo Braunstein"
        self.pseed = params.pseed
        self.prob_sus = params.psus / (1 - self.pseed)
        self.params = params
        self.window_length = window_length
        self.maxit0 = maxit0
        self.maxit1 = maxit1
        self.damp0 = damp0
        self.damp1 = damp1
        self.tol = tol
        self.window_length = window_length
        self.memory_decay = memory_decay
        self.print_callback = print_callback
        self.tau = tau


    def init(self, N, T):
        self.T = T
        self.N = N
        pi = lambda : sib.PiecewiseLinear(self.params.prob_i, T+2)
        pr = lambda : sib.PiecewiseLinear(self.params.prob_r, T+2) 
        self.f = sib.FactorGraph(params=self.params, individuals=[(i, self.params.prob_i, self.params.prob_r, pi(), pr()) for i in range(N)])
        self.contacts = []
        self.bi = np.zeros((N, self.T + 2))
        self.br = np.zeros((N, self.T + 2))
        self.bpSs = np.full(T, np.nan)
        self.bpIs = np.full(T, np.nan)
        self.bpRs = np.full(T, np.nan)
        self.bpseeds = np.full(T, np.nan)
        self.lls = np.full(T, np.nan)
        self.all_obs = [[] for t in range(T + 1)]
        prob_i, prob_r = pi(), pr()
        self.pi = np.array([np.array(prob_i.theta) for i in range(N)])
        self.pr = np.array([np.array(prob_r.theta) for i in range(N)])

    def rank(self, t_day, daily_contacts, daily_obs, data):

        for obs in daily_obs:
            self.f.append_observation(obs[0],obs[1],obs[2])
            self.all_obs[obs[2]] += [obs]

        
        for c in daily_contacts:
            self.f.append_contact(*c)
        
        ### add fake obs
        for i in range(self.N):
            self.f.append_observation(i,-1,t_day)
        
        if t_day >= self.window_length:
            t_start = t_day - self.window_length
            print("...adjust prob_i0 and prob_r0")
            nodes = self.f.nodes
            for i in range(self.N):
                n = nodes[i]
                p1, p2 = n.bt[0], n.bt[1]
                norm_i, norm_r = p1 + p2 + 1e-10, p1 * self.pr[i,1] + p2 * self.pr[i,0] + 1e-10
                for t in range(self.T):
                    self.pi[i,t] = (p1 * self.pi[i,t+1] + p2 * self.pi[i,t]) / norm_i
                    self.pr[i,t] = (p1 * self.pr[i,t+1] + p2 * self.pr[i,t]) / norm_r
                n.prob_i0.theta = sib.RealParams(self.pi[i,:])
                n.prob_r0.theta = sib.RealParams(self.pr[i,:])

            print("...drop first time and reset observations")
            self.f.drop_time(t_start)
            self.f.reset_observations(sum(self.all_obs[t_start+1:], []))

            if self.memory_decay < 1:
                self.f.params.pseed = 1/3
                self.f.params.psus = 2/3*self.prob_sus
                print(f"pI at intial time: {sum(self.bi[:,t_start])}")
                for i in range(self.N):
                    self.f.nodes[i].ht[0] *= self.bi[i,t_start]
                    self.f.nodes[i].hg[0] *= self.br[i,t_start+1]
            for i in range(self.N):
                self.f.nodes[i].ht[0] = max(self.f.nodes[i].ht[0], self.pseed)
                self.f.nodes[i].hg[0] = max(self.f.nodes[i].hg[0], self.pseed)
                                

        sib.iterate(self.f, maxit=self.maxit0, damping=self.damp0, tol=self.tol, 
                    callback=lambda t,e,f : print(f"sib.iterate(damp={self.damp0}):  {t}/{self.maxit0} {e:1.3e}/{self.tol}", end='    \r', flush=True))
        print()
        sib.iterate(self.f, maxit=self.maxit1, damping=self.damp1, tol=self.tol, 
                    callback=lambda t,e,f : print(f"sib.iterate(damp={self.damp1}):  {t}/{self.maxit1} {e:1.3e}/{self.tol}", end='    \r', flush=True))
        print()

        marg = np.array([sib.marginal_t(n,t_day) for n in self.f.nodes])

        for i in range(self.N):
            self.bi[i,t_day] = (1-self.memory_decay) * marg[i][1] + self.memory_decay * self.pseed
            self.br[i,t_day] = (1-self.memory_decay) * marg[i][2]

        bpS, bpI, bpR = sum(m[0] for m in marg), sum(m[1] for m in marg), sum(m[2] for m in marg)
        nseed = sum(n.bt[0] for n in self.f.nodes[:self.N])
        ll = self.f.loglikelihood()

        data["logger"].info(f"winBP: (S,I,R): ({bpS:.1f}, {bpI:.1f}, {bpR:.1f}), seeds: {nseed:.1f}, ll: {ll:.1f}")

        self.bpSs[t_day] = bpS
        self.bpIs[t_day] = bpI
        self.bpRs[t_day] = bpR
        self.bpseeds[t_day] = nseed
        self.lls[t_day] = ll

        data["<I>"] = self.bpIs
        data["<IR>"] = self.bpRs + self.bpIs
        data["<seeds>"] = self.bpseeds
        data["lls"] = self.lls
        ###### warning
        
        # inf_prob = [[i, 1-self.f.nodes[i].bt[-1]] for i in range(self.N)]
        if self.tau:
            day_start = lambda idx: max(idx - self.tau, 0)
            idx_day = lambda n, t: list(n.times).index(t)
            inf_prob = np.array([[i_n, sum(n.bt[day_start(idx_day(n, t_day)):idx_day(n, t_day)])] for i_n, n in enumerate(self.f.nodes)])
        else:
            inf_prob = [[i, marg[i,1]] for i in range(self.N)]
            
        rank = list(sorted(inf_prob, key=lambda tup: tup[1], reverse=True))
        return rank
    
class dummy_logger():
    def info(self, x):
        print(x)
        
class bp_ranker_class(WinBPProb0Ranker):
    
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
        #print(daily_contacts[0:10])
        daily_obs = []
        print("\nSCENARIO BP")
        if len(obs)!=0:
            obs = pd.DataFrame(obs)
            #print(observations)
            obs = obs[(obs["t_test"] == t-1)]
            obs['t_test'] = np.where(obs['source'] == "ranking", max(t-1, 0),max(t-1-tau, 0))
            #print(t, observations)
            daily_obs = list(obs[["i", "s", "t_test"]].itertuples(index=False, name=None))  
            print("Observations:", t, "obs", len(obs), len(obs[obs["source"] == "ranking"]), len(obs[obs["source"] == "symptomatic"]))
        #print(t, daily_contacts[0])
        #print(t, daily_obs)
        if t==0 and len(daily_contacts) == 0:
            N = self.N
            rank_algo = np.zeros((N,2))
            rank_algo[:,0] = np.arange(N)
            rank_algo[:,1] = np.random.rand(N)
        else:
            rank_algo = self.rank(t, daily_contacts, daily_obs, data)
        rank = sorted(rank_algo, key= lambda tup: tup[1], reverse=True)
        rank = [(int(tup[0]), tup[1])  for tup in rank]
        rank_pd = pd.DataFrame(rank, columns=["i", "score"])
        rank_pd["rank"] = range(model.N)
        #print(rank_pd)
        return rank_pd