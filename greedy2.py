import time
import numpy as np
import pandas as pd
import numba as nb
import scipy.sparse as sparse
from inference_model import MeanField, DynamicMessagePassing
from sir_model import frequency, indicator
from ranking import ranking_random
import sib


def csr_to_list(x):
    x_coo = x.tocoo()
    return zip(x_coo.row, x_coo.col, x_coo.data)


def upd_score(k,Score,lamb,noise):
    
    Score[k] += lamb*lamb + np.random.rand() * noise

def get_nonzero_idx_d(x):
    idx = x.tocoo().nonzero()
    y = x.tocsr()
    return *idx, np.array(y[idx])

@nb.njit()
def count_valid_c1(alli, allj, allt, maxS, minR):
    """
    Count valid contacts, given i, j, and t
    """
    res = np.zeros(alli.shape[0], dtype=np.bool_)
    for l in range(alli.shape[0]):
        i = alli[l]
        j = allj[l]
        t = allt[l]
        if t > max(maxS[i], maxS[j]):
            if t < minR[j]:
                res[l] = True

    return res

@nb.njit()
def sum_score_c1(scores, rands, column_i, lamb, noise):
    for l in range(column_i.shape[0]):
        i = column_i[l]
        scores[i] += lamb + rands[l] * noise

def create_mat_c2(contacts_cut2, model, debug=False):
    mat = []
    times =[]
    c2t = contacts_cut2.sort_values("t", ascending=False)

    for t, gr in c2t.groupby("t",sort=False):
        v = sparse.coo_matrix((np.ones(len(gr),np.int), (gr.i.to_numpy(), gr.j.to_numpy())), shape=(model.N, model.N))
        if len(mat)> 0:
            mat.append(v.tocsr()+mat[-1])
        else:
            mat.append(v.tocsr())
        times.append(t)
        if debug: print(f"Appeding contacts, t={t}")

    mat_c2= dict(zip(times, mat))
    return mat_c2

def ranking_tracing_secnn(T, model, observations, params, noise = 1e-19, timing=False):
    """
    Contact Tracing up to second nearest neighbors
    
    params["tau"] = tau
    params["lamb"] = lamb
    
    Returns: ranked dataframe encounters[["i","rank","score","count"]]
    Authors: Sibyl-team
    """
    
    from collections import Counter
    import pickle, gzip
    
        
    tau = params["tau"]
    lamb = params["lamb"]
    
    if (T < tau):
        return ranking_random(T, model, observations, params)

    t0 = time.time()
    observ = pd.DataFrame(observations)
    if timing:
        print("t setup observ {:.3f} ms".format((time.time()-t0)*1000))

    if T - tau > T:
        ## save data
        print("SAVING")
        with gzip.open("transmissions_csr_gr.pk.gz","w") as f:
            pickle.dump(model.transmissions,f,protocol=4)
        observ.to_feather("observ.fth")
        print(T, model.N, params)

    t0 = time.time()
    observ = observ[(observ["t_test"] <= T)]
    temp =[]
    for t_contact in range(T-tau, T+1):
        d = get_nonzero_idx_d(model.transmissions[t_contact])
        temp.append(pd.DataFrame(dict(i=d[0],j=d[1],t=t_contact)))
    contacts = pd.concat(temp, ignore_index=True)
    if timing:
        print("t setup contacts {:.3f} ms".format((time.time()-t0)*1000))
    t0 = time.time()
    idx_R = observ[observ['s'] == 2]['i'].to_numpy() # observed R
    idx_I = observ[observ['s'] == 1]['i'].to_numpy() # observed I
    idx_S = observ[(observ['s'] == 0) & (observ['t_test'] == T)]['i'].to_numpy() # observed S at T -> put them at the tail of the ranking
    
    idx_alli = contacts['i'].unique()
    idx_allj = contacts['j'].unique()
    idx_all = np.union1d(idx_alli, idx_allj)
    idx_non_obs = np.setdiff1d(range(0,model.N), idx_all) # these have no contacts -> tail of the ranking
    
    idx_to_inf = np.setdiff1d(idx_all, idx_I) # rm I anytime
    idx_to_inf = np.setdiff1d(idx_to_inf, idx_S) # rm S at time T
    idx_to_inf = np.setdiff1d(idx_to_inf, idx_R) # rm R anytime
    if timing:
        print("t setup idx {:.3f} ms".format((time.time()-t0)*1000))
    t0 = time.time()
    
    maxS = -1 * np.ones(model.N)
    minR = T * np.ones(model.N)
    for i, s, t_test, in  observ[["i", "s", "t_test"]].to_numpy():
        if s == 0 and t_test < T:
            maxS[i] = max(maxS[i], t_test)
        if s == 2:
            minR[i] = min(minR[i], t_test)
        # I can consider a contact as potentially contagious if T > minR > t_contact > maxS,
        # the maximum time at which I am observed as S (for both infector and
        # infected)
    if timing:
        print("t first loop: {:.3f} ms".format((time.time()-t0)*1000))
    t0 = time.time()
        
    contacts_cut = contacts[(contacts["i"].isin(idx_to_inf)) \
                           & (contacts["j"].isin(idx_I))]

    Score = np.zeros(model.N)#dict([(i, 0) for i in range(model.N)])
    #Score = manager.dict([(i, 0) for i in range(model.N)])
    Count = np.zeros(model.N)#dict([(i, 0) for i in range(model.N)])
    contacts_cut2 = dict()
    if len(contacts_cut) > 0:
        # (i,j) are both unknown
        #print(datetime.datetime.now(), "make contacts_cut2")
        contacts_cut2 = contacts[(contacts["i"].isin(idx_to_inf)) \
                                & (contacts["j"].isin(idx_to_inf))]
        #print(datetime.datetime.now(), "end contacts_cut2")

    valid_idx_c1 = count_valid_c1(*[contacts_cut[k].to_numpy() for k in ("i","j","t")], maxS, minR)
    good_c1 = contacts_cut.iloc[valid_idx_c1]
    if timing:
        print("t contacts cut: {:.3f} ms".format((time.time()-t0)*1000))
    t0 = time.time()
    idxi_counts, icounts = np.unique(good_c1["i"], return_counts=True)

    #print("t loop contacts: {:.3f} ms".format((time.time()-t0)*1000))
    #t0 = time.time()
    #np.save("idxk_old",idxk)
    Count[idxi_counts] += icounts
    mrands = np.random.rand(len(good_c1))

    sum_score_c1(Score, mrands, good_c1["i"].to_numpy(), lamb, noise)

    mat_c1 = {}
    for t, gr in good_c1.groupby("t"):
        v = sparse.coo_matrix((np.ones(len(gr),np.int), (gr.i.to_numpy(), gr.j.to_numpy())), shape=(model.N, model.N))
        mat_c1[t] = v.tocsr()

    mat_c2 = create_mat_c2(contacts_cut2, model)
    sum_counts = None
    for t in sorted(mat_c1.keys()):
        ## select the rows (i) where the num is 0
        ## I have n_i rows 
        idx_i_c1= np.unique(mat_c1[t].nonzero()[0])      
        res = mat_c1[t][idx_i_c1,:].sum(1).T * mat_c2[t+1][idx_i_c1,:] ##vector product
        #print(res.shape)
        if sum_counts is None:
            sum_counts=res
        else:
            sum_counts+=res

    sec_NN = sum_counts.sum()
    sum_counts = np.array(sum_counts)[0]
    idx_nonzero_j = sum_counts.nonzero()[0]
    counts_j = sum_counts[idx_nonzero_j]

    #for (k, occk) in zip(idx_nonzero_j, counts_j):
    #    Score[k] += lamb*lamb*occk
    Score[idx_nonzero_j] += (lamb**2 * counts_j)
    
    if timing:
        print("t loop contacts: {:.3f} ms".format((time.time()-t0)*1000))
    t0 = time.time()

    print(f"first NN c: {len(contacts_cut)}. second NN c: {sec_NN}")
    
    for i in range(0,model.N):
        if i in idx_non_obs:
            Score[i] = -1 + np.random.rand() * noise
        if i in idx_I and i not in idx_R:
            Score[i] = model.N * observ[(observ['i'] == i) & (observ['s'] == 1)]['t_test'].max() + np.random.rand() * noise
        elif i in idx_S: #at time T
            Score[i] = -1 + np.random.rand() * noise
        elif i in idx_R: #anytime
            Score[i] = -1 + np.random.rand() * noise
    if timing:
        print("t final loop: {:.3f} ms".format((time.time()-t0)*1000))
    #print("Score; ", np.array([Score[v] for v in range(50)]))
    t0 = time.time()

    idxrank = np.argsort(Score)[::-1] ## reverse sort the scores (higher to lower)
    scores = Score[idxrank] ## get the scores
    count = Count[idxrank]

    #i rank score count
    encounters = pd.DataFrame({"i": idxrank, "rank": range(0,model.N), "score": scores, "count": count })
    if timing: print("t set output: {:.3f} ms".format((time.time()-t0)*1000))
    return encounters
