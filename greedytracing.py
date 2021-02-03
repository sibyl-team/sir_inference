
import time
import numpy as np
import pandas as pd
import numba as nb

from ranking import ranking_random, get_nonzero_idx_d

@nb.njit(parallel=True)
def _set_ranking(contacts_cut, contacts_c2, N, lamb, noise, maxS, minR):
    idxk = []
    #c2_i = contacts_c2[:,0]
    #count_all = np.zeros(N, dtype=np.int)
    #c2_t = contacts_c2[:,2]
    scores = np.zeros(N)
    counts = np.zeros(N)
    for l in nb.prange(contacts_cut.shape[0]):
        i = contacts_cut[l,0]
        j = contacts_cut[l,1]
        t = contacts_cut[l,2]
        # i to be estimated, j is infected
        if t > max(maxS[i], maxS[j]):
            if t < minR[j]:
                scores[i] += lamb + np.random.rand() * noise
                counts[i] += 1.0
                # get neighbors k from future contacts (i,k), from the set of the unknown nodes
                # column 0 is i, column 1 is j, column 2 is t
                midx = (contacts_c2[:,0] == i) & (contacts_c2[:,2] > max(t, maxS[i]))
                aux = contacts_c2[midx][:,1]
                idxk.append(aux)
                #count_all[aux] += 1

    #out_res = np.concatenate(idxk, axis=None)
    return scores, counts, idxk

@nb.njit()
def count_valid_c1(alli, allj, allt, maxS, minR):
    """
    Count valid contacts, given i, j, and t
    """
    res = np.zeros(alli.shape[0], dtype=np.bool_)
    for l in nb.prange(alli.shape[0]):
        i = alli[l]
        j = allj[l]
        t = allt[l]
        if t > max(maxS[i], maxS[j]):
            if t < minR[j]:
                res[l] = True

    return res

def ranking_tracing_secnn(T, model, observations, params, noise = 1e-19):
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
    
    print("t first loop: {:.3f} ms".format((time.time()-t0)*1000))
    t0 = time.time()
        
    contacts_cut = contacts[(contacts["i"].isin(idx_to_inf)) \
                           & (contacts["j"].isin(idx_I))]

    #Score = dict([(i, 0) for i in range(model.N)])
    #Score = manager.dict([(i, 0) for i in range(model.N)])
    #Count = dict([(i, 0) for i in range(model.N)])
    contacts_cut2 = contacts[(contacts["i"].isin(idx_to_inf)) \
                                & (contacts["j"].isin(idx_to_inf))]
        #print(datetime.datetime.now(), "end contacts_cut2")
    print("t contacts cut: {:.3f} ms".format((time.time()-t0)*1000))
    t0 = time.time()
    #print(c_cut2)
    Score = np.zeros(model.N)
    Count = np.zeros(model.N)
    ##contacts_c1 = contacts_cut.to_numpy()
    valid_c1 = count_valid_c1(*[contacts_cut[k].to_numpy() for k in ("i","j","t")], maxS, minR)
    valid_contacts_c1 = contacts_cut.iloc[valid_c1]
    t1 = time.time()-t0
    idxk = []
    for i, j, t in valid_contacts_c1[["i", "j", "t"]].to_numpy():
        # i to be estimated, j is infected
        if t > max(maxS[i], maxS[j]):
            if t < minR[j]:
                Score[i] += lamb + np.random.rand() * noise
                Count[i] += 1.0
                # get neighbors k from future contacts (i,k), from the set of the unknown nodes
                #aux = contacts_cut2[i][(contacts_cut2["t"] > max(t, maxS[i]))]["j"].to_numpy()
                aux = contacts_cut2[(contacts_cut2["i"] == i) & (contacts_cut2["t"] > max(t, maxS[i]) )]["j"].to_numpy()
                idxk = np.concatenate((idxk, aux), axis = None).astype(np.int32)
    t2 = time.time()-t0
    t0 = time.time()
    #idxk = np.concatenate(allidx, axis=None)
    #Score = pd.Series(Score)
    #print("t loop contacts: {:.3f} ms".format((time.time()-t0)*1000))
    #t0 = time.time()
    #np.save("idxk_new",idxk)
    t3 = time.time() -t0
    t0 = time.time()
    sec_NN = len(idxk)
    value_occ = Counter(idxk).items()
    t4 = time.time() - t0
    t0 = time.time()
    
    for (k, occk) in value_occ:
        Score[k] += lamb*lamb*occk 
    tf = time.time() -t0
    times = np.array(tuple((t1,t2,t3,t4,tf)))
    print("t contacts loop: {:.3f} ms".format((times.sum())*1000))
    print("times: ",times*1000)
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
    print("t final loop: {:.3f} ms".format((time.time()-t0)*1000))
    #print("Score: ", Score[:50])
    t0 = time.time()
    #sorted_Score = #sorted(Score.items(),key=lambda item: item[1], reverse=True)
    idxrank = np.argsort(Score)[::-1]
    scores = Score[idxrank]
    count = Count[idxrank]
    #print("Scoreidx: ", idxrank[:50])
    #idxrank = [item[0] for item in sorted_Score]
    #scores = [item[1] for item in sorted_Score]
    #count = [Count[i] for i in idxrank] 

    #i rank score count
    
    encounters = pd.DataFrame({"i": idxrank, "rank": range(0,model.N), "score": scores, "count": count })
    print("t set output: {:.3f} ms".format((time.time()-t0)*1000))
    return encounters