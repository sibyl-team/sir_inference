import numpy as np
import pandas as pd
import time
import numba as nb
import scipy.sparse as sparse
from inference_model import MeanField, DynamicMessagePassing
from sir_model import frequency, indicator
from scipy.sparse import csr_matrix
import sib



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

def create_mat_c2(contacts_cut2, model):
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
        print(f"Appeding contacts, t={t}")

    mat_c2= dict(zip(times, mat))
    return mat_c2

def csr_to_list(x):
    x_coo = x.tocoo()
    return zip(x_coo.row, x_coo.col, x_coo.data)


def ranking_inference(t, model, observations, params):
    """Inference starting from t_start.

    Run Mean Field from t_start to t, starting from all susceptible and
    resetting the probas according to observations.

    params["lamb"] : lamb
    params["mu"] : mu
    params["t_start"] : t_start
    params["tau"] : tau
    params["algo"] : "MF" (Mean Field) or "DMP" (Dynamic Message Passing)
    params["init"] : "all_S" (all susceptible) or "freqs" (frequency at t_start)

    Returns: ranked dataframe probas[["i","rank","score","p_I","p_R","p_S"]]
    If t < t_start cannot do the inference ranking, returns a random ranking.
    """
    t_start = params["t_start"]
    tau = params["tau"]
    mu = params["mu"]
    lamb = params["lamb"]
    if (t < t_start):
        return ranking_random(t, model, observations, params)
    algo = MeanField if params["algo"] == "MF" else DynamicMessagePassing
    if params["init"] == "all_S":
        initial_probas = indicator(np.zeros(model.N))
    else:
        initial_probas = frequency(model.states[t_start])
    infer = algo(initial_probas, model.x_pos, model.y_pos)
    # shift by t_start
    for obs in observations:
        obs["t"] = obs["t_test"] - t_start
        obs["t_I"] = obs["t"] - tau
    # set lambda and mu (general)
    rec_prob = mu*np.ones(model.N)
    transm = []
    for t0, A in enumerate(model.transmissions[t_start:t+1]):
        B = A.copy()
        B.tocsr()[B.nonzero()] = lamb
        transm.append(B)
    #infer.time_evolution(
    #    model.recover_probas, model.transmissions[t_start:t+1], observations,
    #    print_every=0
    #)
    infer.time_evolution(
        rec_prob, transm, observations, print_every=0)
    probas = pd.DataFrame(
        infer.probas[t-t_start, :, :],
        columns=["p_S", "p_I", "p_R"]
    )
    probas["i"] = range(model.N)
    # some i will have the same probas
    # -> we add a random value to shuffle the ranking
    probas["rand"] = np.random.rand(model.N)
    probas = probas.sort_values(by=["p_I", "rand"], ascending=False)
    probas.reset_index(drop=True, inplace=True)
    probas["rank"] = range(model.N)
    probas["score"] = probas["p_I"]
    return probas


def ranking_backtrack(t, model, observations, params):
    """Mean Field starting from t - delta.

    Run Mean Field from t - delta to t, starting from all susceptible and
    resetting the probas according to observations.
    
    params["lamb"] : lamb
    params["mu"] : mu
    params["delta"] : delta
    params["tau"] : tau
    params["algo"] : "MF" (Mean Field) or "DMP" (Dynamic Message Passing)
    params["init"] : "all_S" (all susceptible) or "freqs" (frequency at t_start)

    Returns: ranked dataframe probas[["i","rank","score","p_I","p_R","p_S"]]
    If t < delta cannot do the backtrack ranking, returns a random ranking.
    """
    delta = params["delta"]
    tau = params["tau"]
    mu = params["mu"]
    lamb = params["lamb"]
    #if (t < delta):
    #    return ranking_random(t, model, observations, params)
    t_start = max(t - delta, 0)
    algo = MeanField if params["algo"] == "MF" else DynamicMessagePassing
    if params["init"] == "all_S":
        initial_probas = indicator(np.zeros(model.N))
    else:
        initial_probas = frequency(model.states[t_start])
    infer = algo(initial_probas, model.x_pos, model.y_pos)
    # shift by t_start
    for obs in observations:
        obs["t"] = obs["t_test"] - t_start
        obs["t_I"] = obs["t"] - tau
    # set lambda and mu (general)
    rec_prob = mu*np.ones(model.N)
    transm = []
    for t0, A in enumerate(model.transmissions[t_start:t+1]):
        B = A.copy()
        B=B.tocsr()
        B[B.nonzero()] = lamb
        transm.append(B)
    #infer.time_evolution(
    #    model.recover_probas, model.transmissions[t_start:t+1], observations,
    #    print_every=0
    #)
    infer.time_evolution(
        rec_prob, transm, observations, print_every=0)
    probas = pd.DataFrame(
        infer.probas[t-t_start, :, :], columns=["p_S", "p_I", "p_R"]
    )
    probas["i"] = range(model.N)
    # some i will have the same probas
    # -> we add a random value to shuffle the ranking
    probas["rand"] = np.random.rand(model.N)
    probas = probas.sort_values(by=["p_I", "rand"], ascending=False)
    probas.reset_index(drop=True, inplace=True)
    probas["rank"] = range(model.N)
    probas["score"] = probas["p_I"]
    return probas


def ranking_random(t, model, observations, params):
    """Random ranking.

    Returns: ranked dataframe df[["i","rank","score"]]
    """
    ranked = np.random.permutation(model.N)
    df = pd.DataFrame({
        "i":ranked, "rank":range(model.N), "score":np.linspace(1, 0, model.N),"count":np.linspace(1, 0, model.N)
    })
    return df


def ranking_tracing(t, model, observations, params):
    """Naive contact tracing.

    Search for all individuals that have been in contact during [t-tau, t]
    with the individuals last tested positive (observations s=I at t_test=t-1).

    params["tau"] = tau

    Returns: ranked dataframe encounters[["i","rank","score","count"]]
    If t < tau cannot do the tracing ranking, returns a random ranking.
    """
    tau = params["tau"]
    if (t < tau):
        return ranking_random(t, model, observations, params)
    # last_tested : observations s=I for t-tau <= t_test < t
    last_tested = set(
        obs["i"] for obs in observations
        if obs["s"] == 1 and (t - tau <= obs["t_test"]) and (obs["t_test"] < t)
    )
    # contacts with last_tested people during [t - tau, t]
    contacts = pd.DataFrame(
        dict(i=i, j=j, t=t_contact)
        for t_contact in range(t - tau, t)
        for i, j, lamb in csr_to_list(model.transmissions[t_contact])
        if j in last_tested and lamb # lamb = 0 does not count
    )
    encounters = pd.DataFrame({"i": range(model.N)})
    # no encounters -> count = 0
    if (contacts.shape[0] == 0):
        encounters["count"] = 0
    else:
        counts = contacts.groupby("i").size() # number of encounters for all i
        encounters["count"] = encounters["i"].map(counts).fillna(0)
    # many i will have the same count
    # -> we add a random value to shuffle the ranking
    encounters["rand"] = np.random.rand(model.N)
    encounters = encounters.sort_values(by=["count", "rand"], ascending=False)
    encounters.reset_index(drop=True, inplace=True)
    encounters["rank"] = range(model.N)
    encounters["score"] = encounters["count"]
    return encounters

def upd_score(k,Score,lamb,noise):
    
    Score[k] += lamb*lamb + np.random.rand() * noise

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
    
    observ = pd.DataFrame(observations)
    observ = observ[(observ["t_test"] <= T)]
    contacts = pd.DataFrame(
        dict(i=i, j=j, t=t_contact)
        for t_contact in range(T - tau, T+1)
        for i, j, lamb in csr_to_list(model.transmissions[t_contact])
        if lamb # lamb = 0 does not count
    )
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
        
    contacts_cut = contacts[(contacts["i"].isin(idx_to_inf)) \
                           & (contacts["j"].isin(idx_I))]

    Score = dict([(i, 0) for i in range(model.N)])
    #Score = manager.dict([(i, 0) for i in range(model.N)])
    Count = dict([(i, 0) for i in range(model.N)])
    if len(contacts_cut):
        contacts_cut2 = contacts[(contacts["i"].isin(idx_to_inf)) \
                                & (contacts["j"].isin(idx_to_inf))]
    valid_idx_c1 = count_valid_c1(*[contacts_cut[k].to_numpy() for k in ("i","j","t")], maxS, minR)
    good_c1 = contacts_cut.iloc[valid_idx_c1]
    for i, j, t in good_c1[["i", "j", "t"]].to_numpy():
        # i to be estimated, j is infected
        Score[i] += lamb + np.random.rand() * noise
        Count[i] += 1.0
    mat_c1 = {}
    for t, gr in good_c1.groupby("t"):
        v = sparse.coo_matrix((np.ones(len(gr),np.int), (gr.i.to_numpy(), gr.j.to_numpy())), shape=(model.N, model.N))
        mat_c1[t] = v.tocsr()
    sec_NN = 0
    if len(contacts_cut):
        mat_c2 = create_mat_c2(contacts_cut2, model)
        sum_counts = None
        for t in sorted(mat_c1.keys()):
            ## select the rows (i) where the num is 0
            ## I have n_i rows 
            idx_i_c1= np.unique(mat_c1[t].nonzero()[0])      
            res = mat_c1[t][idx_i_c1,:].sum(1).T * mat_c2[t+1][idx_i_c1,:] ##vector product
            if sum_counts is None:
                sum_counts=res
            else:
                sum_counts+=res
        
        if sum_counts is not None:
            sec_NN = sum_counts.sum()
            sum_counts = np.array(sum_counts)[0]
            idx_nonzero_j = sum_counts.nonzero()[0]
            counts_j = sum_counts[idx_nonzero_j]

            for (k, occk) in zip(idx_nonzero_j, counts_j):
                Score[k] += lamb*lamb*occk 
    
    #print(datetime.datetime.now(), "end 2nd loop")

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
    sorted_Score = sorted(Score.items(),key=lambda item: item[1], reverse=True)
    idxrank = [item[0] for item in sorted_Score]
    scores = [item[1] for item in sorted_Score]
    count = [Count[i] for i in idxrank] 
    #i rank score count
    encounters = pd.DataFrame({"i": list(idxrank), "rank": range(0,model.N), "score": list(scores), "count": count })
    #print(encounters)
    #if save_data:
    #    encounters.to_csv("scores_counts_CT2.csv",index=False)
    return encounters

def ranking_tracing_backtrack(t, model, observations, params):
    """Naive contact tracing + backtrack

    First rank according to contact tracing (past contact or not), then by
    the MF/DMP probas.

    params["delta"] : delta
    params["tau"] : tau
    params["algo"] : "MF" (Mean Field) or "DMP" (Dynamic Message Passing)
    params["init"] : "all_S" (all susceptible) or "freqs" (frequency at t_start)

    Returns: ranked dataframe df[["i","rank","score","count","p_I","p_R","p_S"]]
    If t < t_start or t < tau cannot do the tracing + backtrack ranking,
    returns a random ranking.
    """
    tau = params["tau"]
    if (t < tau):
        return ranking_random(t, model, observations, params)
    delta = params["delta"]
    if (t < delta):
        return ranking_random(t, model, observations, params)
    encounters = ranking_tracing(t, model, observations, params)
    encounters.drop(columns=["rank","score"], inplace=True)
    probas = ranking_backtrack(t, model, observations, params)
    probas.drop(columns=["rank","score"], inplace=True)
    df = pd.merge(encounters, probas, on=["i"], how="inner")
    df["past_contact"] = 1*(df["count"] > 0)
    df["score"] = df["past_contact"] + df["p_I"]
    # some i will have the same score
    # -> we add a random value to shuffle the ranking
    df["rand"] = np.random.rand(model.N)
    df = df.sort_values(by=["score", "rand"], ascending=False)
    df.reset_index(drop=True, inplace=True)
    df["rank"] = range(model.N)
    return df





RANKINGS = {
    "tracing_backtrack": ranking_tracing_backtrack,
    "inference": ranking_inference,
    "backtrack": ranking_backtrack,
    "tracing": ranking_tracing,
    "random": ranking_random,
    "tracing2nd": ranking_tracing_secnn
}
