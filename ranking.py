import time
import numpy as np
import pandas as pd
from inference_model import MeanField, DynamicMessagePassing
from sir_model import frequency, indicator
import sib


def csr_to_list(x):
    x_coo = x.tocoo()
    return zip(x_coo.row, x_coo.col, x_coo.data)


def ranking_inference(t, model, observations, params):
    """Inference starting from t_start.

    Run Mean Field from t_start to t, starting from all susceptible and
    resetting the probas according to observations.

    params["t_start"] : t_start
    params["tau"] : tau
    params["algo"] : "MF" (Mean Field) or "DMP" (Dynamic Message Passing)
    params["init"] : "all_S" (all susceptible) or "freqs" (frequency at t_start)

    Returns: ranked dataframe probas[["i","rank","score","p_I","p_R","p_S"]]
    If t < t_start cannot do the inference ranking, returns a random ranking.
    """
    t_start = params["t_start"]
    tau = params["tau"]
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
    infer.time_evolution(
        model.recover_probas, model.transmissions[t_start:t+1], observations,
        print_every=0
    )
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

    params["delta"] : delta
    params["tau"] : tau
    params["algo"] : "MF" (Mean Field) or "DMP" (Dynamic Message Passing)
    params["init"] : "all_S" (all susceptible) or "freqs" (frequency at t_start)

    Returns: ranked dataframe probas[["i","rank","score","p_I","p_R","p_S"]]
    If t < delta cannot do the backtrack ranking, returns a random ranking.
    """
    delta = params["delta"]
    tau = params["tau"]
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
    infer.time_evolution(
        model.recover_probas, model.transmissions[t_start:t+1], observations,
        print_every=0
    )
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

def get_nonzero_idx_d(x):
    idx = x.tocoo().nonzero()
    y = x.tocsr()
    return *idx, np.array(y[idx])

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

    Score = dict([(i, 0) for i in range(model.N)])
    #Score = manager.dict([(i, 0) for i in range(model.N)])
    Count = dict([(i, 0) for i in range(model.N)])
    contacts_cut2 = dict()
    if len(contacts_cut) > 0:
        # (i,j) are both unknown
        #print(datetime.datetime.now(), "make contacts_cut2")
        #for i in idx_to_inf:
        #    contacts_cut2[i] = contacts[(contacts["i"] == i) \
        #                         & (contacts["j"].isin(idx_to_inf))]
        contacts_cut2 = contacts[(contacts["i"].isin(idx_to_inf)) \
                                & (contacts["j"].isin(idx_to_inf))]
        #print(datetime.datetime.now(), "end contacts_cut2")
    print("t contacts cut: {:.3f} ms".format((time.time()-t0)*1000))
    t0 = time.time()
    idxk = []
    for i, j, t in contacts_cut[["i", "j", "t"]].to_numpy():
        # i to be estimated, j is infected
        if t > max(maxS[i], maxS[j]):
            if t < minR[j]:
                Score[i] += lamb + np.random.rand() * noise
                Count[i] += 1.0
                # get neighbors k from future contacts (i,k), from the set of the unknown nodes
                #aux = contacts_cut2[i][(contacts_cut2["t"] > max(t, maxS[i]))]["j"].to_numpy()
                aux = contacts_cut2[(contacts_cut2["i"] == i) & (contacts_cut2["t"] > max(t, maxS[i]) )]["j"].to_numpy()
                idxk = np.concatenate((idxk, aux), axis = None)
    #print("t loop contacts: {:.3f} ms".format((time.time()-t0)*1000))
    #t0 = time.time()
    #np.save("idxk_old",idxk)
    sec_NN = len(idxk)
    value_occ = Counter(idxk).items()
    
    for (k, occk) in value_occ:
        Score[k] += lamb*lamb*occk
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
    print("t final loop: {:.3f} ms".format((time.time()-t0)*1000))
    #print("Score; ", np.array([Score[v] for v in range(50)]))
    t0 = time.time()
    sorted_Score = sorted(Score.items(),key=lambda item: item[1], reverse=True)
    idxrank = [item[0] for item in sorted_Score]
    scores = [item[1] for item in sorted_Score]
    #print("Score idx: ", np.array([idxrank[v] for v in range(50)]))
    count = [Count[i] for i in idxrank] 

    #i rank score count
    encounters = pd.DataFrame({"i": list(idxrank), "rank": range(0,model.N), "score": list(scores), "count": count })
    print("t set output: {:.3f} ms".format((time.time()-t0)*1000))
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
