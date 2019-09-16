import numpy as np
import random
import matplotlib.pyplot as plt

def read():
    representations = []
    taux_clic = []
    with open("CTR.txt") as f:
        for line in f.readlines():
            num, repr, taux = line.split(':')
            representations.append(list(map(float, repr.split(';'))))
            taux_clic.append(list(map(float, taux.split(';'))))
    for l in taux_clic:
        assert len(l)==10, l
    return np.array(representations), np.array(taux_clic)

def random_strat(representations, taux_clic):
    return np.random.randint(0, len(taux_clic[0]), len(representations))
    
def static_best_strat(representations, taux_clic):
    """strategie qui triche"""
    best = np.argmax(np.sum(taux_clic, axis=0))
    return np.array([best] * len(representations))    
  
def optim_strat(representations, taux_clic):
    """strategie qui triche"""
    return np.argmax(taux_clic, axis=1)

def score(res, taux_clic):
    #return taux_clic[:, res].sum()
    s = 0
    for i in range(len(res)):
        s += taux_clic[i][res[i]]
    return s

def baselines(representations, taux_clic):
    n = len(representations)
    annonceurs = len(taux_clic[0])
    print("representations", representations.shape)
    print("taux_clic", taux_clic.shape)
    print(taux_clic)
    for strat in (random_strat, static_best_strat, optim_strat):
        print(strat)
        res = strat(representations, taux_clic)
        print(res.shape)
        print(res)
        assert res.shape == (n,) 
        #assert np.all(np.all(0 <= res < annonceurs))
        print(score(res, taux_clic))
        print()

def UCB(representations, taux_clic):
    n = len(representations)
    b = len(taux_clic[0])
    scores = [[] for _ in range(b)]

    def B(score, t):
        return np.mean(score) + np.sqrt(2 * np.log(t) / len(score))

    list_chosen = []
    chosen = None
    for iteration, line in enumerate(taux_clic):
        if iteration < b: # initialisation
            chosen = iteration
        else:
            upper_bounds = [B(score, iteration) for score in scores]
            chosen = np.argmax(upper_bounds)
        
        list_chosen.append(chosen)
        scores[chosen].append(line[chosen])

    return np.array(list_chosen)

def LinUCB(representations, taux_clic, alpha):
    list_chosen = []
    d = len(representations[0])
    na = len(taux_clic[0])
    A = [np.identity(d) for _ in range(na)]
    b = [np.zeros((d, 1)) for _ in range(na)]

    for iteration in range(len(representations)):
        x = representations[iteration]
        for action in range(na):
            inv_a = np.inv(A[action])
            theta = inv_a.dot(b[action])
            expected_payoff = theta.T.dot(x) +\
            alpha * np.sqrt(x.T.dot(inv_a).dot(x))
        
        chosen = np.argmax()
        list_chosen.append(chosen)
        scores[chosen].append(line[chosen])

    return np.array(list_chosen)

def plot(list_res, names, taux_clic):
    list_scores = [[taux_clic[i][res[i]] for i in range(len(taux_clic))] for res in list_res]
    scores_cumules = [np.cumsum(score) for score in list_scores]
    
    for score_cumule, name in zip(scores_cumules, names):
        plt.plot(score_cumule, label=name)
    plt.legend()
    plt.show()


def main():
    representations, taux_clic = read()
    print(representations.shape)
    n = len(representations)
    annonceurs = len(taux_clic[0])
    baselines(representations, taux_clic)

    res_UCB = UCB(representations, taux_clic)
    res_opti = optim_strat(representations, taux_clic)
    
    plot([res_UCB, res_opti], ["UCB", "opti"], taux_clic)
main()
