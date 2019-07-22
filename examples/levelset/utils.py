import torch, torch.nn as nn, numpy as np, matplotlib.pyplot as plt
from topological_loss import PersistenceDgm
import time
def circlefn(i, j, n):
    r = np.sqrt((i - n/2.)**2 + (j - n/2.)**2)
    return np.exp(-(r - n/3.)**2/(n*2))


def gen_circle(n):
    beta = np.empty((n,n))
    for i in range(n):
        for j in range(n):
            beta[i,j] = circlefn(i,j,n)
    return beta

def savepersistence(n, beta_t, ground_t, beta, beta_ols, path):
    ########### save persistence Diagram #######
    outplot = PersistenceDgm((n,n))
    z, f = outplot.dgmplot(beta_t)
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(15,10))
    ax[0][0].set(xlim=(-1.1, 1.1), ylim=(-1.1, 1.1))
    if len(z)>0:
    	ax[0][0].plot(z[:,0], z[:,1],'bo')
    if len(f)>0:
    	ax[0][0].plot(f[:,0], f[:,1],'ro')
    ax[0][0].set_title("output PersistenceDgm")
    
    inplot = PersistenceDgm((n,n))
    z, f = inplot.dgmplot(ground_t)
    ax[0][1].set(xlim=(-1.1, 1.1), ylim=(-1.1, 1.1))
    if len(z)>0:
    	ax[0][1].plot(z[:,0], z[:,1],'bo')
    if len(f)>0:
    	ax[0][1].plot(f[:,0], f[:,1],'ro')
    ax[0][1].set_title("Ground Truth PersistenceDgm")
 
    for i in range(2):
        ax[0][i].set_xlabel('Death')
        ax[0][i].set_ylabel('Birth')

    ############ save outputs #############
    beta_est = beta_t.detach().numpy()
    ax[1][0].imshow(beta)
    ax[1][0].set_title("Truth")
    ax[1][1].imshow(beta_ols)
    ax[1][1].set_title("OLS")
    ax[1][2].imshow(beta_est)
    ax[1][2].set_title("Topology Regularization")
    for i in range(3):
        ax[1][i].set_yticklabels([])
        ax[1][i].set_xticklabels([])
        ax[1][i].tick_params(bottom=False, left=False)
    t = time.time()
    plt.savefig(path+'imgs/'+'persistence_dgm'+str(t)+'.png')