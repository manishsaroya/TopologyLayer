from __future__ import print_function
import torch, torch.nn as nn, numpy as np, matplotlib.pyplot as plt
from topologylayer.nn import LevelSetLayer2D, SumBarcodeLengths, PartialSumBarcodeLengths
import pdb
import argparse
import datetime
import os
from tensorboardX import SummaryWriter
from scipy.optimize import linear_sum_assignment
import time

# generate circle on grid
# generate circle on grid
n = 20
def circlefn(i, j, n):
    r = np.sqrt((i - n/2.)**2 + (j - n/2.)**2)
    return np.exp(-(r - n/3.)**2/(n*2))

def gen_circle(n):
    beta = np.empty((n,n))
    for i in range(n):
        for j in range(n):
            beta[i,j] = circlefn(i,j,n)
    return beta

beta = gen_circle(n)


##############PARAMETERS ##############
parser = argparse.ArgumentParser()
parser.add_argument('--log_dir_top', type=str, default='./logs/top/'+ datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")+'/')
parser.add_argument('--log_dir_mse', type=str, default='./logs/mse/'+ datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")+'/')
parser.add_argument('--runs',type=int,default=1)
args = parser.parse_args()

# Make log directory for logging losses.
if not os.path.exists(args.log_dir_top):
    os.makedirs(args.log_dir_top)
    os.makedirs(args.log_dir_top+'imgs/')
    writer_top = SummaryWriter(logdir= args.log_dir_top)

if not os.path.exists(args.log_dir_mse):
    os.makedirs(args.log_dir_mse)
    os.makedirs(args.log_dir_mse+'imgs/')
    writer_mse = SummaryWriter(logdir= args.log_dir_mse)

m = 350
X = np.random.randn(m, n**2)
y = X.dot(beta.flatten()) + 0.05*np.random.randn(m)
beta_ols = (np.linalg.lstsq(X, y, rcond=None)[0]).reshape(n,n)

def reduceinfo(info):
    r = []
    for i in info:
        if abs(i[0]) != np.inf and abs(i[1])!= np.inf and i[0]!=i[1]:
            r.append(i.detach().numpy())
    return r


class PersistenceDgm(nn.Module):
    def __init__(self, size):
        super(PersistenceDgm, self).__init__()
        self.pdfn = LevelSetLayer2D(size=size,  sublevel=False)

    def dgmplot(self, image):
        dgminfo = self.pdfn(image)
        #pdb.set_trace()
        z = np.asarray(reduceinfo(dgminfo[0][0]))
        f = np.asarray(reduceinfo(dgminfo[0][1]))
        return z, f

class TopLoss(nn.Module):
    def __init__(self, size, skip=1):
        super(TopLoss, self).__init__()
        self.pdfn = LevelSetLayer2D(size=size,  sublevel=False)
        self.pdfn_g = LevelSetLayer2D(size=size, sublevel=False)
        self.topfn = PartialSumBarcodeLengths(dim=1, skip=skip)
        self.topfn2 = SumBarcodeLengths(dim=0)

    def correspondence(self, reduced_dgminfo, reduced_dgminfo_g):
        ################### Hungarian algorithm ###############
        #creating a cost matrix with rows containing ground persistence and columns prediction persistence
        cost = []
        for i in reduced_dgminfo:
            row = []
            for j in reduced_dgminfo_g:
                row.append(torch.norm(i-j).detach().numpy())
            cost.append(row)
        #pdb.set_trace()
        return linear_sum_assignment(cost)
        #######################################################


    def computeloss(self, dgminfohom, dgminfohom_g):
        #ordered_prediction = []
        ordered_ground_truth = []
        # clean up the dgm info
        reduced_dgminfo = []
        reduced_dgminfo_g = []
        for i in dgminfohom:
            if abs(i[0]) != np.inf and abs(i[1])!= np.inf and i[0]!=i[1]:
                reduced_dgminfo.append(i)
        for i in dgminfohom_g:
            if abs(i[0]) != np.inf and abs(i[1])!= np.inf and i[0]!=i[1]:
                reduced_dgminfo_g.append(i)
        reduced_dgminfo = torch.stack(reduced_dgminfo)
        reduced_dgminfo_g = torch.stack(reduced_dgminfo_g)
        
        #pdb.set_trace()
        # get correspondence between persistence points
        p_ind, g_ind = self.correspondence(reduced_dgminfo, reduced_dgminfo_g)
        # fill mean in all ground truth
        for i in reduced_dgminfo:
            ordered_ground_truth.append(torch.stack([torch.mean(i), torch.mean(i)]))  # stack not required we don't care for ground backprop.

        for i in range(len(p_ind)):
            ordered_ground_truth[p_ind[i]] = reduced_dgminfo_g[g_ind[i]]

        final_loss = torch.norm(reduced_dgminfo - torch.stack(ordered_ground_truth))
        return final_loss

    def forward(self, beta, ground):
        dgminfo = self.pdfn(beta)
        dgminfo_g = self.pdfn_g(ground)

        ############ Code starts ##########################
        zero_loss = self.computeloss(dgminfo[0][0],dgminfo_g[0][0])
        one_loss = self.computeloss(dgminfo[0][1],dgminfo_g[0][1])
        return zero_loss + one_loss #zero_loss #self.topfn(dgminfo) + self.topfn2(dgminfo)

def savepersistence(beta_t, ground_t, beta, beta_ols):
    ########### save persistence Diagram #######
    outplot = PersistenceDgm((20,20))
    z, f = outplot.dgmplot(beta_t)
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(15,10))
    ax[0][0].set(xlim=(-1.1, 1.1), ylim=(-1.1, 1.1))
    ax[0][0].plot(z[:,0], z[:,1],'bo')
    ax[0][0].plot(f[:,0], f[:,1],'ro')
    ax[0][0].set_title("output PersistenceDgm")
    
    inplot = PersistenceDgm((20,20))
    z, f = inplot.dgmplot(ground_t)
    ax[0][1].set(xlim=(-1.1, 1.1), ylim=(-1.1, 1.1))
    ax[0][1].plot(z[:,0], z[:,1],'bo')
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
    plt.savefig(args.log_dir_top+'imgs/'+'persistence_dgm'+str(t)+'.png')

tloss = TopLoss((20,20)) # topology penalty
dloss = nn.MSELoss() # data loss

beta_t = torch.autograd.Variable(torch.tensor(beta_ols).type(torch.float), requires_grad=True)
X_t = torch.tensor(X, dtype=torch.float, requires_grad=False)
y_t = torch.tensor(y, dtype=torch.float, requires_grad=False)
ground_t = torch.tensor(beta, dtype=torch.float, requires_grad=False)
optimizer = torch.optim.Adam([beta_t], lr=1e-2)

for i in range(1500):
    optimizer.zero_grad()
    tlossi = tloss(beta_t, ground_t)
    dlossi = dloss(y_t, torch.matmul(X_t, beta_t.view(-1)))
    loss = 0.02*tlossi + dlossi
    loss.backward()
    optimizer.step()

    writer_top.add_scalar('loss', tlossi.data.item(), i)
    writer_mse.add_scalar('loss',dlossi.data.item(),i)

    if (i % 10 == 0):
        print(i, tlossi.item(), dlossi.item())

    if (i%10 ==0):
        savepersistence(beta_t, ground_t, beta, beta_ols)

writer_top.close()
writer_mse.close()

savepersistence(beta_t, ground_t, beta, beta_ols)
