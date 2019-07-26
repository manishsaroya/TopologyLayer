from __future__ import print_function
import torch, torch.nn as nn, numpy as np, matplotlib.pyplot as plt
from topological_loss import PersistenceDgm, TopLoss
from utils import savepersistence, gen_circle, circlefn
import pdb
import argparse
import datetime
import os
from tensorboardX import SummaryWriter
from x_map_gen import Exploration
import pickle

##############PARAMETERS ##############
n = 32
m = 700
parser = argparse.ArgumentParser()
parser.add_argument('--log_dir_top', type=str, default='./logs/top/'+ datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")+'/')
parser.add_argument('--log_dir_mse', type=str, default='./logs/mse/'+ datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")+'/')
parser.add_argument('--runs',type=int,default=1)
args = parser.parse_args()
#######################################

############## LOGGING INFO ###########
# Make log directory for logging losses.
if not os.path.exists(args.log_dir_top):
    os.makedirs(args.log_dir_top)
    os.makedirs(args.log_dir_top+'imgs/')
    writer_top = SummaryWriter(logdir= args.log_dir_top)

if not os.path.exists(args.log_dir_mse):
    os.makedirs(args.log_dir_mse)
    os.makedirs(args.log_dir_mse+'imgs/')
    writer_mse = SummaryWriter(logdir= args.log_dir_mse)
########################################


# generate circle on grid
beta = gen_circle(n)
#explore = Exploration(32, 15, 0.7)
#beta = explore.generate_map()

#with open('ground_truth_{}.pickle'.format(32), 'wb') as handle:
#    pickle.dump(beta, handle)
#with open('ground_truth_{}.pickle'.format(32),'rb') as tf:
#        beta = pickle.load(tf)
#plt.imshow(beta)
#plt.show()
#pdb.set_trace()

X = np.random.randn(m, n**2)
y = X.dot(beta.flatten()) + 0.05*np.random.randn(m)
beta_ols = (np.linalg.lstsq(X, y, rcond=None)[0]).reshape(n,n)

tloss = TopLoss((n,n)) # topology penalty
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
        savepersistence(n, beta_t, ground_t, beta, beta_ols, args.log_dir_top)

writer_top.close()
writer_mse.close()

savepersistence(n, beta_t, ground_t, beta, beta_ols, args.log_dir_top)
