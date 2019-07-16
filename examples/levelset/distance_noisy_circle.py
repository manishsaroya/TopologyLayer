from __future__ import print_function
import torch, torch.nn as nn, numpy as np, matplotlib.pyplot as plt
from topologylayer.nn import LevelSetLayer2D, SumBarcodeLengths, PartialSumBarcodeLengths
import pdb
# generate circle on grid
# generate circle on grid
n = 50
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

m = 1500
X = np.random.randn(m, n**2)
y = X.dot(beta.flatten()) + 0.05*np.random.randn(m)
beta_ols = (np.linalg.lstsq(X, y, rcond=None)[0]).reshape(n,n)

class TopLoss(nn.Module):
    def __init__(self, size):
        super(TopLoss, self).__init__()
        self.pdfn = LevelSetLayer2D(size=size,  sublevel=False)
        self.topfn = PartialSumBarcodeLengths(dim=1, skip=1)
        self.topfn2 = SumBarcodeLengths(dim=0)

    def forward(self, beta, ground):
        dgminfo = self.pdfn(beta)
        ################## Debugging ###################
        count = 0
        total_start_count = 0
        total_end_count = 0
        for i in range(dgminfo[0][0].shape[0]):
            if abs(dgminfo[0][0][i][0])==np.inf:
                total_start_count = total_start_count +1
            if abs(dgminfo[0][0][i][1])==np.inf:
                total_end_count = total_end_count +1
            if dgminfo[0][0][i][0]==dgminfo[0][0][i][1]:
                count = count + 1
        #pdb.set_trace()
        dgminfo_g = self.pdfn(ground)
        count_g = 0
        total_start_count_g = 0
        total_end_count_g = 0
        for i in range(dgminfo_g[0][0].shape[0]):
            if abs(dgminfo_g[0][0][i][0])==np.inf:
                total_start_count_g = total_start_count_g +1
            if abs(dgminfo_g[0][0][i][1])==np.inf:
                total_end_count_g = total_end_count_g +1
            if dgminfo_g[0][0][i][0]==dgminfo_g[0][0][i][1]:
                count_g = count_g + 1
        #pdb.set_trace()

        ############ Code starts ##########################
        ordered_prediction = []
        ordered_ground_truth = []
        data = np.copy(dgminfo)
        for i in dgminfo[0][0]:
            # find j which is closest to i
            if abs(i[0]) != np.inf and abs(i[1])!= np.inf and i[0]!=i[1]:
                dist = np.inf
                shortest = None
                index = 0
                zero_hom_g = dgminfo_g[0][0]
                zero_hom = i
                if len(zero_hom_g) > 0:
                    for j in range(len(zero_hom_g)):
                        if abs(zero_hom_g[j][0]) != np.inf and abs(zero_hom_g[j][1])!= np.inf and zero_hom_g[j][0]!=zero_hom_g[j][1]:
                            #pdb.set_trace()
                            if torch.norm(zero_hom-zero_hom_g[j], 2) < dist:
                                dist = torch.norm(zero_hom-zero_hom_g[j])
                                shortest = zero_hom_g[j]
                                index = j
                    zero_hom_g = zero_hom_g[zero_hom_g!=shortest]
                    #zero_hom_g.pop(index)
                    #pdb.set_trace()
                    ordered_prediction.append(zero_hom)
                    ordered_ground_truth.append(shortest)
                    #torch.cat([ordered_prediction,zero_hom],0)
                    #torch.cat((ordered_ground_truth,shortest),0)
                else:
                    #torch.cat((ordered_prediction,zero_hom),0)
                    #torch.cat((ordered_ground_truth,[torch.mean(zero_hom), torch.mean(zero_hom)]),0)
                    ordered_prediction.append(zero_hom)
                    ordered_ground_truth.append([torch.mean(zero_hom), torch.mean(zero_hom)])
        pdb.set_trace() 

        final_loss = torch.norm(torch.stack(ordered_prediction)- torch.stack(ordered_ground_truth))


        return self.topfn(dgminfo) + self.topfn2(dgminfo)

tloss = TopLoss((50,50)) # topology penalty
dloss = nn.MSELoss() # data loss

beta_t = torch.autograd.Variable(torch.tensor(beta_ols).type(torch.float), requires_grad=True)
X_t = torch.tensor(X, dtype=torch.float, requires_grad=False)
y_t = torch.tensor(y, dtype=torch.float, requires_grad=False)
ground_t = torch.tensor(beta, dtype=torch.float, requires_grad=False)
optimizer = torch.optim.Adam([beta_t], lr=1e-2)
for i in range(500):
    optimizer.zero_grad()
    tlossi = tloss(beta_t, ground_t)
    dlossi = dloss(y_t, torch.matmul(X_t, beta_t.view(-1)))
    loss = 0.1*tlossi + dlossi
    loss.backward()
    optimizer.step()
    if (i % 10 == 0):
        print(i, tlossi.item(), dlossi.item())


# save figure
beta_est = beta_t.detach().numpy()
fig, ax = plt.subplots(ncols=3, figsize=(15,5))
ax[0].imshow(beta)
ax[0].set_title("Truth")
ax[1].imshow(beta_ols)
ax[1].set_title("OLS")
ax[2].imshow(beta_est)
ax[2].set_title("Topology Regularization")
for i in range(3):
    ax[i].set_yticklabels([])
    ax[i].set_xticklabels([])
    ax[i].tick_params(bottom=False, left=False)
plt.savefig('noisy_circle.png')
