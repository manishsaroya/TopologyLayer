"""
This file save the ground truth persistence homology such that we don't have to
compute it while training.

Author: Manish Saroya
Contact: saroyam@orgonstate.edu

"""
import pickle 
import numpy as np 
import torch, torch.nn as nn
from topologylayer.nn import LevelSetLayer2D
import matplotlib.pyplot as plt
import pdb
def reduceinfo(info):
    r = []
    for i in info:
        if abs(i[0]) != np.inf and abs(i[1])!= np.inf and i[0]!=i[1]:
            r.append(i.detach().numpy())
    return r

class PersistenceDgm(nn.Module):
    def __init__(self, size):
        super(PersistenceDgm, self).__init__()
        self.pdfn = LevelSetLayer2D(size=size, sublevel=False)

    def dgmplot(self, image):
        dgminfo = self.pdfn(image)
        return dgminfo

    def generatePersistence(self, data, type_):
        p = []
        print("Computing persistence for ", type_)
        for i in range(len(data)):
        	ground_t = torch.tensor(data[i], dtype=torch.float, requires_grad=False)
        	dgm = self.dgmplot(ground_t)
        	p.append(dgm)
        	# z = np.asarray(reduceinfo(dgm[0][0]))
        	# f = np.asarray(reduceinfo(dgm[0][1]))
        	# if i%100==0:
        	# 	pdb.set_trace()
        	print(
            '\r[Generating persistence {} of {}]'.format(
                i,
                int(len(data)),
            ),
            end=''
            )
        return p


size = 32
with open('ground_truth_dataset_{}.pickle'.format(size),'rb') as tf:
	groundTruthData = pickle.load(tf)

pobj = PersistenceDgm((size,size))
#dgm = pobj.dgmplot(ground_t)

persistence = {}
persistence["train"] = pobj.generatePersistence(groundTruthData["train"], "training")
persistence["validation"] = pobj.generatePersistence(groundTruthData["validation"], "validation")
persistence["test"] = pobj.generatePersistence(groundTruthData["test"], "test")

with open('ground_truth_dataset_peristenceDgm{}.pickle'.format(32), 'wb') as handle:
	pickle.dump(persistence, handle)

#pdb.set_trace()


