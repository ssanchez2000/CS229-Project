'''
Copyright 2016 Jihun Hamm
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
       http://www.apache.org/licenses/LICENSE-2.0
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License
'''

import numpy as np
import scipy.io
import src.minimaxFilter as minimaxFilter
from filterAlg_NN import NN1
from src.learningAlg import mlogreg


##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

mat = scipy.io.loadmat('genki.mat')
#nsubjs = np.asscalar(mat['nsubjs'])
K1 = np.asscalar(mat['K1'])
K2 = np.asscalar(mat['K2'])

D = np.asscalar(mat['D'])
Ntrain = np.asscalar(mat['Ntrain'])
Ntest = np.asscalar(mat['Ntest'])
N = Ntrain + Ntest
y1_train = mat['y1_train']-1
y1_test = mat['y1_test']-1
y1 = np.hstack((y1_train,y1_test))
del y1_train, y1_test
y2_train = mat['y2_train']-1
y2_test = mat['y2_test']-1
y2 = np.hstack((y2_train,y2_test))
del y2_train, y2_test

Xtrain = mat['Xtrain']
Xtest = mat['Xtest']
X = np.hstack((Xtrain,Xtest))
del Xtrain, Xtest

ind_train_dom1 = [[range(Ntrain)]]
ind_test_dom1 = [[range(Ntrain,Ntrain+Ntest)]]


##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

ntrials = 1
ds = [10]#[10,20,50,100]

lambda0 = 1E-6
lambda1 = 1E-6
lambda2 = 1E-6

maxiter_minimax = 100
maxiter_final = 50
rho = 10.

##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##%% Minimax + NN1

rates1_minimax2 = np.nan*np.ones((len(ds),ntrials))
rates2_minimax2 = np.nan*np.ones((len(ds),ntrials))

W0_minimax2 = [[[] for i in range(ntrials)] for j in range(len(ds))]


for trial in range(ntrials):
    for j in range(len(ds)):
        d = ds[j]
        nhs = [20,d] #
        # Nick added 'd' to the dictionary of hparams0
        hparams0 = {'D':D, 'nhs':nhs, 'activation':'sigmoid', 'l':lambda0, 'd':d}
        hparams1 = {'K':K1, 'l':lambda1, 'd':d}
        hparams2 = {'K':K2,'l':lambda2, 'd':d}

        if False:
            W0 = NN1.init(hparams0)
        else:
            print('Pre-training by autoencoder')
            W0 = NN1.initByAutoencoder(X[:,ind_train_dom1[trial][0]].squeeze(),hparams0)
            
        W1 = mlogreg.init(hparams1)
        W2 = mlogreg.init(hparams2)

        for iter in range(maxiter_minimax):
            if True:#iter==maxiter_minimax-1:
                G_train = NN1.g(W0,X[:,ind_train_dom1[trial][0]].squeeze(),hparams0)
                #% Full training
                tW1,f1 = mlogreg.train(G_train,y1[:,ind_train_dom1[trial][0]].squeeze(),hparams1,None,maxiter_final)
                tW2,f2 = mlogreg.train(G_train,y2[:,ind_train_dom1[trial][0]].squeeze(),hparams2,None,maxiter_final)

                #% Testing error
                G_test = NN1.g(W0,X[:,ind_test_dom1[trial][0]].squeeze(),hparams0)

                rate1,_ = mlogreg.accuracy(tW1,G_test,y1[:,ind_test_dom1[trial][0]].squeeze())
                rate2,_ = mlogreg.accuracy(tW2,G_test,y2[:,ind_test_dom1[trial][0]].squeeze())

                print('minimax (NN): rho=%f, d=%d, trial=%d, rate1=%f, rate2=%f\n' % \
                    (rho,d,trial,rate1,rate2))

                rates1_minimax2[j,trial] = rate1
                rates2_minimax2[j,trial] = rate2
                
                W0_minimax2[j][trial] = W0
            
            W0,W1,W2 = minimaxFilter.run(W0,W1,W2,rho,'alt',1,\
                X[:,ind_train_dom1[trial][0]].squeeze(), \
                y1[:,ind_train_dom1[trial][0]].squeeze(),\
                y2[:,ind_train_dom1[trial][0]].squeeze(),\
                NN1,mlogreg,mlogreg,\
                hparams0,hparams1,hparams2)


'''
>>> rates1_minimax2
array([[ 0.89 ],
       [ 0.88 ],
       [ 0.895],
       [ 0.885]])
>>> rates2_minimax2
array([[ 0.6  ],
       [ 0.565],
       [ 0.64 ],
       [ 0.64 ]])
'''


np.savez('test_NN_genki.npz',\
    W0_minimax2=[W0_minimax2], \
    rates1_minimax2=[rates1_minimax2],\
    rates2_minimax2=[rates2_minimax2]\
    )



