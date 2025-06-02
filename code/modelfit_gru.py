import numpy as np
import scipy
import pandas as pd
import os
from sklearn.decomposition import PCA
import torch
import sys

directory = '/Users/hayoungsong/Documents/_postdoc/modelmind/github'
directory_output = '/Users/hayoungsong/Documents/_postdoc/modelmind/output/pc50_season'
sys.path.append(directory+'/model')
from emKeyValue import emKeyValue
from gru import gru

def conv_r2z(r):
    with np.errstate(invalid='ignore', divide='ignore'):
        return 0.5 * (np.log(1 + r) - np.log(1 - r))
def conv_z2r(z):
    with np.errstate(invalid='ignore', divide='ignore'):
        return (np.exp(2 * z) - 1) / (np.exp(2 * z) + 1)


seed = int(sys.argv[1])
nPC = 50
torch.manual_seed(seed), np.random.seed(seed)

##########################################################
# data load
##########################################################
def create_train_test_episode(directory=directory, nPC=50):
    train_data, train_idd = [], []
    for ep in range(2, 18+1):
        if ep<10: emb = np.load(directory+'/clip/S01E0'+str(ep)+'.npz')['frames']
        else: emb = np.load(directory+'/clip/S01E'+str(ep)+'.npz')['frames']
        train_data.append(emb), train_idd.append(np.repeat(ep, emb.shape[0]))
    train_data, train_idd = np.concatenate(train_data, axis=0), np.concatenate(train_idd, axis=0)

    test_scene_order = {1: np.array([19, 10,  1, 32, 44, 16, 13,  3, 37,  8, 24,  7, 18, 45, 30, 39, 15, 36, 42, 11, 27,  9, 23, 21, 12, 33,  6, 22, 14, 28, 46, 47, 48, 34, 38, 31,  5, 26, 40, 17, 41,  4, 20, 35, 29,  2, 25, 43]),
                        2: np.array([35, 29,  2, 25, 43, 33,  6, 22, 14, 28, 34, 38, 31,  5, 26, 24,  7, 18, 45, 30, 16, 13,  3, 37,  8, 40, 17, 41,  4, 20, 46, 47, 48, 27, 9, 23, 21, 12, 19, 10,  1, 32, 44, 39, 15, 36, 42, 11]),
                        3: np.array([27,  9, 23, 21, 12, 39, 15, 36, 42, 11, 40, 17, 41,  4, 20, 19, 10, 1, 32, 44, 35, 29,  2, 25, 43, 34, 38, 31,  5, 26, 46, 47, 48, 16, 13,  3, 37,  8, 33,  6, 22, 14, 28, 24,  7, 18, 45, 30])}

    test_data, test_idd = [], []
    for scc in range(1, 48+1):
        emb = np.load(directory+'/clip/scene'+str(scc)+'.npz')['frames']
        test_data.append(emb), test_idd.append(np.repeat(scc, emb.shape[0]))
    test_data, test_idd = np.concatenate(test_data, axis=0), np.concatenate(test_idd, axis=0)

    train_mu_, train_sd_ = np.mean(train_data, 0), np.std(train_data, 0)
    train_data_z = (train_data - train_mu_) / train_sd_
    test_data_z = (test_data - train_mu_) / train_sd_

    pca = PCA(n_components=nPC)
    pca.fit(train_data_z)
    print('explained variance ' + str(np.round(np.sum(pca.explained_variance_ratio_) * 100, 3)) + ' %')
    train_data_pc = pca.transform(train_data_z)
    test_data_pc = pca.transform(test_data_z)

    train_mu_, train_sd_ = np.mean(train_data_pc, 0), np.std(train_data_pc, 0)
    train_data_pc_z = (train_data_pc - train_mu_) / train_sd_
    test_data_pc_z = (test_data_pc - train_mu_) / train_sd_

    test_input, test_scene_index = {}, {}
    for grp in range(1, 3+1):
        data, index = [], []
        for scc in test_scene_order[grp]:
            data.append(test_data_pc_z[test_idd==scc,:])
            index.append(np.repeat(scc, len(np.where(test_idd==scc)[0])))
        data, index = np.concatenate(data, axis=0), np.concatenate(index, axis=0)
        test_input[grp], test_scene_index[grp] = data.T, index
    return train_data_pc_z.T, train_idd, test_input, test_scene_index, test_scene_order

train_input, train_scene_index, test_input, test_scene_index, test_scene_order = create_train_test_episode(directory=directory, nPC=nPC)
causal_relationship = np.array(pd.read_csv(directory+'/data/causal_relationship.csv', header=None))
memory_retrieval = np.array(pd.read_csv(directory+'/data/memory_retrieval.csv', header=None))

##########################################################
# run model
##########################################################
input_dim = test_input[1].shape[0]
hidden_dim = input_dim*2

model = gru(input_dim)

nanid = np.triu(np.zeros((48,48))+1,1)
nanid[nanid==0] = np.nan

niter = 100
iter_loss, iter_acc = np.zeros((niter, 18-2+1)), np.zeros((niter, 18-2+1))
test_iter_loss, test_iter_acc = np.zeros((niter, 3)), np.zeros((niter, 3))

for iter in range(niter):
    trainorder = np.arange(2, 18+1)[np.random.permutation(18-2+1)]
    for ep in trainorder:
        X = train_input[:, train_scene_index==ep]
        sceneid = np.repeat(ep, len(np.where(train_scene_index==ep)[0]))

        ########### gruEM ###########
        loss, log_loss, log_acc, log_h, log_yhat = model.forward(X)
        model.update_weights(loss)

        print('iter'+str(iter+1)+' ep'+str(ep)+' gru /  loss: '+str(loss.item())+', acc: '+str(conv_z2r(np.mean(conv_r2z(log_acc)))))
        iter_loss[iter, ep-2], iter_acc[iter, ep-2] = loss.detach().numpy(), np.mean(conv_r2z(log_acc))

    ##########################################################
    # test
    ##########################################################
    model_h = np.zeros((test_input[1].shape[1] - 1, hidden_dim, 3))
    h_cat = np.zeros((48, 48, 3))

    for grp in range(1, 3+1):
        X = test_input[grp]
        sceneid = test_scene_index[grp]
        scene = np.array(pd.read_csv('/Users/hayoungsong/Documents/2024COCO/socialaha/beh/groupscene.csv')['g'+str(grp)+'.sceneid'])

        ###############
        loss, log_loss, log_acc, log_h, log_yhat = model.forward_nograd(X)
        h_scc = np.zeros((log_h.shape[1], 48))
        for sci, scc in enumerate(test_scene_order[grp]):
            h_scc[:, sci] = log_h[np.where(sceneid[1:] == scc)[0], :].mean(0)
        h_corr = conv_r2z(np.corrcoef(h_scc.T))
        model_h[:,:,grp-1] = log_h
        test_iter_loss[iter,grp-1], test_iter_acc[iter,grp-1] = loss.item(), log_acc.mean()

        tmp = h_corr[np.argsort(scene), :]
        tmp = tmp[:, np.argsort(scene)]
        h_cat[:,:,grp-1] = tmp

    h_cat = np.mean(h_cat,2)*nanid

    ##########################################################
    # test-ana
    ##########################################################
    print('  h:    '+str(scipy.stats.spearmanr(causal_relationship[nanid==1], h_cat[nanid==1])))

    if os.path.exists(directory_output+'/seed'+str(seed)+'_gru')==False:
        os.mkdir(directory_output+'/seed'+str(seed)+'_gru')
    np.savez_compressed(directory_output+'/seed'+str(seed)+'_gru/summ_'+str(iter+1),
                        h_cat=h_cat)
    np.savez_compressed(directory_output+'/seed'+str(seed)+'_gru/lossacc',
                        iter_loss=iter_loss, iter_acc=iter_acc, test_iter_loss=test_iter_loss, test_iter_acc=test_iter_acc)
    torch.save({
        'i2h': model.i2h.state_dict(),
        'h2h': model.h2h.state_dict(),
        'h2o': model.h2o.state_dict()
    }, directory_output+'/seed'+str(seed)+'_gru/model_'+str(iter+1)+'.pth')
