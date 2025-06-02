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

condition = int(sys.argv[1]) # original, attnshuff, fixKQ
seed = int(sys.argv[2])
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
n_memory = test_input[1].shape[1]-1-1

EM = []
if condition=='original': model = emKeyValue(input_dim, n_memory)
elif condition=='attnshuff': model = emKeyValue(input_dim, n_memory, fixK=False, fixQ=False, attnshuff=True)
elif condition=='fixKQ': model = emKeyValue(input_dim, n_memory, fixK=True, fixQ=True, attnshuff=False)

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
        loss, EM, log_loss, log_acc, log_h, log_m, log_k, log_q, log_yhat, log_attn, log_m_sc = model.forward(X, sceneid, EM)
        model.update_weights(loss)

        print('iter'+str(iter+1)+' ep'+str(ep)+' gruEM /  loss: '+str(loss.item())+', acc: '+str(conv_z2r(np.mean(conv_r2z(log_acc)))))
        iter_loss[iter, ep-2], iter_acc[iter, ep-2] = loss.detach().numpy(), np.mean(conv_r2z(log_acc))

    ##########################################################
    # test
    ##########################################################
    model_h, model_m = np.zeros((test_input[1].shape[1]-1, hidden_dim, 3)), np.zeros((test_input[1].shape[1]-1, hidden_dim, 3))
    h_cat, m_cat = np.zeros((48,48,3)), np.zeros((48,48,3))
    model_q, model_k = np.zeros((test_input[1].shape[1]-1, hidden_dim, 3)), np.zeros((test_input[1].shape[1]-1, hidden_dim, 3))
    q_cat, k_cat = np.zeros((48,48,3)), np.zeros((48,48,3))

    retrieval_mat, retrieval_mat_eb = np.zeros((48, 48, 3)), np.zeros((48, 48, 3, 10+10+1))
    for grp in range(1, 3+1):
        X = test_input[grp]
        sceneid = test_scene_index[grp]
        scene = np.array(pd.read_csv(directory+'/data/groupscene.csv')['g'+str(grp)+'.sceneid'])

        ###############
        loss, _, log_loss, log_acc, log_h, log_m, log_k, log_q, log_yhat, log_attn, log_m_sc = model.forward_nograd(X, sceneid, [])
        log_m_sc = np.concatenate((np.array([np.nan]), np.array(log_m_sc), np.array([np.nan])))

        m_scc = np.zeros((log_m.shape[1], 48))
        h_scc = np.zeros((log_h.shape[1], 48))
        q_scc = np.zeros((log_q.shape[1], 48))
        k_scc = np.zeros((log_k.shape[1], 48))
        for sci, scc in enumerate(test_scene_order[grp]):
            m_scc[:, sci] = log_m[np.where(sceneid[1:] == scc)[0], :].mean(0)
            h_scc[:, sci] = log_h[np.where(sceneid[1:] == scc)[0], :].mean(0)
            q_scc[:, sci] = log_q[np.where(sceneid[1:] == scc)[0], :].mean(0)
            k_scc[:, sci] = log_k[np.where(sceneid[1:] == scc)[0], :].mean(0)
        h_corr = conv_r2z(np.corrcoef(h_scc.T))
        m_corr = conv_r2z(np.corrcoef(m_scc.T))
        q_corr = conv_r2z(np.corrcoef(q_scc.T))
        k_corr = conv_r2z(np.corrcoef(k_scc.T))
        model_h[:,:,grp-1] = log_h
        model_m[:,:,grp-1] = log_m
        model_q[:,:,grp-1] = log_q
        model_k[:,:,grp-1] = log_k
        test_iter_loss[iter,grp-1], test_iter_acc[iter,grp-1] = loss.item(), log_acc.mean()

        tmp = h_corr[np.argsort(scene), :]
        tmp = tmp[:, np.argsort(scene)]
        h_cat[:,:,grp-1] = tmp

        tmp = m_corr[np.argsort(scene), :]
        tmp = tmp[:, np.argsort(scene)]
        m_cat[:,:,grp-1] = tmp

        tmp = q_corr[np.argsort(scene), :]
        tmp = tmp[:, np.argsort(scene)]
        q_cat[:,:,grp-1] = tmp

        tmp = k_corr[np.argsort(scene), :]
        tmp = tmp[:, np.argsort(scene)]
        k_cat[:,:,grp-1] = tmp

        retrieval = np.zeros((48, 48))
        for t in range(sceneid.shape[0]):
            if np.isnan(sceneid[t]) or np.isnan(log_m_sc[t]):
                pass
            else:
                retrieval[int(sceneid[t] - 1), int(log_m_sc[t] - 1)] = retrieval[int(sceneid[t] - 1), int(
                    log_m_sc[t] - 1)] + 1
        for i1 in range(48 - 1):
            for i2 in range(i1 + 1, 48):
                retrieval[i1, i2] = retrieval[i1, i2] + retrieval[i2, i1]
                retrieval[i2, i1] = 0
        retrieval_mat[:, :, grp - 1] = retrieval

        for separation in range(-10, 10+1):
            retrieval = np.zeros((48, 48))
            evonset = np.zeros((len(sceneid),))
            for t in range(len(sceneid) - 1):
                if sceneid[t] != sceneid[t + 1]:
                    evonset[t + 1 + separation: t + 1 + separation + 1] = 1
            for t in range(len(sceneid) - 1):
                if evonset[t] == 1:
                    if np.isnan(sceneid[t]) or np.isnan(log_m_sc[t]):
                        pass
                    else:
                        retrieval[int(sceneid[t]-1), int(log_m_sc[t]-1)] = retrieval[int(sceneid[t]-1), int(log_m_sc[t]-1)]+1
            for i1 in range(48-1):
                for i2 in range(i1+1, 48):
                    retrieval[i1,i2] = retrieval[i1,i2]+retrieval[i2,i1]
                    retrieval[i2,i1] = 0
            retrieval_mat_eb[:, :, grp - 1, separation+10] = retrieval

    h_cat, m_cat = np.mean(h_cat,2)*nanid, np.mean(m_cat,2)*nanid
    q_cat, k_cat = np.mean(q_cat,2)*nanid, np.mean(k_cat,2)*nanid
    retrieval_mat = np.nanmean(retrieval_mat, 2) * nanid
    retrieval_mat_eb = np.nanmean(retrieval_mat_eb, 2)

    ##########################################################
    # test-ana
    ##########################################################
    print('  h:    '+str(scipy.stats.spearmanr(causal_relationship[nanid==1], h_cat[nanid==1])))
    print('  m:    '+str(scipy.stats.spearmanr(causal_relationship[nanid==1], m_cat[nanid==1])))
    print('  retr: '+str(scipy.stats.spearmanr(retrieval_mat[np.where(~np.isnan(memory_retrieval))], memory_retrieval[np.where(~np.isnan(memory_retrieval))])))

    if os.path.exists(directory_output+'/seed'+str(seed)+'_'+condition)==False:
        os.mkdir(directory_output+'/seed'+str(seed)+'_'+condition)
    np.savez_compressed(directory_output+'/seed'+str(seed)+'_'+condition+'/summ_'+str(iter+1),
                        h_cat=h_cat, m_cat=m_cat, q_cat=q_cat, k_cat=k_cat,
                        retrieval=retrieval_mat, retrieval_eb=retrieval_mat_eb)
    np.savez_compressed(directory_output+'/seed'+str(seed)+'_'+condition+'/lossacc',
                        iter_loss=iter_loss, iter_acc=iter_acc, test_iter_loss=test_iter_loss, test_iter_acc=test_iter_acc)
    torch.save({
        'i2h': model.i2h.state_dict(),
        'h2h': model.h2h.state_dict(),
        'hm2o': model.hm2o.state_dict(),
        'W_k': model.W_k,
        'W_q': model.W_q
    }, directory_output+'/seed'+str(seed)+'_'+condition+'/model_'+str(iter+1)+'.pth')
