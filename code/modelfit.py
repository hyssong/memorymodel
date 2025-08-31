import numpy as np
import scipy
import pandas as pd
import os
import torch
import sys
import pickle

directory = '/directory' # ****** set current directory ******
sys.path.append(directory+'/model')
from emKeyValue import emKeyValue

def conv_r2z(r):
    with np.errstate(invalid='ignore', divide='ignore'):
        return 0.5 * (np.log(1 + r) - np.log(1 - r))
def conv_z2r(z):
    with np.errstate(invalid='ignore', divide='ignore'):
        return (np.exp(2 * z) - 1) / (np.exp(2 * z) + 1)

###########################################################
# conditions and hyperparameters
###########################################################
# condition='original'
# seed=1
# param_alpha=0.5
# param_tau=0.1
condition = sys.argv[1]          # 'original', 'attnshuff'
seed = int(sys.argv[2])          # 1-20
param_alpha = float(sys.argv[3]) # base alpha = 0.5, range [0, 0.25, 0.5, 0.75, 1]
param_tau = float(sys.argv[4])   # base tau   = 0.1, range [0.1, 0.5, 1]

niter = 50 # training iterations

# setting seed
torch.manual_seed(seed), np.random.seed(seed)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

# setting output directory
directory_output = directory+'/output/alpha'+str(param_alpha)+'_tau'+str(param_tau)
if os.path.exists(directory_output)==False:
    os.mkdir(directory_output)
if os.path.exists(directory_output+'/seed'+str(seed)+'_'+condition)==False:
    os.mkdir(directory_output+'/seed'+str(seed)+'_'+condition)

##########################################################
# load input
##########################################################
with open(directory+'/input/input.pkl', 'rb') as f:
    data = pickle.load(f)
train_input, train_scene_index, test_input, test_scene_index, test_scene_order = data['train_input'], data['train_scene_index'], data['test_input'], data['test_scene_index'], data['test_scene_order']

causal_relationship = np.array(pd.read_csv(directory+'/data/causal_relationship.csv', header=None))
memory_retrieval = np.array(pd.read_csv(directory+'/data/memory_retrieval.csv', header=None))

nanid = np.triu(np.zeros((48,48))+1,1)
nanid[nanid==0] = np.nan

##########################################################
# run model
##########################################################
input_dim = test_input[1].shape[0]
hidden_dim = input_dim*2
n_memory = test_input[1].shape[1]-1-1

EM = []
if condition=='original': model = emKeyValue(input_dim, n_memory, alpha=param_alpha, tau=param_tau)
elif condition=='attnshuff': model = emKeyValue(input_dim, n_memory, alpha=param_alpha, tau=param_tau, attnshuff=True)

iter_loss, iter_acc = np.zeros((niter, 18-2+1)), np.zeros((niter, 18-2+1))
test_iter_loss, test_iter_acc = np.zeros((niter, 3)), np.zeros((niter, 3))
for iter in range(niter):
    ##########################################################
    # train on episodes 2-18, save the model parameters
    ##########################################################
    trainorder = np.arange(2, 18+1)[np.random.permutation(18-2+1)]
    for ep in trainorder:
        X = train_input[:, train_scene_index==ep]
        sceneid = np.repeat(ep, len(np.where(train_scene_index==ep)[0]))

        loss, EM, log_loss, log_acc, log_h, log_m, log_k, log_q, log_yhat, log_attn, log_m_sc = model.forward(X, sceneid, EM)
        model.update_weights(loss)
        print('iter'+str(iter+1)+' ep'+str(ep)+' gruEM /  loss: '+str(loss.item())+', acc: '+str(conv_z2r(np.mean(conv_r2z(log_acc)))))
        iter_loss[iter, ep-2], iter_acc[iter, ep-2] = loss.detach().numpy(), np.mean(conv_r2z(log_acc))

    torch.save({
        'i2h': model.i2h.state_dict(),
        'h2h': model.h2h.state_dict(),
        'hm2o': model.hm2o.state_dict(),
        'W_k': model.W_k,
        'W_q': model.W_q
    }, directory_output+'/seed'+str(seed)+'_'+condition+'/model_'+str(iter+1)+'.pth')

    ##########################################################
    # test on episode 1, three scrambled-order groups
    ##########################################################
    model_h, model_m, model_q, model_k = np.zeros((test_input[1].shape[1]-1, hidden_dim, 3)), np.zeros((test_input[1].shape[1]-1, hidden_dim, 3)), np.zeros((test_input[1].shape[1]-1, hidden_dim, 3)), np.zeros((test_input[1].shape[1]-1, hidden_dim, 3))
    model_attn = np.zeros((test_input[1].shape[1]-1, test_input[1].shape[1]-1, 3))+np.nan
    model_m_sc = np.zeros((test_input[1].shape[1], 3))

    h_cat, m_cat, q_cat, k_cat, retrieval_mat = np.zeros((48,48,3)), np.zeros((48,48,3)), np.zeros((48,48,3)), np.zeros((48,48,3)), np.zeros((48,48,3))

    for grp in range(1, 3+1):
        X = test_input[grp]
        sceneid = test_scene_index[grp]
        scene = np.array(pd.read_csv(directory+'/data/groupscene.csv')['g'+str(grp)+'.sceneid'])

        ###############
        torch.manual_seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
        loss, _, log_loss, log_acc, log_h, log_m, log_k, log_q, log_yhat, log_attn, log_m_sc = model.forward_nograd(X, sceneid, [])
        log_m_sc = np.concatenate((np.array([np.nan]), np.array(log_m_sc), np.array([np.nan])))

        for i in range(len(log_attn)):
            model_attn[i+1,:len(log_attn[i]),grp-1] = log_attn[i]
        model_m_sc[:,grp-1] =log_m_sc

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
                retrieval[int(sceneid[t] - 1), int(log_m_sc[t] - 1)] = retrieval[int(sceneid[t] - 1), int(log_m_sc[t] - 1)] + 1
        for i1 in range(48 - 1):
            for i2 in range(i1 + 1, 48):
                retrieval[i1, i2] = retrieval[i1, i2] + retrieval[i2, i1]
                retrieval[i2, i1] = 0
        retrieval_mat[:, :, grp - 1] = retrieval

    h_cat, m_cat = np.mean(h_cat,2)*nanid, np.mean(m_cat,2)*nanid
    q_cat, k_cat = np.mean(q_cat,2)*nanid, np.mean(k_cat,2)*nanid
    retrieval_mat = np.nanmean(retrieval_mat, 2) * nanid

    ##########################################################
    # test-ana
    ##########################################################
    print('  h:    '+str(scipy.stats.spearmanr(causal_relationship[nanid==1], h_cat[nanid==1])))
    print('  retr: '+str(scipy.stats.spearmanr(retrieval_mat[np.where(~np.isnan(memory_retrieval))], memory_retrieval[np.where(~np.isnan(memory_retrieval))])))

    np.savez_compressed(directory_output+'/seed'+str(seed)+'_'+condition+'/summ_'+str(iter+1),
                        h_cat=h_cat, m_cat=m_cat, q_cat=q_cat, k_cat=k_cat, retrieval=retrieval_mat)
    np.savez_compressed(directory_output+'/seed'+str(seed)+'_'+condition+'/param_'+str(iter+1),
                        model_h=model_h, model_m=model_m, model_q=model_q, model_k=model_k, model_attn=model_attn, model_m_sc=model_m_sc)
    np.savez_compressed(directory_output+'/seed'+str(seed)+'_'+condition+'/lossacc',
                        iter_loss=iter_loss, iter_acc=iter_acc, test_iter_loss=test_iter_loss, test_iter_acc=test_iter_acc)

