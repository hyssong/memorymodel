import numpy as np
import scipy
import os
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl
import pingouin as pg
import pickle
from statsmodels.stats.multitest import fdrcorrection
mpl.rcParams['font.family'] = 'Helvetica'
np.random.seed(0)

def conv_r2z(r):
    with np.errstate(invalid='ignore', divide='ignore'):
        return 0.5 * (np.log(1 + r) - np.log(1 - r))
def conv_z2r(z):
    with np.errstate(invalid='ignore', divide='ignore'):
        return (np.exp(2 * z) - 1) / (np.exp(2 * z) + 1)

def partial_correlation(X, Y, A):
    df = pd.DataFrame({'X': X, 'Y': Y, 'A': A})
    pcorr = pg.partial_corr(data=df, x='X', y='Y', covar='A', method='spearman')
    return np.array(pcorr['r']).item(), np.array(pcorr['p-val']).item()

##########################################
dir_base = os.getcwd()

nanid = np.triu(np.zeros((48,48))+1,1)
nanid[nanid==0] = np.nan
causal_relationship = np.array(pd.read_csv(dir_base+'/data/causal_relationship.csv', header=None))
memory_retrieval = np.array(pd.read_csv(dir_base+'/data/memory_retrieval.csv', header=None))

with open(dir_base+'/input/input.pkl', 'rb') as f: data_input = pickle.load(f)
nPC=50
smm_ = np.zeros((nPC, 48))
for scc in range(1, 48+1):
    smm_[:,scc-1] = data_input['test_input'][1][:,np.where(data_input['test_scene_index'][1]==scc)[0]].mean(1)
input_similarity = conv_r2z(np.corrcoef(smm_.T))
input_similarity[np.where(np.isnan(nanid))]=np.nan

res = scipy.stats.spearmanr(causal_relationship[nanid==1], input_similarity[nanid==1])
print(f"Spearman correlation (Causal vs Input):     rho = {res.correlation:.3f}, p = {res.pvalue:.3e}")
res = scipy.stats.spearmanr(causal_relationship[np.where(~np.isnan(memory_retrieval))], memory_retrieval[np.where(~np.isnan(memory_retrieval))])
print(f"Spearman correlation (Causal vs Memory):    rho = {res.correlation:.3f}, p = {res.pvalue:.3e}")
res = scipy.stats.spearmanr(memory_retrieval[np.where(~np.isnan(memory_retrieval))], input_similarity[np.where(~np.isnan(memory_retrieval))])
print(f"Spearman correlation (Memory vs Input):     rho = {res.correlation:.3f}, p = {res.pvalue:.3e}")


##########################################
param_alpha, param_tau, niter = 0.5, 0.1, 50
directory = dir_base+'/output/alpha'+str(param_alpha)+'_tau'+str(param_tau)

conditions = ['directMem', 'copypaste', 'attnshuff', 'gru', 'original']
colors = ['#1f77b4','#BE5958', '#DB9123','#2BCF3B','#000000']
nseed = 20
# directMem: no key
# copypaste: no key-query
# attnshuff: shuffled memory
# gru: no EM
# original: EM-GRU

partial_h_cdt, partial_k_cdt, partial_q_cdt = {}, {}, {}
causal_h_cdt, causal_k_cdt, causal_q_cdt = {}, {}, {}
kq_memory_cdt, partial_kq_memory_cdt = {}, {}
for cdi, cdt in enumerate(conditions): # cdi, cdt = 0, 'original'
    print(cdt+' seed'+str(nseed))
    partial_h, partial_k, partial_q = np.zeros((nseed, niter))+np.nan, np.zeros((nseed, niter))+np.nan, np.zeros((nseed, niter))+np.nan
    causal_h, causal_k, causal_q = np.zeros((nseed, niter))+np.nan, np.zeros((nseed, niter))+np.nan, np.zeros((nseed, niter))+np.nan
    kq_memory, partial_kq_memory = np.zeros((nseed, niter))+np.nan, np.zeros((nseed, niter))+np.nan

    for sd in range(1, nseed+1):
        foldername = 'seed'+str(sd)+'_'+cdt
        for iter in range(1, niter+1):
            data = np.load(directory + '/' + foldername + '/summ_' + str(iter) + '.npz')
            h_cat       = data['h_cat']       if 'h_cat'       in data else np.full((48, 48), np.nan)
            k_cat       = data['k_cat']       if 'k_cat'       in data else np.full((48, 48), np.nan)
            q_cat       = data['q_cat']       if 'q_cat'       in data else np.full((48, 48), np.nan)
            retrieval   = data['retrieval']   if 'retrieval'   in data else np.full((48, 48), np.nan)
            # gru: no k, q, ret
            # copypaste: no k, q
            # directMem: no k

            if not np.all(np.isnan(h_cat)): partial_h[sd-1, iter-1] = conv_r2z(partial_correlation(h_cat[nanid==1], causal_relationship[nanid==1], input_similarity[nanid==1])[0])
            if not np.all(np.isnan(k_cat)): partial_k[sd-1, iter-1] = conv_r2z(partial_correlation(k_cat[nanid==1], causal_relationship[nanid==1], input_similarity[nanid==1])[0])
            if not np.all(np.isnan(q_cat)): partial_q[sd-1, iter-1] = conv_r2z(partial_correlation(q_cat[nanid==1], causal_relationship[nanid==1], input_similarity[nanid==1])[0])

            if not np.all(np.isnan(h_cat)): causal_h[sd-1, iter-1] = conv_r2z(scipy.stats.spearmanr(h_cat[nanid==1], causal_relationship[nanid==1])[0])
            if not np.all(np.isnan(k_cat)): causal_k[sd-1, iter-1] = conv_r2z(scipy.stats.spearmanr(k_cat[nanid==1], causal_relationship[nanid==1])[0])
            if not np.all(np.isnan(q_cat)): causal_q[sd-1, iter-1] = conv_r2z(scipy.stats.spearmanr(q_cat[nanid==1], causal_relationship[nanid==1])[0])

            data = np.load(directory + '/' + foldername + '/param_' + str(iter) + '.npz')
            model_h       = data['model_h']       if 'model_h'       in data else np.full((598, 100, 3), np.nan)
            model_k       = data['model_k']       if 'model_k'       in data else np.full((598, 100, 3), np.nan)
            model_q       = data['model_q']       if 'model_q'       in data else np.full((598, 100, 3), np.nan)

            h_scc, k_scc, q_scc = np.zeros((100,48,3))+np.nan, np.zeros((100,48,3))+np.nan, np.zeros((100,48,3))+np.nan
            if not np.all(np.isnan(model_h)):
                for grp in range(1, 3+1):
                    for scc in range(1, 48+1):
                        h_scc[:, scc-1, grp-1] = model_h[np.where(data_input['test_scene_index'][grp][1:] == scc)[0], :, grp-1].mean(0)
            if not np.all(np.isnan(model_k)):
                for grp in range(1, 3+1):
                    for scc in range(1, 48+1):
                        k_scc[:, scc-1, grp-1] = model_k[np.where(data_input['test_scene_index'][grp][1:] == scc)[0], :, grp-1].mean(0)
            if not np.all(np.isnan(model_q)):
                for grp in range(1, 3+1):
                    for scc in range(1, 48+1):
                        q_scc[:, scc-1, grp-1] = model_q[np.where(data_input['test_scene_index'][grp][1:] == scc)[0], :, grp-1].mean(0)

            kq_mat = np.zeros((48, 48, 3))
            for grp in range(3):
                if cdt=='copypaste': # no k, q
                    kq = scipy.stats.spearmanr(h_scc[:,:,grp], h_scc[:,:,grp])[0][:48,48:]
                elif cdt=='directMem': # no k
                    kq = scipy.stats.spearmanr(h_scc[:,:,grp], q_scc[:,:,grp])[0][:48,48:]
                else:
                    kq = scipy.stats.spearmanr(k_scc[:,:,grp], q_scc[:,:,grp])[0][:48,48:]
                kq[np.where(np.triu(np.zeros((48,48))+1, 1)==0)] = np.nan
                kq_mat[:,:,grp] = kq
            kq_mat = np.mean(conv_r2z(kq_mat),2)

            if not np.all(np.isnan(kq_mat)):
                kq_memory[sd-1, iter-1] = conv_r2z(scipy.stats.spearmanr(kq_mat[np.where(~np.isnan(memory_retrieval))], memory_retrieval[np.where(~np.isnan(memory_retrieval))])[0])
                partial_kq_memory[sd-1, iter-1] = conv_r2z(partial_correlation(kq_mat[np.where(~np.isnan(memory_retrieval))], memory_retrieval[np.where(~np.isnan(memory_retrieval))], input_similarity[np.where(~np.isnan(memory_retrieval))])[0])

    partial_h_cdt[cdt], partial_k_cdt[cdt], partial_q_cdt[cdt] = partial_h, partial_k, partial_q
    causal_h_cdt[cdt], causal_k_cdt[cdt], causal_q_cdt[cdt] = causal_h, causal_k, causal_q
    kq_memory_cdt[cdt], partial_kq_memory_cdt[cdt] = kq_memory, partial_kq_memory

cdt = 'original'
hk, kq, qh = np.zeros((nseed, niter, 3)), np.zeros((nseed, niter, 3)), np.zeros((nseed, niter, 3))
for sd in range(1, nseed + 1):
    foldername = 'seed' + str(sd) + '_' + cdt
    for iter in range(1, niter + 1):
        data = np.load(directory + '/' + foldername + '/param_' + str(iter) + '.npz')
        for grp in range(3):
            hk[sd-1, iter-1, grp] = conv_r2z(scipy.stats.spearmanr(data['model_h'][:,:,grp].flatten(), data['model_k'][:,:,grp].flatten())[0])
            kq[sd-1, iter-1, grp] = conv_r2z(scipy.stats.spearmanr(data['model_k'][:,:,grp].flatten(), data['model_q'][:,:,grp].flatten())[0])
            qh[sd-1, iter-1, grp] = conv_r2z(scipy.stats.spearmanr(data['model_q'][:,:,grp].flatten(), data['model_h'][:,:,grp].flatten())[0])
hk, kq, qh = np.mean(hk, 2), np.mean(kq, 2), np.mean(qh, 2)


#############################################
fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(3, 3, figsize=(9.85, 8))
for cdi, cdt in enumerate(conditions):
    partial_h, partial_k, partial_q = partial_h_cdt[cdt], partial_k_cdt[cdt], partial_q_cdt[cdt]
    causal_h, causal_k, causal_q = causal_h_cdt[cdt], causal_k_cdt[cdt], causal_q_cdt[cdt]
    kq_memory, partial_kq_memory = kq_memory_cdt[cdt], partial_kq_memory_cdt[cdt]

    line1, = ax1.plot(np.arange(niter)+1, conv_z2r(np.mean(causal_h,0)), color=colors[cdi], linewidth=1)
    line2, = ax2.plot(np.arange(niter)+1, conv_z2r(np.mean(causal_k,0)), color=colors[cdi], linewidth=1)
    line3, = ax3.plot(np.arange(niter)+1, conv_z2r(np.mean(causal_q,0)), color=colors[cdi], linewidth=1)
    line4, = ax4.plot(np.arange(niter) + 1, conv_z2r(np.mean(partial_h, 0)), color=colors[cdi], linewidth=1)
    line5, = ax5.plot(np.arange(niter) + 1, conv_z2r(np.mean(partial_k, 0)), color=colors[cdi], linewidth=1)
    line6, = ax6.plot(np.arange(niter) + 1, conv_z2r(np.mean(partial_q, 0)), color=colors[cdi], linewidth=1)

    ax1.fill_between(np.arange(niter)+1, conv_z2r(np.mean(causal_h,0)-np.std(causal_h,0)/np.sqrt(nseed)),
                     conv_z2r(np.mean(causal_h,0)+np.std(causal_h,0)/np.sqrt(nseed)) , color=colors[cdi], alpha=0.3, edgecolor='none')
    ax2.fill_between(np.arange(niter)+1, conv_z2r(np.mean(causal_k,0)-np.std(causal_k,0)/np.sqrt(nseed)),
                     conv_z2r(np.mean(causal_k,0)+np.std(causal_k,0)/np.sqrt(nseed)) , color=colors[cdi], alpha=0.3, edgecolor='none')
    ax3.fill_between(np.arange(niter)+1, conv_z2r(np.mean(causal_q,0)-np.std(causal_q,0)/np.sqrt(nseed)),
                     conv_z2r(np.mean(causal_q,0)+np.std(causal_q,0)/np.sqrt(nseed)) , color=colors[cdi], alpha=0.3, edgecolor='none')
    ax4.fill_between(np.arange(niter)+1, conv_z2r(np.mean(partial_h,0)-np.std(partial_h,0)/np.sqrt(nseed)),
                     conv_z2r(np.mean(partial_h,0)+np.std(partial_h,0)/np.sqrt(nseed)) , color=colors[cdi], alpha=0.3, edgecolor='none')
    ax5.fill_between(np.arange(niter)+1, conv_z2r(np.mean(partial_k,0)-np.std(partial_k,0)/np.sqrt(nseed)),
                     conv_z2r(np.mean(partial_k,0)+np.std(partial_k,0)/np.sqrt(nseed)) , color=colors[cdi], alpha=0.3, edgecolor='none')
    ax6.fill_between(np.arange(niter)+1, conv_z2r(np.mean(partial_q,0)-np.std(partial_q,0)/np.sqrt(nseed)),
                     conv_z2r(np.mean(partial_q,0)+np.std(partial_q,0)/np.sqrt(nseed)) , color=colors[cdi], alpha=0.3, edgecolor='none')

    line7, = ax7.plot(np.arange(niter)+1, conv_z2r(np.mean(kq_memory,0)), color=colors[cdi], linewidth=1)
    ax7.fill_between(np.arange(niter)+1, conv_z2r(np.mean(kq_memory,0)-np.std(kq_memory,0)/np.sqrt(nseed)),
                     conv_z2r(np.mean(kq_memory,0)+np.std(kq_memory,0)/np.sqrt(nseed)) , color=colors[cdi], alpha=0.3, edgecolor='none')
    line8, = ax8.plot(np.arange(niter)+1, conv_z2r(np.mean(partial_kq_memory,0)), color=colors[cdi], linewidth=1)
    ax8.fill_between(np.arange(niter)+1, conv_z2r(np.mean(partial_kq_memory,0)-np.std(partial_kq_memory,0)/np.sqrt(nseed)),
                     conv_z2r(np.mean(partial_kq_memory,0)+np.std(partial_kq_memory,0)/np.sqrt(nseed)) , color=colors[cdi], alpha=0.3, edgecolor='none')

ax9.plot(np.arange(1, niter+1), conv_z2r(np.mean(hk, 0)), color='#326C43')
ax9.plot(np.arange(1, niter+1), conv_z2r(np.mean(kq, 0)), color='#854E0C')
ax9.plot(np.arange(1, niter+1), conv_z2r(np.mean(qh, 0)), color='#946DA1')
ax9.fill_between(np.arange(1, niter+1), conv_z2r(np.mean(hk, 0) - np.std(hk, 0) / np.sqrt(nseed)),
                conv_z2r(np.mean(hk, 0) + np.std(hk, 0) / np.sqrt(nseed)), color='#326C43', alpha=0.3, edgecolor='none')
ax9.fill_between(np.arange(1, niter+1), conv_z2r(np.mean(kq, 0) - np.std(kq, 0) / np.sqrt(nseed)),
                conv_z2r(np.mean(kq, 0) + np.std(kq, 0) / np.sqrt(nseed)), color='#854E0C', alpha=0.3, edgecolor='none')
ax9.fill_between(np.arange(1, niter+1), conv_z2r(np.mean(qh, 0) - np.std(qh, 0) / np.sqrt(nseed)),
                conv_z2r(np.mean(qh, 0) + np.std(qh, 0) / np.sqrt(nseed)), color='#946DA1', alpha=0.3, edgecolor='none')

for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9]:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    ax.tick_params(width=0.8)
plt.tight_layout()
