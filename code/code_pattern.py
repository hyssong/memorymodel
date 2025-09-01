import numpy as np
import scipy
import matplotlib.pyplot as plt
import pickle

def conv_r2z(r):
    with np.errstate(invalid='ignore', divide='ignore'):
        return 0.5 * (np.log(1 + r) - np.log(1 - r))
def conv_z2r(z):
    with np.errstate(invalid='ignore', divide='ignore'):
        return (np.exp(2 * z) - 1) / (np.exp(2 * z) + 1)
def spearman_corrcoef(a):
    return scipy.stats.spearmanr(a, axis=1, nan_policy='omit').correlation

#####################
dir_base = '/directory' # ****** set current directory ******

with open(dir_base+'/input/input.pkl', 'rb') as f: data_input = pickle.load(f)
value = []
for scc in range(1, 48+1):
    within = data_input['test_input'][1][:,np.where(data_input['test_scene_index'][1]==scc)[0]]
    corrv = conv_r2z(spearman_corrcoef(within.T))
    if scc==1: value = corrv[np.where(np.triu(np.zeros(corrv.shape)+1,1)==1)]
    else: value = np.concatenate((value, corrv[np.where(np.triu(np.zeros(corrv.shape)+1,1)==1)]))
print('input: mean '+str(conv_z2r(value.mean()))+' (sd '+str(conv_z2r(value.std()))+')')

#####################
param_alpha, param_tau, niter = 0.5, 0.1, 50
directory = dir_base+'/output/alpha'+str(param_alpha)+'_tau'+str(param_tau)

conditions = ['directMem', 'copypaste', 'attnshuff', 'gru', 'original']
colors = ['#1f77b4','#BE5958', '#DB9123','#2BCF3B','#000000']
nseed = 20

fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(11, 3))
for cdi, cdt in enumerate(conditions):

    print(cdt+' seed'+str(nseed))
    pattern_h, pattern_k, pattern_q = np.zeros((nseed, niter)), np.zeros((nseed, niter)), np.zeros((nseed, niter))
    for sd in range(1, nseed+1):
        foldername = 'seed'+str(sd)+'_'+cdt
        for iter in range(1, niter+1):
            data = np.load(directory + '/' + foldername + '/param_' + str(iter) + '.npz')
            model_h = data.get('model_h', None)
            model_k = data.get('model_k', None)
            model_q = data.get('model_q', None)

            value_h_, value_k_, value_q_ = [], [], []
            for grp in range(3):
                index = data_input['test_scene_index'][grp+1][1:]
                value_h, value_k, value_q = [], [], []
                for scc in range(1, 48 + 1):
                    if model_h is not None:
                        corrv = conv_r2z(spearman_corrcoef(model_h[np.where(index==scc)[0],:,grp]))
                        if scc==1: value_h = corrv[np.where(np.triu(np.zeros(corrv.shape)+1, 1)==1)]
                        else: value_h = np.concatenate((value_h, corrv[np.where(np.triu(np.zeros(corrv.shape)+1, 1)==1)]))
                    if model_k is not None:
                        corrv = conv_r2z(spearman_corrcoef(model_k[np.where(index==scc)[0],:,grp]))
                        if scc==1: value_k = corrv[np.where(np.triu(np.zeros(corrv.shape)+1, 1)==1)]
                        else: value_k = np.concatenate((value_k, corrv[np.where(np.triu(np.zeros(corrv.shape)+1, 1)==1)]))
                    if model_q is not None:
                        corrv = conv_r2z(spearman_corrcoef(model_q[np.where(index==scc)[0],:,grp]))
                        if scc==1: value_q = corrv[np.where(np.triu(np.zeros(corrv.shape)+1, 1)==1)]
                        else: value_q = np.concatenate((value_q, corrv[np.where(np.triu(np.zeros(corrv.shape)+1, 1)==1)]))
                if len(value_h)>0: value_h_.append(value_h.mean())
                if len(value_k)>0: value_k_.append(value_k.mean())
                if len(value_q)>0: value_q_.append(value_q.mean())
            value_h_, value_k_, value_q_ = np.array(value_h_).mean(), np.array(value_k_).mean(), np.array(value_q_).mean()
            pattern_h[sd-1,iter-1], pattern_k[sd-1,iter-1], pattern_q[sd-1,iter-1] = value_h_, value_k_, value_q_

    line1, = ax1.plot(np.arange(niter)+1, conv_z2r(np.mean(pattern_h,0)), color=colors[cdi], linewidth=1)
    line2, = ax2.plot(np.arange(niter)+1, conv_z2r(np.mean(pattern_k,0)), color=colors[cdi], linewidth=1)
    line3, = ax3.plot(np.arange(niter)+1, conv_z2r(np.mean(pattern_q,0)), color=colors[cdi], linewidth=1)

    ax1.fill_between(np.arange(niter)+1, conv_z2r(np.mean(pattern_h,0)-np.std(pattern_h,0)/np.sqrt(nseed)),
                     conv_z2r(np.mean(pattern_h,0)+np.std(pattern_h,0)/np.sqrt(nseed)) , color=colors[cdi], alpha=0.3, edgecolor='none')
    ax2.fill_between(np.arange(niter)+1, conv_z2r(np.mean(pattern_k,0)-np.std(pattern_k,0)/np.sqrt(nseed)),
                     conv_z2r(np.mean(pattern_k,0)+np.std(pattern_k,0)/np.sqrt(nseed)) , color=colors[cdi], alpha=0.3, edgecolor='none')
    ax3.fill_between(np.arange(niter)+1, conv_z2r(np.mean(pattern_q,0)-np.std(pattern_q,0)/np.sqrt(nseed)),
                     conv_z2r(np.mean(pattern_q,0)+np.std(pattern_q,0)/np.sqrt(nseed)) , color=colors[cdi], alpha=0.3, edgecolor='none')

for ax in [ax1, ax2, ax3]:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim([-0.5, 50.5])
plt.tight_layout()

