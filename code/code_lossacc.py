import numpy as np
import scipy
import os
import matplotlib.pyplot as plt
import glob
import matplotlib as mpl
mpl.rcParams['font.family'] = 'Helvetica'

def conv_r2z(r):
    with np.errstate(invalid='ignore', divide='ignore'):
        return 0.5 * (np.log(1 + r) - np.log(1 - r))
def conv_z2r(z):
    with np.errstate(invalid='ignore', divide='ignore'):
        return (np.exp(2 * z) - 1) / (np.exp(2 * z) + 1)

##########################################################
# result load
##########################################################
conditions = ['directMem', 'copypaste', 'attnshuff', 'gru', 'original']
colorr = ['#1f77b4','#BE5958', '#DB9123','#2BCF3B','#000000']
# directMem: no key
# copypaste: no key-query
# attnshuff: shuffled memory
# gru: no EM
# original: EM-GRU

nPC = 50
donne = 50
nseed = 20
errorbar = 'confidence_interval' # 'standard_error'
param_alpha, param_tau = 0.5, 0.1

dir_base = os.path.dirname(os.getcwd())
directory = dir_base+'/output/alpha'+str(param_alpha)+'_tau'+str(param_tau)


train_loss, train_acc, test_loss, test_acc = {}, {}, {}, {}

# train result
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(9,3.5))
for cdi, cdt in enumerate(conditions):
    iter_loss_cat, iter_acc_cat = [], []
    print(cdt + ' seed' + str(nseed))
    for sd in range(1, nseed+1):
        foldername = 'seed'+str(sd)+'_'+cdt
        data = np.load(directory + '/' + foldername + '/lossacc.npz')
        iter_loss, iter_acc = data['iter_loss'], data['iter_acc']

        iter_loss_cat.append(np.mean(iter_loss,1)[:donne])
        iter_acc_cat.append(np.mean(iter_acc,1)[:donne])
    train_loss[cdt], train_acc[cdt] = np.array(iter_loss_cat), np.array(iter_acc_cat)
    ax1.plot(np.arange(donne)+1, np.mean(np.array(iter_loss_cat),0), color=colorr[cdi], linewidth=1)
    if errorbar == 'confidence_interval':
        ax1.fill_between(np.arange(donne)+1, np.mean(np.array(iter_loss_cat),0)-1.96*np.std(np.array(iter_loss_cat),0),
                         np.mean(np.array(iter_loss_cat),0)+1.96*np.std(np.array(iter_loss_cat),0), color=colorr[cdi], alpha=0.4,
                         edgecolor='none')
    elif errorbar == 'standard_error':
        ax1.fill_between(np.arange(donne)+1, np.mean(np.array(iter_loss_cat),0)-np.std(np.array(iter_loss_cat),0)/np.sqrt(nseed),
                         np.mean(np.array(iter_loss_cat),0)+np.std(np.array(iter_loss_cat),0)/np.sqrt(nseed), color=colorr[cdi], alpha=0.4,
                         edgecolor='none')
    ax2.plot(np.arange(donne)+1, conv_z2r(np.mean(np.array(iter_acc_cat),0)), color=colorr[cdi], linewidth=1)
    if errorbar == 'confidence_interval':
        ax2.fill_between(np.arange(donne)+1, conv_z2r(np.mean(np.array(iter_acc_cat),0)-1.96*np.std(np.array(iter_acc_cat),0)),
                         conv_z2r(np.mean(np.array(iter_acc_cat),0)+1.96*np.std(np.array(iter_acc_cat),0)), color=colorr[cdi], alpha=0.4,
                         edgecolor='none')
    elif errorbar == 'standard_error':
        ax2.fill_between(np.arange(donne)+1, conv_z2r(np.mean(np.array(iter_acc_cat),0)-np.std(np.array(iter_acc_cat),0)/np.sqrt(nseed)),
                         conv_z2r(np.mean(np.array(iter_acc_cat),0)+np.std(np.array(iter_acc_cat),0)/np.sqrt(nseed)), color=colorr[cdi], alpha=0.4,
                         edgecolor='none')
for ax in [ax1, ax2]:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel('training iterations', fontsize=14)
ax1.set_ylabel('train loss', fontsize=14)
ax2.set_ylabel('train accuracy', fontsize=14)

# test result
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(9,3.5))
minimum = []
for cdi, cdt in enumerate(conditions):
    test_iter_loss_cat, test_iter_acc_cat = [], []
    print(cdt + ' seed' + str(nseed))
    for sd in range(1, nseed+1):
        foldername = 'seed'+str(sd)+'_'+cdt
        data = np.load(directory + '/' + foldername + '/lossacc.npz')
        test_iter_loss, test_iter_acc = data['test_iter_loss'], data['test_iter_acc']

        test_iter_loss_cat.append(np.mean(test_iter_loss,1)[:donne])
        test_iter_acc_cat.append(np.mean(test_iter_acc,1)[:donne])
        minimum.append(np.where(np.mean(test_iter_loss,1)[:donne]==np.min(np.mean(test_iter_loss,1)[:donne]))[0][0]+1)

    test_loss[cdt], test_acc[cdt] = np.array(test_iter_loss_cat), np.array(test_iter_acc_cat)
    ax1.plot(np.arange(donne)+1, np.mean(np.array(test_iter_loss_cat),0), color=colorr[cdi], linewidth=1)
    if errorbar == 'confidence_interval':
        ax1.fill_between(np.arange(donne)+1, np.mean(np.array(test_iter_loss_cat),0)-1.96*np.std(np.array(test_iter_loss_cat),0),
                         np.mean(np.array(test_iter_loss_cat),0)+1.96*np.std(np.array(test_iter_loss_cat),0), color=colorr[cdi], alpha=0.4,
                         edgecolor='none')
    elif errorbar == 'standard_error':
        ax1.fill_between(np.arange(donne)+1, np.mean(np.array(test_iter_loss_cat),0)-np.std(np.array(test_iter_loss_cat),0)/np.sqrt(nseed),
                         np.mean(np.array(test_iter_loss_cat),0)+np.std(np.array(test_iter_loss_cat),0)/np.sqrt(nseed), color=colorr[cdi], alpha=0.4,
                         edgecolor='none')
    ax2.plot(np.arange(donne)+1, conv_z2r(np.mean(np.array(test_iter_acc_cat),0)), color=colorr[cdi], linewidth=1)
    if errorbar == 'confidence_interval':
        ax2.fill_between(np.arange(donne)+1, conv_z2r(np.mean(np.array(test_iter_acc_cat),0)-1.96*np.std(np.array(test_iter_acc_cat),0)),
                         conv_z2r(np.mean(np.array(test_iter_acc_cat),0)+1.96*np.std(np.array(test_iter_acc_cat),0)), color=colorr[cdi], alpha=0.4,
                         edgecolor='none')
    elif errorbar == 'standard_error':
        ax2.fill_between(np.arange(donne)+1, conv_z2r(np.mean(np.array(test_iter_acc_cat),0)-np.std(np.array(test_iter_acc_cat),0)/np.sqrt(nseed)),
                         conv_z2r(np.mean(np.array(test_iter_acc_cat),0)+np.std(np.array(test_iter_acc_cat),0)/np.sqrt(nseed)), color=colorr[cdi], alpha=0.4,
                         edgecolor='none')
for ax in [ax1, ax2]:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel('training iterations', fontsize=14)
ax1.set_ylabel('test loss', fontsize=14)
ax2.set_ylabel('test accuracy', fontsize=14)

print('test loss minimized at: '+str(np.mean(minimum)))
