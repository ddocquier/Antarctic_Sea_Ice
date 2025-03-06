#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute Liang index over observed period (1982-2023) for each different member
Compute ensemble mean by combining p values through Fisher test

Large ensembles: EC-Earth3 (SMHI-LENS), CESM2-LE, MPI-ESM1-2-LR, CanESM5, ACCESS-ESM1-5

Variables:
Summer sea-ice extent (DJF)
Previous sea-ice extent
Previous Antarctic mean surface air temperature (<60S)
Previous Antarctic mean SST (<60S)
Previous Southern Annular Mode (SAM)
Previous Amundsen Sea Low (ASL)
Previous Niño3.4
Previous DMI

Last updated: 31/01/2025

@author: David Docquier
"""

# Import libraries
import numpy as np
import sys
import matplotlib.pyplot as plt
import seaborn as sns # for creating a matrix plot
from matplotlib.patches import Rectangle # for drawing rectangles around elements in a matrix
from scipy.stats import combine_pvalues # for combining p values (Fisher test)

# Import my functions
sys.path.append('/home/dadocq/Documents/Codes/Liang/')
from function_liang_nvar import compute_liang_nvar

# Parameters
season = 'OND' # MAM, AMJ, MJJ, JJA, JAS, ASO, SON, OND / default: OND (spring) or JAS (winter)
model = 'CanESM5' # SMHI-LENS; CESM2-LE; MPI-ESM1-2-LR; CanESM5; ACCESS-ESM1-5
if model == 'SMHI-LENS':
    string_model = 'EC-Earth3'
elif model == 'CESM2-LE':
    string_model = 'CESM2'
else:
    string_model = model
nvar = 8 # number of variables (1: SSIE; 2: PSIE; 3: T_2m; 4: SST; 5: SAM; 6: ASL; 7: ENSO; 8: DMI)
first_year = 1970 # starting year included in the computation (1970 or 1982)
if first_year == 1982:
    last_year = 2023
elif first_year == 1970:
    last_year = 2099
dt = 1 # time step (years)
n_iter = 1000 # number of bootstrap realizations
conf = 1.96 # for models: 1.96 if 95% confidence interval (normal distribution); 1.65 if 90% and 2.57 if 99%
compute_liang = True # True compute Liang index; False: load existing value
save_var = True
save_fig = True

# Interpolate NaN values
def interpolate_nan(array_like):
    array = array_like.copy()
    nans = np.isnan(array)
    
    def get_x(a):
        return a.nonzero()[0]

    array[nans] = np.interp(get_x(nans), get_x(~nans), array[~nans])

    return array

# Working directories
dir_input = '/home/dadocq/Documents/Papers/My_Papers/RESIST_Antarctic/output/seasons/'
dir_fig = '/home/dadocq/Documents/Papers/My_Papers/RESIST_Antarctic/figures/seasons/'

# Number of members
if model == 'ACCESS-ESM1-5':
    n_members = 40
else:
    n_members = 50

# Function to test significance (based on the confidence interval)
def compute_sig(var,error,conf):
    if (var-conf*error < 0. and var+conf*error < 0.) or (var-conf*error > 0. and var+conf*error > 0.):
        sig = 1
    else:
        sig = 0
    z = var / error # z score
    pval = np.exp(-0.717 * z - 0.416 * z**2.) # p value for 95% confidence interval (https://www.bmj.com/content/343/bmj.d2304)
    return sig,pval

# Load variables (saved via save_timeseries.py)
filename = dir_input + string_model + '_Antarctic_timeseries_' + season + '.npy'
sie_summer,sie_previous,tas,sst,sam,asl,nino,dmi = np.load(filename,allow_pickle=True)

# Set NaN values to 0 for summer SIE
sie_summer[np.isnan(sie_summer)] = 0

# Interpolate NaN values for ASL
for m in np.arange(n_members):
    asl[m,:] = interpolate_nan(asl[m,:])

# Shift summer SIE to the left (so that it lags other variables)
sie_summer = np.roll(sie_summer,-1,axis=1)

# Take years of interest and save variables
ind_first_year = int(first_year-1970)
ind_last_year = int(last_year-1970+1)
nyears_new = ind_last_year - ind_first_year
sie_summer2 = np.zeros((n_members,nyears_new))
sie_previous2 = np.zeros((n_members,nyears_new))
tas2 = np.zeros((n_members,nyears_new))
sam2 = np.zeros((n_members,nyears_new))
sst2 = np.zeros((n_members,nyears_new))
asl2 = np.zeros((n_members,nyears_new))
nino2 = np.zeros((n_members,nyears_new))
dmi2 = np.zeros((n_members,nyears_new))
for m in np.arange(n_members):
    sie_summer2[m,:] = sie_summer[m,ind_first_year:ind_last_year]
    sie_previous2[m,:] = sie_previous[m,ind_first_year:ind_last_year]
    tas2[m,:] = tas[m,ind_first_year:ind_last_year]
    sam2[m,:] = sam[m,ind_first_year:ind_last_year]
    sst2[m,:] = sst[m,ind_first_year:ind_last_year]
    asl2[m,:] = asl[m,ind_first_year:ind_last_year]
    nino2[m,:] = nino[m,ind_first_year:ind_last_year]
    dmi2[m,:] = dmi[m,ind_first_year:ind_last_year]

# Compute ensemble mean
sie_summer_ensmean = np.nanmean(sie_summer2,axis=0)
sie_previous_ensmean = np.nanmean(sie_previous2,axis=0)
tas_ensmean = np.nanmean(tas2,axis=0)
sam_ensmean = np.nanmean(sam2,axis=0)
sst_ensmean = np.nanmean(sst2,axis=0)
asl_ensmean = np.nanmean(asl2,axis=0)
nino_ensmean = np.nanmean(nino2,axis=0)
dmi_ensmean = np.nanmean(dmi2,axis=0)
    
# Detrend data (remove ensemble mean)
for m in np.arange(n_members):
    sie_summer2[m,:] = sie_summer2[m,:] - sie_summer_ensmean
    sie_previous2[m,:] = sie_previous2[m,:] - sie_previous_ensmean
    tas2[m,:] = tas2[m,:] - tas_ensmean
    sam2[m,:] = sam2[m,:] - sam_ensmean
    sst2[m,:] = sst2[m,:] - sst_ensmean
    asl2[m,:] = asl2[m,:] - asl_ensmean
    nino2[m,:] = nino2[m,:] - nino_ensmean
    dmi2[m,:] = dmi2[m,:] - dmi_ensmean

# Compute relative transfer of information (tau) and correlation coefficient (R) and save in files
filename = dir_input + 'Liang_' + model + '_' + season + '_' + str(n_iter) + 'boot_' + str(first_year) + '-' + str(last_year) + '_members.npy'
if compute_liang == True:
    tau = np.zeros((n_members,nvar,nvar))
    R = np.zeros((n_members,nvar,nvar))
    error_tau = np.zeros((n_members,nvar,nvar))
    error_R = np.zeros((n_members,nvar,nvar))
    for m in np.arange(n_members):
        print(m)
        xx = np.array((sie_summer2[m,:],sie_previous2[m,:],tas2[m,:],sst2[m,:],sam2[m,:],asl2[m,:],nino2[m,:],dmi2[m,:]))
        notused,tau[m,:,:],R[m,:,:],notused,error_tau[m,:,:],error_R[m,:,:] = compute_liang_nvar(xx,dt,n_iter)
    if save_var == True:
        np.save(filename,[tau,R,error_tau,error_R])
else:
    tau,R,error_tau,error_R = np.load(filename,allow_pickle=True)

# Compute ensemble mean tau and R
tau_ensmean = np.nanmean(tau,axis=0)
R_ensmean = np.nanmean(R,axis=0)

# Compute p-value and significance of different members (different from 0) based on the confidence interval
sig_tau = np.zeros((n_members,nvar,nvar))
sig_R = np.zeros((n_members,nvar,nvar))
pval_tau = np.zeros((n_members,nvar,nvar))
pval_R = np.zeros((n_members,nvar,nvar))
for m in np.arange(n_members):
    for i in np.arange(nvar):
        for j in np.arange(nvar):
            sig_tau[m,i,j],pval_tau[m,i,j] = compute_sig(tau[m,i,j],error_tau[m,i,j],conf)
            sig_R[m,i,j],pval_R[m,i,j] = compute_sig(R[m,i,j],error_R[m,i,j],conf)

# Combine p-values (Fisher test) of different members
pval_tau_fisher = np.zeros((nvar,nvar))
pval_R_fisher = np.zeros((nvar,nvar))
for i in np.arange(nvar):
    for j in np.arange(nvar):
        pval_tau_fisher[i,j] = combine_pvalues(pval_tau[:,i,j],method='fisher')[1]
        pval_R_fisher[i,j] = combine_pvalues(pval_R[:,i,j],method='fisher')[1]

# Labels
label_names = ['SSIE','PSIE','T$_{2m}$','SST','SAM','ASL','N3.4','DMI']

# Plot options
fig,ax = plt.subplots(1,2,figsize=(24,13))
fig.subplots_adjust(left=0.05,bottom=0.01,right=0.95,top=0.85,wspace=0.15,hspace=0.15)
cmap_tau = plt.cm.YlOrRd._resample(15)
cmap_R = plt.cm.bwr._resample(16)
sns.set(font_scale=1.8)

# Matrix of tau
tau_annotations = np.round(np.abs(tau_ensmean),2)
tau_plot = sns.heatmap(np.abs(tau_ensmean),annot=tau_annotations,fmt='',annot_kws={'color':'k','fontsize':18},cmap=cmap_tau,
    cbar_kws={'label':r'$\|\tau\|$ ($\%$)','orientation':'horizontal','pad':0.05},vmin=0,vmax=70,
    xticklabels=label_names,yticklabels=label_names,linewidths=0.1,linecolor='gray',ax=ax[0])
tau_plot.set_title(r'Rate of information transfer $\|\tau\|$' + '\n',fontsize=24)
tau_plot.set_title('a \n',loc='left',fontsize=32,fontweight='bold')
for j in np.arange(nvar):
    for i in np.arange(nvar):
        if pval_tau_fisher[j,i] < 0.05:
            tau_plot.add_patch(Rectangle((i+0.05,j+0.2),0.9,0.6,fill=False,edgecolor='blue',linewidth=3))
tau_plot.set_xticklabels(tau_plot.get_xmajorticklabels(),fontsize=20)
tau_plot.xaxis.set_ticks_position('top')
tau_plot.set_xlabel('TO...',loc='left',fontsize=20)
tau_plot.xaxis.set_label_position('top')
tau_plot.set_yticklabels(tau_plot.get_ymajorticklabels(),fontsize=20)
tau_plot.set_ylabel('FROM...',loc='top',fontsize=20)
                   
# Matrix of R
R_annotations = np.round(R_ensmean,2)
R_plot = sns.heatmap(R_ensmean,annot=R_annotations,fmt='',annot_kws={'color':'k','fontsize':18},cmap=cmap_R,
    cbar_kws={'label':'$R$','orientation':'horizontal','pad':0.05},vmin=-1,vmax=1,
    xticklabels=label_names,yticklabels=label_names,linewidths=0.1,linecolor='gray',ax=ax[1])
R_plot.set_title('Correlation coefficient $R$ \n ',fontsize=24)
R_plot.set_title('b \n',loc='left',fontsize=32,fontweight='bold')
for j in np.arange(nvar):
    for i in np.arange(nvar):
        if pval_R_fisher[j,i] < 0.05 and j != i:
               R_plot.add_patch(Rectangle((i+0.05,j+0.2),0.9,0.6,fill=False,edgecolor='black',linewidth=3))
R_plot.set_xticklabels(R_plot.get_xmajorticklabels(),fontsize=20)
R_plot.xaxis.set_ticks_position('top')
R_plot.set_yticklabels(R_plot.get_ymajorticklabels(),fontsize=20)

# Save figure
if save_fig == True:
    filename = dir_fig + 'Liang_' + model + '_' + season + '_' + str(n_iter) + 'boot_' + str(first_year) + '-' + str(last_year) + '_members.png'
    fig.savefig(filename)