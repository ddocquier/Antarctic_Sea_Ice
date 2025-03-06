#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot correlation from models (separate members) over 1970-2099

Liang index computed via compute_liang_seasons_test_members.py

Large ensembles: EC-Earth3 (SMHI-LENS), CESM2-LE, MPI-ESM1-2-LR, CanESM5, ACCESS-ESM1-5

Target variable: Summer sea-ice extent (DJF)

Drivers:
Previous winter/spring sea-ice extent (JAS or OND)
Previous Antarctic mean surface air temperature (<60S; JAS or OND)
Previous Antarctic mean SST (<60S; JAS or OND)
Previous Southern Annular Mode (SAM; JAS or OND)
Previous Amundsen Sea Low (ASL; JAS or OND)
Previous Niño3.4 (JAS or OND)
Previous DMI (JAS or OND)

Last updated: 12/02/2024

@author: David Docquier
"""

# Standard libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import combine_pvalues # for combining p values (Fisher test)

# Parameters
first_year = 1970 # starting year included in the computation (1970 or 1982)
last_year = 2099
nvar = 8 # number of variables (1: SSIE; 2: PSIE; 3: T_2m; 4: SST; 5: SAM; 6: ASL; 7: ENSO; 8: DMI)
conf_obs = 1.96 # for observations
save_fig = True

# Function to compute p value (based on the standard error)
def compute_pval(var,error):
    z = var / error # z score
    pval = np.exp(-0.717 * z - 0.416 * z**2.) # p value for 95% confidence interval (https://www.bmj.com/content/343/bmj.d2304)
    return pval

# Working directories
dir_input = '/home/dadocq/Documents/Papers/My_Papers/RESIST_Antarctic/output/seasons/'
dir_fig = '/home/dadocq/Documents/Papers/My_Papers/RESIST_Antarctic/LaTeX/'

# Load ACCESS-ESM1-5
filename = dir_input + 'Liang_ACCESS-ESM1-5_OND_1000boot_' + str(first_year) + '-' + str(last_year) + '_members.npy'
tau_access,R_access,error_tau_access,error_R_access = np.load(filename,allow_pickle=True)
tau_access_ensmean = np.nanmean(tau_access,axis=0)
R_access_ensmean = np.nanmean(R_access,axis=0)
n_members_access = np.size(tau_access,0)

# Load CanESM5
filename = dir_input + 'Liang_CanESM5_OND_1000boot_' + str(first_year) + '-' + str(last_year) + '_members.npy'
tau_canesm,R_canesm,error_tau_canesm,error_R_canesm = np.load(filename,allow_pickle=True)
tau_canesm_ensmean = np.nanmean(tau_canesm,axis=0)
R_canesm_ensmean = np.nanmean(R_canesm,axis=0)
n_members_canesm = np.size(tau_canesm,0)

# Load CESM2
filename = dir_input + 'Liang_CESM2-LE_OND_1000boot_' + str(first_year) + '-' + str(last_year) + '_members.npy'
tau_cesm,R_cesm,error_tau_cesm,error_R_cesm = np.load(filename,allow_pickle=True)
tau_cesm_ensmean = np.nanmean(tau_cesm,axis=0)
R_cesm_ensmean = np.nanmean(R_cesm,axis=0)
n_members_cesm = np.size(tau_cesm,0)

# Load EC-Earth3
filename = dir_input + 'Liang_SMHI-LENS_OND_1000boot_' + str(first_year) + '-' + str(last_year) + '_members.npy'
tau_ecearth,R_ecearth,error_tau_ecearth,error_R_ecearth = np.load(filename,allow_pickle=True)
tau_ecearth_ensmean = np.nanmean(tau_ecearth,axis=0)
R_ecearth_ensmean = np.nanmean(R_ecearth,axis=0)
n_members_ecearth = np.size(tau_ecearth,0)

# Load MPI-ESM1.2-LR
filename = dir_input + 'Liang_MPI-ESM1-2-LR_OND_1000boot_' + str(first_year) + '-' + str(last_year) + '_members.npy'
tau_mpi,R_mpi,error_tau_mpi,error_R_mpi = np.load(filename,allow_pickle=True)
tau_mpi_ensmean = np.nanmean(tau_mpi,axis=0)
R_mpi_ensmean = np.nanmean(R_mpi,axis=0)
n_members_mpi = np.size(tau_mpi,0)

# Compute p value of ACCESS-ESM1-5 for each member
pval_tau_access = np.zeros((n_members_access,nvar,nvar))
pval_R_access = np.zeros((n_members_access,nvar,nvar))
for m in np.arange(n_members_access):
    for i in np.arange(nvar):
        for j in np.arange(nvar):
            pval_tau_access[m,i,j] = compute_pval(tau_access[m,i,j],error_tau_access[m,i,j])
            pval_R_access[m,i,j] = compute_pval(R_access[m,i,j],error_R_access[m,i,j])
            
# Compute p value of CanESM5 for each member
pval_tau_canesm = np.zeros((n_members_canesm,nvar,nvar))
pval_R_canesm = np.zeros((n_members_canesm,nvar,nvar))
for m in np.arange(n_members_canesm):
    for i in np.arange(nvar):
        for j in np.arange(nvar):
            pval_tau_canesm[m,i,j] = compute_pval(tau_canesm[m,i,j],error_tau_canesm[m,i,j])
            pval_R_canesm[m,i,j] = compute_pval(R_canesm[m,i,j],error_R_canesm[m,i,j])

# Compute p value of CESM2 for each member
pval_tau_cesm = np.zeros((n_members_cesm,nvar,nvar))
pval_R_cesm = np.zeros((n_members_cesm,nvar,nvar))
for m in np.arange(n_members_cesm):
    for i in np.arange(nvar):
        for j in np.arange(nvar):
            pval_tau_cesm[m,i,j] = compute_pval(tau_cesm[m,i,j],error_tau_cesm[m,i,j])
            pval_R_cesm[m,i,j] = compute_pval(R_cesm[m,i,j],error_R_cesm[m,i,j])
            
# Compute p value of EC-Earth for each member
pval_tau_ecearth = np.zeros((n_members_ecearth,nvar,nvar))
pval_R_ecearth = np.zeros((n_members_ecearth,nvar,nvar))
for m in np.arange(n_members_ecearth):
    for i in np.arange(nvar):
        for j in np.arange(nvar):
            pval_tau_ecearth[m,i,j] = compute_pval(tau_ecearth[m,i,j],error_tau_ecearth[m,i,j])
            pval_R_ecearth[m,i,j] = compute_pval(R_ecearth[m,i,j],error_R_ecearth[m,i,j])

# Compute p value of MPI-ESM1.2-LR for each member
pval_tau_mpi = np.zeros((n_members_mpi,nvar,nvar))
pval_R_mpi = np.zeros((n_members_mpi,nvar,nvar))
for m in np.arange(n_members_mpi):
    for i in np.arange(nvar):
        for j in np.arange(nvar):
            pval_tau_mpi[m,i,j] = compute_pval(tau_mpi[m,i,j],error_tau_mpi[m,i,j])
            pval_R_mpi[m,i,j] = compute_pval(R_mpi[m,i,j],error_R_mpi[m,i,j])
            
# Combine p-values (Fisher test) of different members
pval_tau_fisher_access = np.zeros((nvar,nvar))
pval_R_fisher_access = np.zeros((nvar,nvar))
pval_tau_fisher_canesm = np.zeros((nvar,nvar))
pval_R_fisher_canesm = np.zeros((nvar,nvar))
pval_tau_fisher_cesm = np.zeros((nvar,nvar))
pval_R_fisher_cesm = np.zeros((nvar,nvar))
pval_tau_fisher_ecearth = np.zeros((nvar,nvar))
pval_R_fisher_ecearth = np.zeros((nvar,nvar))
pval_tau_fisher_mpi = np.zeros((nvar,nvar))
pval_R_fisher_mpi = np.zeros((nvar,nvar))
for i in np.arange(nvar):
    for j in np.arange(nvar):
        pval_tau_fisher_access[i,j] = combine_pvalues(pval_tau_access[:,i,j],method='fisher')[1]
        pval_R_fisher_access[i,j] = combine_pvalues(pval_R_access[:,i,j],method='fisher')[1]
        pval_tau_fisher_canesm[i,j] = combine_pvalues(pval_tau_canesm[:,i,j],method='fisher')[1]
        pval_R_fisher_canesm[i,j] = combine_pvalues(pval_R_canesm[:,i,j],method='fisher')[1]
        pval_tau_fisher_cesm[i,j] = combine_pvalues(pval_tau_cesm[:,i,j],method='fisher')[1]
        pval_R_fisher_cesm[i,j] = combine_pvalues(pval_R_cesm[:,i,j],method='fisher')[1]
        pval_tau_fisher_ecearth[i,j] = combine_pvalues(pval_tau_ecearth[:,i,j],method='fisher')[1]
        pval_R_fisher_ecearth[i,j] = combine_pvalues(pval_R_ecearth[:,i,j],method='fisher')[1]
        pval_tau_fisher_mpi[i,j] = combine_pvalues(pval_tau_mpi[:,i,j],method='fisher')[1]
        pval_R_fisher_mpi[i,j] = combine_pvalues(pval_R_mpi[:,i,j],method='fisher')[1]

# Plot options
index = np.arange(nvar-1)
bar_width = 1

# Labels
label_names = ['PSIE','T$_{2m}$','SST','SAM','ASL','Niño3.4','DMI']

# Figure
fig,ax = plt.subplots(figsize=(9,4))
fig.subplots_adjust(left=0.15,bottom=0.12,right=0.94,top=0.93)

# R ACCESS
ax.plot(index[0]+0.8,R_access_ensmean[1,0],'o',color='orange',markersize=12,label='ACCESS-ESM1.5 (40m)')
for i in np.arange(1,np.size(index)+1):
    if pval_R_fisher_access[i,0] < 0.05:
        ax.plot(index[i-1]+0.8,R_access_ensmean[i,0],'ko',markersize=17)
    ax.plot(index[i-1]+0.8,R_access_ensmean[i,0],'o',color='orange',markersize=12)
    for m in np.arange(n_members_access):
        ax.plot(index[i-1]+0.8,R_access[m,i,0],'.',color='orange',markersize=5)

# R CanESM
ax.plot(index[0]+0.9,R_canesm_ensmean[1,0],'go',markersize=12,label='CanESM5 (50m)')
for i in np.arange(1,np.size(index)+1):
    if pval_R_fisher_canesm[i,0] < 0.05:
        ax.plot(index[i-1]+0.9,R_canesm_ensmean[i,0],'ko',markersize=17)
    ax.plot(index[i-1]+0.9,R_canesm_ensmean[i,0],'go',markersize=12)
    for m in np.arange(n_members_canesm):
        ax.plot(index[i-1]+0.9,R_canesm[m,i,0],'g.',markersize=5)

# R CESM
ax.plot(index[0]+1,R_cesm_ensmean[1,0],'bo',markersize=12,label='CESM2 (50m)')
for i in np.arange(1,np.size(index)+1):
    if pval_R_fisher_cesm[i,0] < 0.05:
        ax.plot(index[i-1]+1,R_cesm_ensmean[i,0],'ko',markersize=17)
    ax.plot(index[i-1]+1,R_cesm_ensmean[i,0],'bo',markersize=12)
    for m in np.arange(n_members_cesm):
        ax.plot(index[i-1]+1,R_cesm[m,i,0],'b.',markersize=5)
 
# R EC-Earth
ax.plot(index[0]+1.1,R_ecearth_ensmean[1,0],'ro',markersize=12,label='EC-Earth3 (50m)')
for i in np.arange(1,np.size(index)+1):
    if pval_R_fisher_ecearth[i,0] < 0.05:
        ax.plot(index[i-1]+1.1,R_ecearth_ensmean[i,0],'ko',markersize=17)
    ax.plot(index[i-1]+1.1,R_ecearth_ensmean[i,0],'ro',markersize=12)
    for m in np.arange(n_members_ecearth):
        ax.plot(index[i-1]+1.1,R_ecearth[m,i,0],'r.',markersize=5)

# R MPI
ax.plot(index[0]+1.2,R_mpi_ensmean[1,0],'o',color='gray',markersize=12,label='MPI-ESM1.2-LR (50m)')
for i in np.arange(1,np.size(index)+1):
    if pval_R_fisher_mpi[i,0] < 0.05:
        ax.plot(index[i-1]+1.2,R_mpi_ensmean[i,0],'ko',markersize=17)
    ax.plot(index[i-1]+1.2,R_mpi_ensmean[i,0],'o',color='gray',markersize=12)
    for m in np.arange(n_members_mpi):
        ax.plot(index[i-1]+1.2,R_mpi[m,i,0],'.',color='gray',markersize=5)

# Labels and legend
ax.set_ylabel('Correlation coefficient $R$',fontsize=20)
ax.tick_params(axis='both',labelsize=16)
ax.set_xticks(np.arange(1,np.size(index)+1))
ax.set_xticklabels(label_names)
ax.grid(linestyle='--')
ax.set_ylim(-1,1)
ax.axes.axhline(y=0,color='k',linestyle='--')
ax.legend(loc='upper right',fontsize=12,shadow=True,frameon=False,ncol=2)

# Save figure
if save_fig == True:
    filename = dir_fig + 'fig_a2.pdf'
    fig.savefig(filename)