#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Figures 3 and 5: Plot Liang index and correlation from models over whole period (1970-2099) as well as observations (1982-2023)

Model large ensembles: EC-Earth3 (SMHI-LENS), CESM2-LE, MPI-ESM1-2-LR, CanESM5, ACCESS-ESM1-5
Liang index computed via compute_liang_seasons.py

Observations
Liang index computed via compute_liang_seasons_obs.py

Target variable: Summer sea-ice extent (DJF)

Drivers:
Previous winter/spring sea-ice extent (JAS or OND)
Previous Antarctic mean surface air temperature (<60S; JAS or OND)
Previous Antarctic mean SST (<60S; JAS or OND)
Previous Southern Annular Mode (SAM; JAS or OND)
Previous Amundsen Sea Low (ASL; JAS or OND)
Previous Niño3.4 (JAS or OND)
Previous DMI (JAS or OND)

Last updated: 05/03/2024

@author: David Docquier
"""

# Standard libraries
import numpy as np
import matplotlib.pyplot as plt

# Parameters
season = 'JAS' # JAS (previous winter), OND (previous spring)
nvar = 8 # number of variables (1: SSIE; 2: PSIE; 3: T_2m; 4: SST; 5: SAM; 6: ASL; 7: ENSO; 8: DMI)
n_iter = 1000 # number of bootstrap realizations; default: 1000
conf = 2.57 # for models: 1.96 if 95% confidence interval (normal distribution); 1.65 if 90% and 2.57 if 99%
conf_obs = 2.57 # for observations
save_fig = True

# Function to test significance (based on the confidence interval)
def compute_sig(var,error,conf):
    if np.abs(var)-conf*error > 0. and np.abs(var)+conf*error > 0.:
        sig = 1
    else:
        sig = 0
    return sig

# Working directories
dir_input = '/home/dadocq/Documents/Papers/My_Papers/RESIST_Antarctic/output/seasons/'
dir_fig = '/home/dadocq/Documents/Papers/My_Papers/RESIST_Antarctic/LaTeX/'

# Load SMHI-LENS
filename = dir_input + 'Liang_SMHI-LENS_' + season + '_' + str(n_iter) + 'boot.npy'
tau_ecearth,R_ecearth,error_tau_ecearth,error_R_ecearth = np.load(filename,allow_pickle=True)

# Load CESM2-LE
filename = dir_input + 'Liang_CESM2-LE_' + season + '_' + str(n_iter) + 'boot.npy'
tau_cesm,R_cesm,error_tau_cesm,error_R_cesm = np.load(filename,allow_pickle=True)

# Load MPI-ESM1-2-LR
filename = dir_input + 'Liang_MPI-ESM1-2-LR_' + season + '_' + str(n_iter) + 'boot.npy'
tau_mpi,R_mpi,error_tau_mpi,error_R_mpi = np.load(filename,allow_pickle=True)

# Load CanESM5
filename = dir_input + 'Liang_CanESM5_' + season + '_' + str(n_iter) + 'boot.npy'
tau_canesm,R_canesm,error_tau_canesm,error_R_canesm = np.load(filename,allow_pickle=True)

# Load ACCESS-ESM1-5
filename = dir_input + 'Liang_ACCESS-ESM1-5_' + season + '_' + str(n_iter) + 'boot.npy'
tau_access,R_access,error_tau_access,error_R_access = np.load(filename,allow_pickle=True)

# Compute significance of SMHI-LENS and CESM2-LE
sig_tau_ecearth = np.zeros((nvar,nvar))
sig_R_ecearth = np.zeros((nvar,nvar))
sig_tau_cesm = np.zeros((nvar,nvar))
sig_R_cesm = np.zeros((nvar,nvar))
for i in np.arange(nvar):
    for j in np.arange(nvar):
        sig_tau_ecearth[i,j] = compute_sig(tau_ecearth[i,j],error_tau_ecearth[i,j],conf)
        sig_R_ecearth[i,j] = compute_sig(R_ecearth[i,j],error_R_ecearth[i,j],conf)
        sig_tau_cesm[i,j] = compute_sig(tau_cesm[i,j],error_tau_cesm[i,j],conf)
        sig_R_cesm[i,j] = compute_sig(R_cesm[i,j],error_R_cesm[i,j],conf)
            
# Compute significance of MPI-ESM1-2-LR
sig_tau_mpi = np.zeros((nvar,nvar))
sig_R_mpi = np.zeros((nvar,nvar))
for i in np.arange(nvar):
    for j in np.arange(nvar):
        sig_tau_mpi[i,j] = compute_sig(tau_mpi[i,j],error_tau_mpi[i,j],conf)
        sig_R_mpi[i,j] = compute_sig(R_mpi[i,j],error_R_mpi[i,j],conf)
            
# Compute significance of CanESM5
sig_tau_canesm = np.zeros((nvar,nvar))
sig_R_canesm = np.zeros((nvar,nvar))
for i in np.arange(nvar):
    for j in np.arange(nvar):
        sig_tau_canesm[i,j] = compute_sig(tau_canesm[i,j],error_tau_canesm[i,j],conf)
        sig_R_canesm[i,j] = compute_sig(R_canesm[i,j],error_R_canesm[i,j],conf)

# Compute significance of ACCESS-ESM1-5
sig_tau_access = np.zeros((nvar,nvar))
sig_R_access = np.zeros((nvar,nvar))
for i in np.arange(nvar):
    for j in np.arange(nvar):
        sig_tau_access[i,j] = compute_sig(tau_access[i,j],error_tau_access[i,j],conf)
        sig_R_access[i,j] = compute_sig(R_access[i,j],error_R_access[i,j],conf)

# Load observations
filename = dir_input + 'Liang_obs_' + season + '.npy'
tau_obs,R_obs,error_tau_obs,error_R_obs = np.load(filename,allow_pickle=True)

# Compute significance of obs.
sig_tau_obs = np.zeros((nvar,nvar))
sig_R_obs = np.zeros((nvar,nvar))
for i in np.arange(nvar):
    for j in np.arange(nvar):
        sig_tau_obs[i,j] = compute_sig(tau_obs[i,j],error_tau_obs[i,j],conf_obs)
        sig_R_obs[i,j] = compute_sig(R_obs[i,j],error_R_obs[i,j],conf_obs)

# Plot options
index = np.arange(nvar-1)
bar_width = 1

# Labels
label_names = ['PSIE','T$_{2m}$','SST','SAM','ASL','Niño3.4','DMI']

# Figure
fig,ax = plt.subplots(2,1,figsize=(12,10))
fig.subplots_adjust(left=0.11,bottom=0.08,right=0.94,top=0.93,hspace=0.4,wspace=0.25)

#######
# tau #
#######

# tau ACCESS
ax[0].errorbar(index[0]+0.8,np.abs(tau_access[1,0]),yerr=conf*error_tau_access[1,0],fmt='o',color='orange',markersize=12,label='ACCESS-ESM1.5')
for i in np.arange(1,np.size(index)+1):
    if sig_tau_access[i,0] == 1:
        ax[0].plot(index[i-1]+0.8,np.abs(tau_access[i,0]),'ko',markersize=17)
    ax[0].errorbar(index[i-1]+0.8,np.abs(tau_access[i,0]),yerr=conf*error_tau_access[i,0],fmt='o',color='orange',markersize=12)
    
# tau CanESM
ax[0].errorbar(index[0]+0.9,np.abs(tau_canesm[1,0]),yerr=conf*error_tau_canesm[1,0],fmt='go',markersize=12,label='CanESM5')
for i in np.arange(1,np.size(index)+1):
    if sig_tau_canesm[i,0] == 1:
        ax[0].plot(index[i-1]+0.9,np.abs(tau_canesm[i,0]),'ko',markersize=17)
    ax[0].errorbar(index[i-1]+0.9,np.abs(tau_canesm[i,0]),yerr=conf*error_tau_canesm[i,0],fmt='go',markersize=12)

# tau CESM
ax[0].errorbar(index[0]+1,np.abs(tau_cesm[1,0]),yerr=conf*error_tau_cesm[1,0],fmt='bo',markersize=12,label='CESM2')
for i in np.arange(1,np.size(index)+1):
    if sig_tau_cesm[i,0] == 1:
        ax[0].plot(index[i-1]+1,np.abs(tau_cesm[i,0]),'ko',markersize=17)
    ax1=ax[0].errorbar(index[i-1]+1,np.abs(tau_cesm[i,0]),yerr=conf*error_tau_cesm[i,0],fmt='bo',markersize=12)

# tau EC-Earth
ax[0].errorbar(index[0]+1.1,np.abs(tau_ecearth[1,0]),yerr=conf*error_tau_ecearth[1,0],fmt='ro',markersize=12,label='EC-Earth3')
for i in np.arange(1,np.size(index)+1):
    if sig_tau_ecearth[i,0] == 1:
        ax[0].plot(index[i-1]+1.1,np.abs(tau_ecearth[i,0]),'ko',markersize=17)
    ax[0].errorbar(index[i-1]+1.1,np.abs(tau_ecearth[i,0]),yerr=conf*error_tau_ecearth[i,0],fmt='ro',markersize=12)

# tau MPI
ax[0].errorbar(index[0]+1.2,np.abs(tau_mpi[1,0]),yerr=conf*error_tau_mpi[1,0],fmt='o',color='gray',markersize=12,label='MPI-ESM1.2-LR')
for i in np.arange(1,np.size(index)+1):
    if sig_tau_mpi[i,0] == 1:
        ax[0].plot(index[i-1]+1.2,np.abs(tau_mpi[i,0]),'ko',markersize=17)
    ax[0].errorbar(index[i-1]+1.2,np.abs(tau_mpi[i,0]),yerr=conf*error_tau_mpi[i,0],fmt='o',color='gray',markersize=12)

# tau obs.
ax1 = ax[0].errorbar(index[0]+1.3,np.abs(tau_obs[1,0]),yerr=conf_obs*error_tau_obs[1,0],fmt='x',color='black',markersize=12,label='Obs. (1982-2023)')
ax1[-1][0].set_linestyle('--')
for i in np.arange(1,np.size(index)+1):
    if sig_tau_obs[i,0] == 1:
        ax[0].plot(index[i-1]+1.3,np.abs(tau_obs[i,0]),'ko',markersize=17,fillstyle='none')
    ax1 = ax[0].errorbar(index[i-1]+1.3,np.abs(tau_obs[i,0]),yerr=conf_obs*error_tau_obs[i,0],fmt='x',color='black',markersize=12)
    ax1[-1][0].set_linestyle('--')

# Labels and legend
ax[0].set_ylabel(r'Information transfer $|\tau|$ ($\%$)',fontsize=20)
ax[0].tick_params(axis='both',labelsize=16)
ax[0].legend(loc='upper right',fontsize=14,shadow=True,frameon=False,ncol=2)
ax[0].set_xticks(np.arange(1,np.size(index)+1))
ax[0].set_xticklabels(label_names)
ax[0].grid(linestyle='--')
ax[0].set_ylim(-2,25)
ax[0].axes.axhline(y=0,color='k',linestyle='--')
ax[0].set_title('a',loc='left',fontsize=25,fontweight='bold')

#####
# R #
#####
    
# R ACCESS
ax[1].errorbar(index[0]+0.8,R_access[1,0],yerr=conf*error_R_access[1,0],fmt='o',color='orange',markersize=12,label='ACCESS-ESM1.5')
for i in np.arange(1,np.size(index)+1):
    if sig_R_access[i,0] == 1:
        ax[1].plot(index[i-1]+0.8,R_access[i,0],'ko',markersize=17)
    ax[1].errorbar(index[i-1]+0.8,R_access[i,0],yerr=conf*error_R_access[i,0],fmt='o',color='orange',markersize=12)
    
# R CanESM
ax[1].errorbar(index[0]+0.9,R_canesm[1,0],yerr=conf*error_R_canesm[1,0],fmt='go',markersize=12,label='CanESM5')
for i in np.arange(1,np.size(index)+1):
    if sig_R_canesm[i,0] == 1:
        ax[1].plot(index[i-1]+0.9,R_canesm[i,0],'ko',markersize=17)
    ax[1].errorbar(index[i-1]+0.9,R_canesm[i,0],yerr=conf*error_R_canesm[i,0],fmt='go',markersize=12)

# R CESM
ax[1].errorbar(index[0]+1,R_cesm[1,0],yerr=conf*error_R_cesm[1,0],fmt='bo',markersize=12,label='CESM2')
for i in np.arange(1,np.size(index)+1):
    if sig_R_cesm[i,0] == 1:
        ax[1].plot(index[i-1]+1,R_cesm[i,0],'ko',markersize=17)
    ax[1].errorbar(index[i-1]+1,R_cesm[i,0],yerr=conf*error_R_cesm[i,0],fmt='bo',markersize=12)

# R EC-Earth
ax[1].errorbar(index[0]+1.1,R_ecearth[1,0],yerr=conf*error_R_ecearth[1,0],fmt='ro',markersize=12,label='EC-Earth3')
for i in np.arange(1,np.size(index)+1):
    if sig_R_ecearth[i,0] == 1:
        ax[1].plot(index[i-1]+1.1,R_ecearth[i,0],'ko',markersize=17)
    ax[1].errorbar(index[i-1]+1.1,R_ecearth[i,0],yerr=conf*error_R_ecearth[i,0],fmt='ro',markersize=12)

# R MPI
ax[1].errorbar(index[0]+1.2,R_mpi[1,0],yerr=conf*error_R_mpi[1,0],fmt='o',color='gray',markersize=12,label='MPI-ESM1.2-LR')
for i in np.arange(1,np.size(index)+1):
    if sig_R_mpi[i,0] == 1:
        ax[1].plot(index[i-1]+1.2,R_mpi[i,0],'ko',markersize=17)
    ax[1].errorbar(index[i-1]+1.2,R_mpi[i,0],yerr=conf*error_R_mpi[i,0],fmt='o',color='gray',markersize=12)

# R obs.
ax1 = ax[1].errorbar(index[0]+1.3,R_obs[1,0],yerr=conf_obs*error_R_obs[1,0],fmt='x',color='black',markersize=12,label='Obs. (1982-2023)')
ax1[-1][0].set_linestyle('--')
for i in np.arange(1,np.size(index)+1):
    if sig_R_obs[i,0] == 1:
        ax[1].plot(index[i-1]+1.3,R_obs[i,0],'ko',markersize=17,fillstyle='none')
    ax1 = ax[1].errorbar(index[i-1]+1.3,R_obs[i,0],yerr=conf_obs*error_R_obs[i,0],fmt='x',color='black',markersize=12)
    ax1[-1][0].set_linestyle('--')

# Labels and legend
ax[1].set_ylabel('Correlation coefficient $R$',fontsize=20)
ax[1].tick_params(axis='both',labelsize=16)
ax[1].set_xticks(np.arange(1,np.size(index)+1))
ax[1].set_xticklabels(label_names)
ax[1].grid(linestyle='--')
ax[1].set_ylim(-1,1)
ax[1].axes.axhline(y=0,color='k',linestyle='--')
ax[1].set_title('b',loc='left',fontsize=25,fontweight='bold')

# Save figure
if save_fig == True:
    if season == 'OND':
        filename = dir_fig + 'fig3.pdf'
    elif season == 'JAS':
        filename = dir_fig + 'fig5.pdf'
    fig.savefig(filename)