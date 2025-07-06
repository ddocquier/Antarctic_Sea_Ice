#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 6: Plot Liang index and correlation from models over time

Large ensembles: EC-Earth3 (SMHI-LENS), CESM2-LE, MPI-ESM1-2-LR, CanESM5, ACCESS-ESM1-5

Liang index computed via compute_liang_ts.py

Variables:
Summer sea-ice extent (DJF)
Previous sea-ice extent
Previous Antarctic mean surface air temperature (<60S)
Previous Antarctic mean SST (<60S)
Previous Southern Annular Mode (SAM)
Previous Amundsen Sea Low (ASL)
Previous Niño3.4
Previous DMI

Last updated: 01/07/2025

@author: David Docquier
"""

# Standard libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# Parameters
range_years = 10 # range of years (default: 10 years)
season = 'OND' # MAM, AMJ, MJJ, JJA, JAS, ASO, SON, OND / default: OND (spring) or JAS (winter)
nvar = 8 # number of variables (1: SSIE; 2: PSIE; 3: T_2m; 4: SST; 5: SAM; 6: ASL; 7: ENSO; 8: DMI)
conf = 2.57 # 1.96 if 95% confidence interval (normal distribution); 1.65 if 90% and 2.57 if 99%
n_iter = 1000 # number of bootstrap realizations; default: 1000
alpha_slope = 0.1 # statistical significance threshold for slope
save_fig = True

# Function to test significance (based on the confidence interval)
def compute_sig(var,error,conf):
    if np.abs(var)-conf*error > 0. and np.abs(var)+conf*error > 0.:
        sig = 1
    else:
        sig = 0
    return sig

# Number of members per model
n_members_ecearth = 50
n_members_cesm = 50
n_members_mpi = 50
n_members_canesm = 50
n_members_access = 40

# Working directories
dir_input = '/home/ddocquier/Documents/Papers/My_Papers/RESIST_Antarctic/output/seasons/'
dir_fig = '/home/ddocquier/Documents/Papers/My_Papers/RESIST_Antarctic/LaTeX/'

# Load tau SMHI-LENS
filename = dir_input + 'Liang_SMHI-LENS_' + season + '_' + str(n_iter) + 'boot_ts.npy'
tau_ecearth,R_ecearth,error_tau_ecearth,error_R_ecearth = np.load(filename,allow_pickle=True)

# Load tau CESM2-LE
filename = dir_input + 'Liang_CESM2-LE_' + season + '_' + str(n_iter) + 'boot_ts.npy'
tau_cesm,R_cesm,error_tau_cesm,error_R_cesm = np.load(filename,allow_pickle=True)

# Load tau MPI-ESM1-2-LR
filename = dir_input + 'Liang_MPI-ESM1-2-LR_' + season + '_' + str(n_iter) + 'boot_ts.npy'
tau_mpi,R_mpi,error_tau_mpi,error_R_mpi = np.load(filename,allow_pickle=True)

# Load tau CanESM5
filename = dir_input + 'Liang_CanESM5_' + season + '_' + str(n_iter) + 'boot_ts.npy'
tau_canesm,R_canesm,error_tau_canesm,error_R_canesm = np.load(filename,allow_pickle=True)

# Load tau ACCESS-ESM1-5
filename = dir_input + 'Liang_ACCESS-ESM1-5_' + season + '_' + str(n_iter) + 'boot_ts.npy'
tau_access,R_access,error_tau_access,error_R_access = np.load(filename,allow_pickle=True)

# Plot options
nt = np.size(tau_ecearth,0)

# Labels
xrange = np.arange(1,nt+1,1)
index = np.arange(nt)
name_xticks = ['1970','','1990','','2010','','2030','','2050','','2070','','2090']

# Compute trends
tau_ecearth_trend = np.zeros((nt,nvar))
tau_cesm_trend = np.zeros((nt,nvar))
tau_mpi_trend = np.zeros((nt,nvar))
tau_canesm_trend = np.zeros((nt,nvar))
tau_access_trend = np.zeros((nt,nvar))
tau_ecearth_pval = np.zeros(nvar)
tau_cesm_pval = np.zeros(nvar)
tau_mpi_pval = np.zeros(nvar)
tau_canesm_pval = np.zeros(nvar)
tau_access_pval = np.zeros(nvar)
for i in np.arange(nvar):
    tau_ecearth_trend[:,i] = linregress(xrange,np.abs(tau_ecearth[0:nt,i,0])).intercept + linregress(xrange,np.abs(tau_ecearth[0:nt,i,0])).slope * xrange
    tau_cesm_trend[:,i] = linregress(xrange,np.abs(tau_cesm[0:nt,i,0])).intercept + linregress(xrange,np.abs(tau_cesm[0:nt,i,0])).slope * xrange
    tau_mpi_trend[:,i] = linregress(xrange,np.abs(tau_mpi[0:nt,i,0])).intercept + linregress(xrange,np.abs(tau_mpi[0:nt,i,0])).slope * xrange
    tau_canesm_trend[:,i] = linregress(xrange,np.abs(tau_canesm[0:nt,i,0])).intercept + linregress(xrange,np.abs(tau_canesm[0:nt,i,0])).slope * xrange
    tau_access_trend[:,i] = linregress(xrange,np.abs(tau_access[0:nt,i,0])).intercept + linregress(xrange,np.abs(tau_access[0:nt,i,0])).slope * xrange
    tau_ecearth_pval[i] = linregress(xrange,np.abs(tau_ecearth[0:nt,i,0])).pvalue
    tau_cesm_pval[i] = linregress(xrange,np.abs(tau_cesm[0:nt,i,0])).pvalue
    tau_mpi_pval[i] = linregress(xrange,np.abs(tau_mpi[0:nt,i,0])).pvalue
    tau_canesm_pval[i] = linregress(xrange,np.abs(tau_canesm[0:nt,i,0])).pvalue
    tau_access_pval[i] = linregress(xrange,np.abs(tau_access[0:nt,i,0])).pvalue

# Compute statistical significance
sig_tau_ecearth = np.zeros((nt,nvar,nvar))
sig_R_ecearth = np.zeros((nt,nvar,nvar))
sig_tau_cesm = np.zeros((nt,nvar,nvar))
sig_R_cesm = np.zeros((nt,nvar,nvar))
sig_tau_mpi = np.zeros((nt,nvar,nvar))
sig_R_mpi = np.zeros((nt,nvar,nvar))
sig_tau_canesm = np.zeros((nt,nvar,nvar))
sig_R_canesm = np.zeros((nt,nvar,nvar))
sig_tau_access = np.zeros((nt,nvar,nvar))
sig_R_access = np.zeros((nt,nvar,nvar))
for t in np.arange(nt):
    for i in np.arange(nvar):
        for j in np.arange(nvar):
            sig_tau_ecearth[t,i,j] = compute_sig(tau_ecearth[t,i,j],error_tau_ecearth[t,i,j],conf)
            sig_R_ecearth[t,i,j] = compute_sig(R_ecearth[t,i,j],error_R_ecearth[t,i,j],conf)
            sig_tau_cesm[t,i,j] = compute_sig(tau_cesm[t,i,j],error_tau_cesm[t,i,j],conf)
            sig_R_cesm[t,i,j] = compute_sig(R_cesm[t,i,j],error_R_cesm[t,i,j],conf)
            sig_tau_mpi[t,i,j] = compute_sig(tau_mpi[t,i,j],error_tau_mpi[t,i,j],conf)
            sig_R_mpi[t,i,j] = compute_sig(R_mpi[t,i,j],error_R_mpi[t,i,j],conf)
            sig_tau_canesm[t,i,j] = compute_sig(tau_canesm[t,i,j],error_tau_canesm[t,i,j],conf)
            sig_R_canesm[t,i,j] = compute_sig(R_canesm[t,i,j],error_R_canesm[t,i,j],conf)
            sig_tau_access[t,i,j] = compute_sig(tau_access[t,i,j],error_tau_access[t,i,j],conf)
            sig_R_access[t,i,j] = compute_sig(R_access[t,i,j],error_R_access[t,i,j],conf)

# Figure
fig,ax = plt.subplots(2,2,figsize=(20,10))
fig.subplots_adjust(left=0.05,bottom=0.05,right=0.95,top=0.93,hspace=0.4,wspace=0.18)

####################
# tau previous SIE #
####################

var = 1

# ACCESS
for i in np.arange(nt):
    if sig_tau_access[i,var,0] == 1:
        ax[0,0].plot(index[i]+1,np.abs(tau_access[i,var,0]),'ko',markersize=15)
ax[0,0].plot(index+1,np.abs(tau_access[:,var,0]),'o-',color='orange',markersize=10,label='ACCESS-ESM1.5')
if tau_access_pval[var] < alpha_slope:
    ax[0,0].plot(xrange,tau_access_trend[:,var],'--',color='orange')
    
# CanESM5
for i in np.arange(nt):
    if sig_tau_canesm[i,var,0] == 1:
        ax[0,0].plot(index[i]+1,np.abs(tau_canesm[i,var,0]),'ko',markersize=15)
ax[0,0].plot(index+1,np.abs(tau_canesm[:,var,0]),'go-',markersize=10,label='CanESM5')
if tau_canesm_pval[var] < alpha_slope:
    ax[0,0].plot(xrange,tau_canesm_trend[:,var],'g--') 
    
# CESM2
for i in np.arange(nt):
    if sig_tau_cesm[i,var,0] == 1:
        ax[0,0].plot(index[i]+1,np.abs(tau_cesm[i,var,0]),'ko',markersize=15)
ax[0,0].plot(index+1,np.abs(tau_cesm[:,var,0]),'bo-',markersize=10,label='CESM2')
if tau_cesm_pval[var] < alpha_slope:
    ax[0,0].plot(xrange,tau_cesm_trend[:,var],'b--')
    
# EC-Earth3
for i in np.arange(nt):
    if sig_tau_ecearth[i,var,0] == 1:
        ax[0,0].plot(index[i]+1,np.abs(tau_ecearth[i,var,0]),'ko',markersize=15)
ax[0,0].plot(index+1,np.abs(tau_ecearth[:,var,0]),'ro-',markersize=10,label='EC-Earth3')
if tau_ecearth_pval[var] < alpha_slope:
    ax[0,0].plot(xrange,tau_ecearth_trend[:,var],'r--')

# MPI
for i in np.arange(nt):
    if sig_tau_mpi[i,var,0] == 1:
        ax[0,0].plot(index[i]+1,np.abs(tau_mpi[i,var,0]),'ko',markersize=15)
ax[0,0].plot(index+1,np.abs(tau_mpi[:,var,0]),'o-',color='gray',markersize=10,label='MPI-ESM1.2-LR')
if tau_mpi_pval[var] < alpha_slope:
    ax[0,0].plot(xrange,tau_mpi_trend[:,var],'-',color='gray')

# Labels and legend
ax[0,0].set_ylabel(r'Information transfer $\|\tau\|$ ($\%$)',fontsize=20)
ax[0,0].tick_params(axis='both',labelsize=18)
ax[0,0].set_xticks(xrange)
ax[0,0].set_xticklabels(name_xticks)
ax[0,0].grid(linestyle='--')
ax[0,0].set_ylim(0,25)
ax[0,0].set_title('a',loc='left',fontsize=25,fontweight='bold')
ax[0,0].set_title('Previous SIE $\longrightarrow$ Summer SIE',loc='center',fontsize=22)
    
    
############
# tau T_2m #
############
    
var = 2
    
# ACCESS
for i in np.arange(nt):
    if sig_tau_access[i,var,0] == 1:
        ax[0,1].plot(index[i]+1,np.abs(tau_access[i,var,0]),'ko',markersize=15)
ax[0,1].plot(index+1,np.abs(tau_access[:,var,0]),'o-',color='orange',markersize=10,label='ACCESS-ESM1.5')
if tau_access_pval[var] < alpha_slope:
    ax[0,1].plot(xrange,tau_access_trend[:,var],'--',color='orange')
    
# CanESM5
for i in np.arange(nt):
    if sig_tau_canesm[i,var,0] == 1:
        ax[0,1].plot(index[i]+1,np.abs(tau_canesm[i,var,0]),'ko',markersize=15)
ax[0,1].plot(index+1,np.abs(tau_canesm[:,var,0]),'go-',markersize=10,label='CanESM5')
if tau_canesm_pval[var] < alpha_slope:
    ax[0,1].plot(xrange,tau_canesm_trend[:,var],'g--') 
    
# CESM2
for i in np.arange(nt):
    if sig_tau_cesm[i,var,0] == 1:
        ax[0,1].plot(index[i]+1,np.abs(tau_cesm[i,var,0]),'ko',markersize=15)
ax[0,1].plot(index+1,np.abs(tau_cesm[:,var,0]),'bo-',markersize=10,label='CESM2')
if tau_cesm_pval[var] < alpha_slope:
    ax[0,1].plot(xrange,tau_cesm_trend[:,var],'b--')
    
# EC-Earth3
for i in np.arange(nt):
    if sig_tau_ecearth[i,var,0] == 1:
        ax[0,1].plot(index[i]+1,np.abs(tau_ecearth[i,var,0]),'ko',markersize=15)
ax[0,1].plot(index+1,np.abs(tau_ecearth[:,var,0]),'ro-',markersize=10,label='EC-Earth3')
if tau_ecearth_pval[var] < alpha_slope:
    ax[0,1].plot(xrange,tau_ecearth_trend[:,var],'r--')

# MPI
for i in np.arange(nt):
    if sig_tau_mpi[i,var,0] == 1:
        ax[0,1].plot(index[i]+1,np.abs(tau_mpi[i,var,0]),'ko',markersize=15)
ax[0,1].plot(index+1,np.abs(tau_mpi[:,var,0]),'o-',color='gray',markersize=10,label='MPI-ESM1.2-LR')
if tau_mpi_pval[var] < alpha_slope:
    ax[0,1].plot(xrange,tau_mpi_trend[:,var],'-',color='gray')

# Labels and legend
ax[0,1].set_ylabel(r'Information transfer $\|\tau\|$ ($\%$)',fontsize=20)
ax[0,1].tick_params(axis='both',labelsize=18)
ax[0,1].set_xticks(xrange)
ax[0,1].set_xticklabels(name_xticks)
ax[0,1].grid(linestyle='--')
ax[0,1].set_ylim(0,25)
ax[0,1].set_title('b',loc='left',fontsize=25,fontweight='bold')
ax[0,1].set_title('T$_{2m}$ $\longrightarrow$ Summer SIE',loc='center',fontsize=22)


###############
# tau SST #
###############

var = 3
    
# ACCESS
for i in np.arange(nt):
    if sig_tau_access[i,var,0] == 1:
        ax[1,0].plot(index[i]+1,np.abs(tau_access[i,var,0]),'ko',markersize=15)
ax[1,0].plot(index+1,np.abs(tau_access[:,var,0]),'o-',color='orange',markersize=10,label='ACCESS-ESM1.5')
if tau_access_pval[var] < alpha_slope:
    ax[1,0].plot(xrange,tau_access_trend[:,var],'--',color='orange')
    
# CanESM5
for i in np.arange(nt):
    if sig_tau_canesm[i,var,0] == 1:
        ax[1,0].plot(index[i]+1,np.abs(tau_canesm[i,var,0]),'ko',markersize=15)
ax[1,0].plot(index+1,np.abs(tau_canesm[:,var,0]),'go-',markersize=10,label='CanESM5')
if tau_canesm_pval[var] < alpha_slope:
    ax[1,0].plot(xrange,tau_canesm_trend[:,var],'g--') 
    
# CESM2
for i in np.arange(nt):
    if sig_tau_cesm[i,var,0] == 1:
        ax[1,0].plot(index[i]+1,np.abs(tau_cesm[i,var,0]),'ko',markersize=15)
ax[1,0].plot(index+1,np.abs(tau_cesm[:,var,0]),'bo-',markersize=10,label='CESM2')
if tau_cesm_pval[var] < alpha_slope:
    ax[1,0].plot(xrange,tau_cesm_trend[:,var],'b--')
    
# EC-Earth3
for i in np.arange(nt):
    if sig_tau_ecearth[i,var,0] == 1:
        ax[1,0].plot(index[i]+1,np.abs(tau_ecearth[i,var,0]),'ko',markersize=15)
ax[1,0].plot(index+1,np.abs(tau_ecearth[:,var,0]),'ro-',markersize=10,label='EC-Earth3')
if tau_ecearth_pval[var] < alpha_slope:
    ax[1,0].plot(xrange,tau_ecearth_trend[:,var],'r--')

# MPI
for i in np.arange(nt):
    if sig_tau_mpi[i,var,0] == 1:
        ax[1,0].plot(index[i]+1,np.abs(tau_mpi[i,var,0]),'ko',markersize=15)
ax[1,0].plot(index+1,np.abs(tau_mpi[:,var,0]),'o-',color='gray',markersize=10,label='MPI-ESM1.2-LR')
if tau_mpi_pval[var] < alpha_slope:
    ax[1,0].plot(xrange,tau_mpi_trend[:,var],'-',color='gray')


# Labels and legend
ax[1,0].set_ylabel(r'Information transfer $\|\tau\|$ ($\%$)',fontsize=20)
ax[1,0].tick_params(axis='both',labelsize=18)
ax[1,0].set_xticks(xrange)
ax[1,0].set_xticklabels(name_xticks)
ax[1,0].grid(linestyle='--')
ax[1,0].set_ylim(0,25)
ax[1,0].set_title('c',loc='left',fontsize=25,fontweight='bold')
ax[1,0].set_title('SST $\longrightarrow$ Summer SIE',loc='center',fontsize=22)


##################
# tau SAM #
##################

var = 4
        
# ACCESS
for i in np.arange(nt):
    if sig_tau_access[i,var,0] == 1:
        ax[1,1].plot(index[i]+1,np.abs(tau_access[i,var,0]),'ko',markersize=15)
ax[1,1].plot(index+1,np.abs(tau_access[:,var,0]),'o-',color='orange',markersize=10,label='ACCESS-ESM1.5')
if tau_access_pval[var] < alpha_slope:
    ax[1,1].plot(xrange,tau_access_trend[:,var],'--',color='orange')
    
# CanESM5
for i in np.arange(nt):
    if sig_tau_canesm[i,var,0] == 1:
        ax[1,1].plot(index[i]+1,np.abs(tau_canesm[i,var,0]),'ko',markersize=15)
ax[1,1].plot(index+1,np.abs(tau_canesm[:,var,0]),'go-',markersize=10,label='CanESM5')
if tau_canesm_pval[var] < alpha_slope:
    ax[1,1].plot(xrange,tau_canesm_trend[:,var],'g--') 
    
# CESM2
for i in np.arange(nt):
    if sig_tau_cesm[i,var,0] == 1:
        ax[1,1].plot(index[i]+1,np.abs(tau_cesm[i,var,0]),'ko',markersize=15)
ax[1,1].plot(index+1,np.abs(tau_cesm[:,var,0]),'bo-',markersize=10,label='CESM2')
if tau_cesm_pval[var] < alpha_slope:
    ax[1,1].plot(xrange,tau_cesm_trend[:,var],'b--')
    
# EC-Earth3
for i in np.arange(nt):
    if sig_tau_ecearth[i,var,0] == 1:
        ax[1,1].plot(index[i]+1,np.abs(tau_ecearth[i,var,0]),'ko',markersize=15)
ax[1,1].plot(index+1,np.abs(tau_ecearth[:,var,0]),'ro-',markersize=10,label='EC-Earth3')
if tau_ecearth_pval[var] < alpha_slope:
    ax[1,1].plot(xrange,tau_ecearth_trend[:,var],'r--')

# MPI
for i in np.arange(nt):
    if sig_tau_mpi[i,var,0] == 1:
        ax[1,1].plot(index[i]+1,np.abs(tau_mpi[i,var,0]),'ko',markersize=15)
ax[1,1].plot(index+1,np.abs(tau_mpi[:,var,0]),'o-',color='gray',markersize=10,label='MPI-ESM1.2-LR')
if tau_mpi_pval[var] < alpha_slope:
    ax[1,1].plot(xrange,tau_mpi_trend[:,var],'-',color='gray')

# Labels and legend
ax[1,1].set_ylabel(r'Information transfer $\|\tau\|$ ($\%$)',fontsize=20)
ax[1,1].tick_params(axis='both',labelsize=18)
ax[1,1].set_xticks(xrange)
ax[1,1].set_xticklabels(name_xticks)
ax[1,1].grid(linestyle='--')
ax[1,1].set_ylim(0,25)
ax[1,1].set_title('d',loc='left',fontsize=25,fontweight='bold')
ax[1,1].set_title('SAM $\longrightarrow$ Summer SIE',loc='center',fontsize=22)
ax[1,1].legend(loc='upper right',fontsize=18,shadow=True,frameon=False)
    

# Save figure
if save_fig == True:
    filename = dir_fig + 'fig6.pdf'
    fig.savefig(filename)