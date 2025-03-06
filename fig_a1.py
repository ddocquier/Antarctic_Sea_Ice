#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Figure A1: Plot correlation from models over whole period (1970-2099) for different seasons

Large ensembles: EC-Earth3 (SMHI-LENS), CESM2-LE, MPI-ESM1-2-LR, CanESM5, ACCESS-ESM1-5
Test with ACCESS-ESM1.5 over different seasons
Liang index computed via compute_liang_seasons.py

Target variable: Summer sea-ice extent (DJF)

Drivers:
Previous sea-ice extent
Previous Antarctic mean surface air temperature (<60S)
Previous Antarctic mean SST (<60S)
Previous Southern Annular Mode (SAM)
Previous Amundsen Sea Low (ASL)
Previous Niño3.4
Previous DMI

Last updated: 12/02/2025

@author: David Docquier
"""

# Standard libraries
import numpy as np
import matplotlib.pyplot as plt

# Parameters
model = 'ACCESS-ESM1-5'
nvar = 8 # number of variables (1: SSIE; 2: PSIE; 3: T_2m; 4: SST; 5: SAM; 6: ASL; 7: ENSO; 8: DMI)
conf = 2.57 # 1.96 if 95% confidence interval (normal distribution); 1.65 if 90% and 2.57 if 99%
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

# Load MAM
filename = dir_input + 'Liang_ACCESS-ESM1-5_MAM_1000boot.npy'
tau_mam,R_mam,error_tau_mam,error_R_mam,notused = np.load(filename,allow_pickle=True)

# Load AMJ
filename = dir_input + 'Liang_ACCESS-ESM1-5_AMJ_1000boot.npy'
tau_amj,R_amj,error_tau_amj,error_R_amj,notused = np.load(filename,allow_pickle=True)

# Load MJJ
filename = dir_input + 'Liang_ACCESS-ESM1-5_MJJ_1000boot.npy'
tau_mjj,R_mjj,error_tau_mjj,error_R_mjj,notused = np.load(filename,allow_pickle=True)

# Load JJA
filename = dir_input + 'Liang_ACCESS-ESM1-5_JJA_1000boot.npy'
tau_jja,R_jja,error_tau_jja,error_R_jja,notused = np.load(filename,allow_pickle=True)

# Load JAS
filename = dir_input + 'Liang_ACCESS-ESM1-5_JAS_1000boot.npy'
tau_jas,R_jas,error_tau_jas,error_R_jas = np.load(filename,allow_pickle=True)

# Load ASO
filename = dir_input + 'Liang_ACCESS-ESM1-5_ASO_1000boot.npy'
tau_aso,R_aso,error_tau_aso,error_R_aso,notused = np.load(filename,allow_pickle=True)

# Load SON
filename = dir_input + 'Liang_ACCESS-ESM1-5_SON_1000boot.npy'
tau_son,R_son,error_tau_son,error_R_son,notused = np.load(filename,allow_pickle=True)

# Load OND
filename = dir_input + 'Liang_ACCESS-ESM1-5_OND_1000boot.npy'
tau_ond,R_ond,error_tau_ond,error_R_ond = np.load(filename,allow_pickle=True)

# Compute p value of ACCESS-ESM1-5
sig_tau_mam = np.zeros((nvar,nvar))
sig_R_mam = np.zeros((nvar,nvar))
sig_tau_amj = np.zeros((nvar,nvar))
sig_R_amj = np.zeros((nvar,nvar))
sig_tau_mjj = np.zeros((nvar,nvar))
sig_R_mjj = np.zeros((nvar,nvar))
sig_tau_jja = np.zeros((nvar,nvar))
sig_R_jja = np.zeros((nvar,nvar))
sig_tau_jas = np.zeros((nvar,nvar))
sig_R_jas = np.zeros((nvar,nvar))
sig_tau_aso = np.zeros((nvar,nvar))
sig_R_aso = np.zeros((nvar,nvar))
sig_tau_son = np.zeros((nvar,nvar))
sig_R_son = np.zeros((nvar,nvar))
sig_tau_ond = np.zeros((nvar,nvar))
sig_R_ond = np.zeros((nvar,nvar))
for i in np.arange(nvar):
    for j in np.arange(nvar):
        sig_tau_mam[i,j] = compute_sig(tau_mam[i,j],error_tau_mam[i,j],conf)
        sig_R_mam[i,j] = compute_sig(R_mam[i,j],error_R_mam[i,j],conf)
        sig_tau_amj[i,j] = compute_sig(tau_amj[i,j],error_tau_amj[i,j],conf)
        sig_R_amj[i,j] = compute_sig(R_amj[i,j],error_R_amj[i,j],conf)
        sig_tau_mjj[i,j] = compute_sig(tau_mjj[i,j],error_tau_mjj[i,j],conf)
        sig_R_mjj[i,j] = compute_sig(R_mjj[i,j],error_R_mjj[i,j],conf)
        sig_tau_jja[i,j] = compute_sig(tau_jja[i,j],error_tau_jja[i,j],conf)
        sig_R_jja[i,j] = compute_sig(R_jja[i,j],error_R_jja[i,j],conf)
        sig_tau_jas[i,j] = compute_sig(tau_jas[i,j],error_tau_jas[i,j],conf)
        sig_R_jas[i,j] = compute_sig(R_jas[i,j],error_R_jas[i,j],conf)
        sig_tau_aso[i,j] = compute_sig(tau_aso[i,j],error_tau_aso[i,j],conf)
        sig_R_aso[i,j] = compute_sig(R_aso[i,j],error_R_aso[i,j],conf)
        sig_tau_son[i,j] = compute_sig(tau_son[i,j],error_tau_son[i,j],conf)
        sig_R_son[i,j] = compute_sig(R_son[i,j],error_R_son[i,j],conf)
        sig_tau_ond[i,j] = compute_sig(tau_ond[i,j],error_tau_ond[i,j],conf)
        sig_R_ond[i,j] = compute_sig(R_ond[i,j],error_R_ond[i,j],conf)

# Plot options
index = np.arange(nvar-1)
bar_width = 1

# Labels
label_names = ['PSIE','T$_{2m}$','SST','SAM','ASL','Niño3.4','DMI']

# Figure
fig,ax = plt.subplots(figsize=(9,4))
fig.subplots_adjust(left=0.15,bottom=0.12,right=0.94,top=0.93)
    
# R MAM
ax.errorbar(index[0]+0.7,R_mam[1,0],yerr=conf*error_R_mam[1,0],fmt='mo',markersize=12,label='MAM')
for i in np.arange(1,np.size(index)+1):
    if sig_R_mam[i,0] == 1:
        ax.plot(index[i-1]+0.7,R_mam[i,0],'ko',markersize=17)
    ax.errorbar(index[i-1]+0.7,R_mam[i,0],yerr=conf*error_R_mam[i,0],fmt='mo',markersize=12)
    
# R AMJ
ax.errorbar(index[0]+0.8,R_amj[1,0],yerr=conf*error_R_amj[1,0],fmt='go',markersize=12,label='AMJ')
for i in np.arange(1,np.size(index)+1):
    if sig_R_amj[i,0] == 1:
        ax.plot(index[i-1]+0.8,R_amj[i,0],'ko',markersize=17)
    ax.errorbar(index[i-1]+0.8,R_amj[i,0],yerr=conf*error_R_amj[i,0],fmt='go',markersize=12)

# R MJJ
ax.errorbar(index[0]+0.9,R_mjj[1,0],yerr=conf*error_R_mjj[1,0],fmt='bo',markersize=12,label='MJJ')
for i in np.arange(1,np.size(index)+1):
    if sig_R_mjj[i,0] == 1:
        ax.plot(index[i-1]+0.9,R_mjj[i,0],'ko',markersize=17)
    ax.errorbar(index[i-1]+0.9,R_mjj[i,0],yerr=conf*error_R_mjj[i,0],fmt='bo',markersize=12)

# R JJA
ax.errorbar(index[0]+1,R_jja[1,0],yerr=conf*error_R_jja[1,0],fmt='ro',markersize=12,label='JJA')
for i in np.arange(1,np.size(index)+1):
    if sig_R_jja[i,0] == 1:
        ax.plot(index[i-1]+1,R_jja[i,0],'ko',markersize=17)
    ax.errorbar(index[i-1]+1,R_jja[i,0],yerr=conf*error_R_jja[i,0],fmt='ro',markersize=12)

# R JAS
ax.errorbar(index[0]+1.1,R_jas[1,0],yerr=conf*error_R_jas[1,0],fmt='o',color='gray',markersize=12,label='JAS')
for i in np.arange(1,np.size(index)+1):
    if sig_R_jas[i,0] == 1:
        ax.plot(index[i-1]+1.1,R_jas[i,0],'ko',markersize=17)
    ax.errorbar(index[i-1]+1.1,R_jas[i,0],yerr=conf*error_R_jas[i,0],fmt='o',color='gray',markersize=12)

# R ASO
ax.errorbar(index[0]+1.2,R_aso[1,0],yerr=conf*error_R_aso[1,0],fmt='yo',markersize=12,label='ASO')
for i in np.arange(1,np.size(index)+1):
    if sig_R_aso[i,0] == 1:
        ax.plot(index[i-1]+1.2,R_aso[i,0],'ko',markersize=17)
    ax.errorbar(index[i-1]+1.2,R_aso[i,0],yerr=conf*error_R_aso[i,0],fmt='yo',markersize=12)
    
# R SON
ax.errorbar(index[0]+1.3,R_son[1,0],yerr=conf*error_R_son[1,0],fmt='co',markersize=12,label='SON')
for i in np.arange(1,np.size(index)+1):
    if sig_R_son[i,0] == 1:
        ax.plot(index[i-1]+1.3,R_son[i,0],'ko',markersize=17)
    ax.errorbar(index[i-1]+1.3,R_son[i,0],yerr=conf*error_R_son[i,0],fmt='co',markersize=12)

# R OND
ax.errorbar(index[0]+1.4,R_ond[1,0],yerr=conf*error_R_ond[1,0],fmt='o',color='orange',markersize=12,label='OND')
for i in np.arange(1,np.size(index)+1):
    if sig_R_ond[i,0] == 1:
        ax.plot(index[i-1]+1.4,R_ond[i,0],'ko',markersize=17)
    ax.errorbar(index[i-1]+1.4,R_ond[i,0],yerr=conf*error_R_ond[i,0],fmt='o',color='orange',markersize=12)

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
    filename = dir_fig + 'fig_a1.pdf'
    fig.savefig(filename)