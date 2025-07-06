#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute Liang index over whole period (1982-2023) for different Antarctic sectors - Observations and Reanalyses

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

Last updated: 23/06/2025

@author: David Docquier
"""

# Import libraries
import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy.stats import linregress
import seaborn as sns # for creating a matrix plot
from matplotlib.patches import Rectangle # for drawing rectangles around elements in a matrix

# Import my functions
sys.path.append('/home/ddocquier/Documents/Codes/Liang/')
from function_liang_nvar import compute_liang_nvar

# Parameters
season = 'SON' # MAM, AMJ, MJJ, JJA, JAS, ASO, SON, OND / default: OND (spring) or JAS (winter)
sector = 'rs' # bas (Bellingshausen-Amundsen Seas), ws (Weddell Sea), io (Indian Ocean), wpo (Western Pacific Ocean), rs (Ross Sea)
nvar = 8 # number of variables (1: SSIE; 2: PSIE; 3: T_2m; 4: SST; 5: SAM; 6: ASL; 7: ENSO; 8: DMI)
dt = 1 # time step (years)
n_iter = 1000 # number of bootstrap realizations
conf = 2.57 # 1.96 if 95% confidence interval (normal distribution); 1.65 if 90% and 2.57 if 99%
save_var = True
save_fig = False

# Working directories
dir_input = '/home/ddocquier/Documents/Papers/My_Papers/RESIST_Antarctic/output/seasons/'
dir_fig = '/home/ddocquier/Documents/Papers/My_Papers/RESIST_Antarctic/figures/seasons/sectors/'

# Function to test significance (based on the confidence interval)
def compute_sig(var,error,conf):
    if (var-conf*error < 0. and var+conf*error < 0.) or (var-conf*error > 0. and var+conf*error > 0.):
        sig = 1
    else:
        sig = 0
    return sig

# Load variables
filename = dir_input + 'Obs_Antarctic_timeseries_' + season + '.npy'
notused,notused,notused,notused,sam,asl,nino,dmi = np.load(filename,allow_pickle=True)

# Load SSIE for sectors
filename = dir_input + 'SSIE_obs_timeseries.npy'
ssie_bas,ssie_ws,ssie_io,ssie_wpo,ssie_rs = np.load(filename,allow_pickle=True)

# Load PSIE for sectors
filename = dir_input + 'SIE_Obs_timeseries_' + season + '.npy'
psie_bas,psie_ws,psie_io,psie_wpo,psie_rs = np.load(filename,allow_pickle=True)

# Load T2m for sectors
filename = dir_input + 'tas_mon_Obs_timeseries_' + season + '.npy'
tas_bas,tas_ws,tas_io,tas_wpo,tas_rs = np.load(filename,allow_pickle=True)

# Load SST for sectors
filename = dir_input + 'sst_mon_Obs_timeseries_' + season + '.npy'
sst_bas,sst_ws,sst_io,sst_wpo,sst_rs = np.load(filename,allow_pickle=True)

# Keep sector of interest for SSIE, WSIE, WSIV, SST and T2m
if sector == 'bas':
    ssie = ssie_bas
    psie = psie_bas
    sst = sst_bas
    tas = tas_bas
elif sector == 'ws':
    ssie = ssie_ws
    psie = psie_ws
    sst = sst_ws
    tas = tas_ws
elif sector == 'io':
    ssie = ssie_io
    psie = psie_io
    sst = sst_io
    tas = tas_io
elif sector == 'wpo':
    ssie = ssie_wpo
    psie = psie_wpo
    sst = sst_wpo
    tas = tas_wpo
elif sector == 'rs':
    ssie = ssie_rs
    psie = psie_rs
    sst = sst_rs
    tas = tas_rs
    
# Take same years for all variables - 1982-2023
ssie = ssie[4::] # DJF 1982-1983 - DJF 2023-2024
psie = psie[3::] # JAS/OND 1982 - JAS/OND 2023
sam = sam[12::]
asl = asl[12::]
nino = nino[12::]
dmi = dmi[12::]

# Remove linear trend
t = np.linspace(0,np.size(ssie)-1,np.size(ssie))
ssie2 = ssie - (linregress(t,ssie).intercept + linregress(t,ssie).slope * t)
psie2 = psie - (linregress(t,psie).intercept + linregress(t,psie).slope * t)
tas2 = tas - (linregress(t,tas).intercept + linregress(t,tas).slope * t)
sam2 = sam - (linregress(t,sam).intercept + linregress(t,sam).slope * t)
sst2 = sst - (linregress(t,sst).intercept + linregress(t,sst).slope * t)
asl2 = asl - (linregress(t,asl).intercept + linregress(t,asl).slope * t)
nino2 = nino - (linregress(t,nino).intercept + linregress(t,nino).slope * t)
dmi2 = dmi - (linregress(t,dmi).intercept + linregress(t,dmi).slope * t)

# Compute relative transfer of information (tau) and correlation coefficient (R) and save in files
filename = dir_input + 'Liang_' + sector + '_obs_' + season + '.npy'
if save_var == True:
    xx = np.array((ssie2,psie2,tas2,sst2,sam2,asl2,nino2,dmi2))
    notused,tau,R,notused,error_tau,error_R = compute_liang_nvar(xx,dt,n_iter)
    np.save(filename,[tau,R,error_tau,error_R])
else:
    tau,R,error_tau,error_R = np.load(filename,allow_pickle=True)

# Compute statistical significance based on the confidence interval
sig_tau = np.zeros((nvar,nvar))
sig_R = np.zeros((nvar,nvar))
for i in np.arange(nvar):
    for j in np.arange(nvar):
        sig_tau[i,j] = compute_sig(tau[i,j],error_tau[i,j],conf)
        sig_R[i,j] = compute_sig(R[i,j],error_R[i,j],conf)

# Labels
label_names = ['SSIE','PSIE','T$_{2m}$','SST','SAM','ASL','N3.4','DMI']

# Plot options
fig,ax = plt.subplots(1,2,figsize=(24,13))
fig.subplots_adjust(left=0.05,bottom=0.01,right=0.95,top=0.85,wspace=0.15,hspace=0.15)
cmap_tau = plt.cm.YlOrRd._resample(15)
cmap_R = plt.cm.bwr._resample(16)
sns.set(font_scale=1.8)

# Matrix of tau
tau_annotations = np.round(np.abs(tau),2)
tau_plot = sns.heatmap(np.abs(tau),annot=tau_annotations,fmt='',annot_kws={'color':'k','fontsize':18},cmap=cmap_tau,
    cbar_kws={'label':r'$\|\tau\|$ ($\%$)','orientation':'horizontal','pad':0.05},vmin=0,vmax=70,
    xticklabels=label_names,yticklabels=label_names,linewidths=0.1,linecolor='gray',ax=ax[0])
tau_plot.set_title(r'Rate of information transfer $\|\tau\|$' + '\n',fontsize=24)
tau_plot.set_title('a \n',loc='left',fontsize=32,fontweight='bold')
for j in np.arange(nvar):
    for i in np.arange(nvar):
        if sig_tau[j,i] == 1:
            tau_plot.add_patch(Rectangle((i+0.05,j+0.2),0.9,0.6,fill=False,edgecolor='blue',linewidth=3))
tau_plot.set_xticklabels(tau_plot.get_xmajorticklabels(),fontsize=20)
tau_plot.xaxis.set_ticks_position('top')
# tau_plot.set_xlabel('TO...',loc='left',fontsize=20)
tau_plot.xaxis.set_label_position('top')
tau_plot.set_yticklabels(tau_plot.get_ymajorticklabels(),fontsize=20)
# tau_plot.set_ylabel('FROM...',loc='top',fontsize=20)
                   
# Matrix of R
R_annotations = np.round(R,2)
R_plot = sns.heatmap(R,annot=R_annotations,fmt='',annot_kws={'color':'k','fontsize':18},cmap=cmap_R,
    cbar_kws={'label':'$R$','orientation':'horizontal','pad':0.05},vmin=-1,vmax=1,
    xticklabels=label_names,yticklabels=label_names,linewidths=0.1,linecolor='gray',ax=ax[1])
R_plot.set_title('Correlation coefficient $R$ \n ',fontsize=24)
R_plot.set_title('b \n',loc='left',fontsize=32,fontweight='bold')
for j in np.arange(nvar):
    for i in np.arange(nvar):
        if sig_R[j,i] == 1 and j != i: 
               R_plot.add_patch(Rectangle((i+0.05,j+0.2),0.9,0.6,fill=False,edgecolor='black',linewidth=3))
R_plot.set_xticklabels(R_plot.get_xmajorticklabels(),fontsize=20)
R_plot.xaxis.set_ticks_position('top')
R_plot.set_yticklabels(R_plot.get_ymajorticklabels(),fontsize=20)

# Save figure
if save_fig == True:
    filename = dir_fig + 'Liang_' + sector + '_obs_' + season + '_' + str(n_iter) + 'boot.png'
    fig.savefig(filename)