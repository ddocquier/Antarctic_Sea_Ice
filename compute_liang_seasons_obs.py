#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute Liang index over whole period (1982-2023) - Observations and Reanalyses

Variables:
Summer sea-ice extent (DJF)
Previous sea-ice extent
Previous Antarctic mean surface air temperature (<60S)
Previous Antarctic mean SST (<60S)
Previous Southern Annular Mode (SAM)
Previous Amundsen Sea Low (ASL)
Previous Niño3.4
Previous DMI

Last updated: 05/03/2025

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
sys.path.append('/home/dadocq/Documents/Codes/Liang/')
from function_liang_nvar import compute_liang_nvar

# Parameters
season = 'JAS' # MAM, AMJ, MJJ, JJA, JAS, ASO, SON, OND / default: OND (spring) or JAS (winter)
nvar = 8 # number of variables (1: SSIE; 2: PSIE; 3: T_2m; 4: SST; 5: SAM; 6: ASL; 7: ENSO; 8: DMI)
dt = 1 # time step (years)
n_iter = 1000 # number of bootstrap realizations
conf = 2.57 # 1.96 if 95% confidence interval (normal distribution); 1.65 if 90% and 2.57 if 99%
save_var = True
save_fig = True

# Working directories
dir_input = '/home/dadocq/Documents/Papers/My_Papers/RESIST_Antarctic/output/seasons/'
dir_fig = '/home/dadocq/Documents/Papers/My_Papers/RESIST_Antarctic/figures/seasons/'

# Function to test significance (based on the confidence interval)
def compute_sig(var,error,conf):
    if np.abs(var)-conf*error > 0. and np.abs(var)+conf*error > 0.:
        sig = 1
    else:
        sig = 0
    return sig

# Load variables (saved via save_timeseries_obs.py)
filename = dir_input + 'Obs_Antarctic_timeseries_' + season + '.npy'
sie_summer,sie_previous,tas,sst,sam,asl,nino34,dmi = np.load(filename,allow_pickle=True)

# Take same years for all variables - 1982-2023
sie_summer = sie_summer[4::] # DJF 1982-1983 - DJF 2023-2024
sie_previous = sie_previous[3::] # JAS/OND 1982 - JAS/OND 2023
tas = tas[12::]
sam = sam[12::]
asl = asl[12::]
nino34 = nino34[12::]
dmi = dmi[12::]

# Remove linear trend
t = np.linspace(0,np.size(sie_summer)-1,np.size(sie_summer))
sie_summer2 = sie_summer - (linregress(t,sie_summer).intercept + linregress(t,sie_summer).slope * t)
sie_previous2 = sie_previous - (linregress(t,sie_previous).intercept + linregress(t,sie_previous).slope * t)
tas2 = tas - (linregress(t,tas).intercept + linregress(t,tas).slope * t)
sam2 = sam - (linregress(t,sam).intercept + linregress(t,sam).slope * t)
sst2 = sst - (linregress(t,sst).intercept + linregress(t,sst).slope * t)
asl2 = asl - (linregress(t,asl).intercept + linregress(t,asl).slope * t)
nino2 = nino34 - (linregress(t,nino34).intercept + linregress(t,nino34).slope * t)
dmi2 = dmi - (linregress(t,dmi).intercept + linregress(t,dmi).slope * t)

# Compute relative transfer of information (tau) and correlation coefficient (R) and save in files
filename = dir_input + 'Liang_obs_' + season + '.npy'
if save_var == True:
    xx = np.array((sie_summer2,sie_previous2,tas2,sst2,sam2,asl2,nino2,dmi2))
    notused,tau,R,notused,error_tau,error_R = compute_liang_nvar(xx,dt,n_iter)
    np.save(filename,[tau,R,error_tau,error_R])
else:
    tau,R,error_tau,error_R = np.load(filename,allow_pickle=True)

# Compute significance (different from 0) based on the confidence interval
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
tau_plot.set_xlabel('TO...',loc='left',fontsize=20)
tau_plot.xaxis.set_label_position('top')
tau_plot.set_yticklabels(tau_plot.get_ymajorticklabels(),fontsize=20)
tau_plot.set_ylabel('FROM...',loc='top',fontsize=20)
                   
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
    filename = dir_fig + 'Liang_obs_' + season + '.png'
    fig.savefig(filename)