#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute Liang index over whole period (1970-2099) and over all members for different Antarctic sectors

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

Last updated: 15/01/2025

@author: David Docquier
"""

# Import libraries
import numpy as np
import sys
import matplotlib.pyplot as plt
import seaborn as sns # for creating a matrix plot
from matplotlib.patches import Rectangle # for drawing rectangles around elements in a matrix

# Import my functions
sys.path.append('/home/dadocq/Documents/Codes/Liang/')
from function_liang_nvar_dx import compute_liang_nvar

# Parameters
season = 'OND' # MAM, AMJ, MJJ, JJA, JAS, ASO, SON, OND / default: OND (spring) or JAS (winter)
sector = 'bas' # bas (Bellingshausen-Amundsen Seas), ws (Weddell Sea), io (Indian Ocean), wpo (Western Pacific Ocean), rs (Ross Sea)
model = 'SMHI-LENS' # SMHI-LENS; CESM2-LE; MPI-ESM1-2-LR; CanESM5; ACCESS-ESM1-5
if model == 'SMHI-LENS':
    string_model = 'EC-Earth3'
elif model == 'CESM2-LE':
    string_model = 'CESM2'
else:
    string_model = model
nvar = 8 # number of variables (1: SSIE; 2: PSIE; 3: T_2m; 4: SST; 5: SAM; 6: ASL; 7: ENSO; 8: DMI)
last_year = 2099 # last year included in the computation
dt = 1 # time step (years)
n_iter = 1000 # number of bootstrap realizations
conf = 2.57 # 1.96 if 95% confidence interval (normal distribution); 1.65 if 90% and 2.57 if 99%
compute_liang = True # True compute Liang index; False: load existing value
save_var = True
save_fig = False

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
dir_fig = '/home/dadocq/Documents/Papers/My_Papers/RESIST_Antarctic/figures/seasons/sectors/'

# Number of members
if model == 'SMHI-LENS' or model == 'CESM2-LE' or model == 'MPI-ESM1-2-LR' or model == 'CanESM5':
    n_members = 50 # number of members
if model == 'ACCESS-ESM1-5':
    n_members = 40

# Function to test significance (based on the confidence interval)
def compute_sig(var,error,conf):
    if (var-conf*error < 0. and var+conf*error < 0.) or (var-conf*error > 0. and var+conf*error > 0.):
        sig = 1
    else:
        sig = 0
    return sig

# Load variables
filename = dir_input + string_model + '_Antarctic_timeseries_' + season + '.npy'
notused,notused,notused,notused,sam,asl,nino,dmi = np.load(filename,allow_pickle=True)

# Load SSIE for sectors
filename = dir_input + 'SSIE_' + string_model + '_timeseries.npy'
ssie_bas,ssie_ws,ssie_io,ssie_wpo,ssie_rs = np.load(filename,allow_pickle=True)

# Load PSIE for sectors
filename = dir_input + 'SIE_' + string_model + '_timeseries_' + season + '.npy'
psie_bas,psie_ws,psie_io,psie_wpo,psie_rs = np.load(filename,allow_pickle=True)

# Load T2m for sectors
filename = dir_input + 'tas_mon_' + string_model + '_timeseries_' + season + '.npy'
tas_bas,tas_ws,tas_io,tas_wpo,tas_rs = np.load(filename,allow_pickle=True)

# Load SST for sectors
filename = dir_input + 'sst_mon_' + string_model + '_timeseries_' + season + '.npy'
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

# Set NaN values to 0 for sea ice
ssie[np.isnan(ssie)] = 0
psie[np.isnan(psie)] = 0

# Interpolate NaN values for ASL
for m in np.arange(n_members):
    asl[m,:] = interpolate_nan(asl[m,:])

# Shift summer SIE to the left (so that it lags other variables)
ssie = np.roll(ssie,-1,axis=1)

# Take years of interest and save variables
ind_last_year = int(last_year-1970+1)
nyears_new = ind_last_year
ssie2 = np.zeros((n_members,nyears_new))
psie2 = np.zeros((n_members,nyears_new))
tas2 = np.zeros((n_members,nyears_new))
sam2 = np.zeros((n_members,nyears_new))
sst2 = np.zeros((n_members,nyears_new))
asl2 = np.zeros((n_members,nyears_new))
nino2 = np.zeros((n_members,nyears_new))
dmi2 = np.zeros((n_members,nyears_new))
for m in np.arange(n_members):
    ssie2[m,:] = ssie[m,0:ind_last_year]
    psie2[m,:] = psie[m,0:ind_last_year]
    tas2[m,:] = tas[m,0:ind_last_year]
    sam2[m,:] = sam[m,0:ind_last_year]
    sst2[m,:] = sst[m,0:ind_last_year]
    asl2[m,:] = asl[m,0:ind_last_year]
    nino2[m,:] = nino[m,0:ind_last_year]
    dmi2[m,:] = dmi[m,0:ind_last_year]

# Compute ensemble mean
ssie_ensmean = np.nanmean(ssie2,axis=0)
psie_ensmean = np.nanmean(psie2,axis=0)
tas_ensmean = np.nanmean(tas2,axis=0)
sam_ensmean = np.nanmean(sam2,axis=0)
sst_ensmean = np.nanmean(sst2,axis=0)
asl_ensmean = np.nanmean(asl2,axis=0)
nino_ensmean = np.nanmean(nino2,axis=0)
dmi_ensmean = np.nanmean(dmi2,axis=0)
    
# Detrend data (remove ensemble mean)
for m in np.arange(n_members):
    ssie2[m,:] = ssie2[m,:] - ssie_ensmean[0:ind_last_year]
    psie2[m,:] = psie2[m,:] - psie_ensmean[0:ind_last_year]
    tas2[m,:] = tas2[m,:] - tas_ensmean[0:ind_last_year]
    sam2[m,:] = sam2[m,:] - sam_ensmean[0:ind_last_year]
    sst2[m,:] = sst2[m,:] - sst_ensmean[0:ind_last_year]
    asl2[m,:] = asl2[m,:] - asl_ensmean[0:ind_last_year]
    nino2[m,:] = nino2[m,:] - nino_ensmean[0:ind_last_year]
    dmi2[m,:] = dmi2[m,:] - dmi_ensmean[0:ind_last_year]

# Compute dx/dt (tendency) of detrended data for Liang index
dssie2 = np.zeros((n_members,nyears_new))
dpsie2 = np.zeros((n_members,nyears_new))
dtas2 = np.zeros((n_members,nyears_new))
dsam2 = np.zeros((n_members,nyears_new))
dsst2 = np.zeros((n_members,nyears_new))
dasl2 = np.zeros((n_members,nyears_new))
dnino2 = np.zeros((n_members,nyears_new))
ddmi2 = np.zeros((n_members,nyears_new))
for m in np.arange(n_members):
    dssie2[m,0:nyears_new-1] = (ssie2[m,1:nyears_new] - ssie2[m,0:nyears_new-1]) / dt
    dpsie2[m,0:nyears_new-1] = (psie2[m,1:nyears_new] - psie2[m,0:nyears_new-1]) / dt
    dtas2[m,0:nyears_new-1] = (tas2[m,1:nyears_new] - tas2[m,0:nyears_new-1]) / dt
    dsam2[m,0:nyears_new-1] = (sam2[m,1:nyears_new] - sam2[m,0:nyears_new-1]) / dt
    dsst2[m,0:nyears_new-1] = (sst2[m,1:nyears_new] - sst2[m,0:nyears_new-1]) / dt
    dasl2[m,0:nyears_new-1] = (asl2[m,1:nyears_new] - asl2[m,0:nyears_new-1]) / dt
    dnino2[m,0:nyears_new-1] = (nino2[m,1:nyears_new] - nino2[m,0:nyears_new-1]) / dt
    ddmi2[m,0:nyears_new-1] = (dmi2[m,1:nyears_new] - dmi2[m,0:nyears_new-1]) / dt

# Concatenate all members and create 1 single time series for each variable
nt_full = nyears_new * n_members
ssie_full = np.reshape(ssie2,nt_full)
dssie_full = np.reshape(dssie2,nt_full)
psie_full = np.reshape(psie2,nt_full)
dpsie_full = np.reshape(dpsie2,nt_full)
tas_full = np.reshape(tas2,nt_full)
dtas_full = np.reshape(dtas2,nt_full)
sam_full = np.reshape(sam2,nt_full)
dsam_full = np.reshape(dsam2,nt_full)
sst_full = np.reshape(sst2,nt_full)
dsst_full = np.reshape(dsst2,nt_full)
asl_full = np.reshape(asl2,nt_full)
dasl_full = np.reshape(dasl2,nt_full)
nino_full = np.reshape(nino2,nt_full)
dnino_full = np.reshape(dnino2,nt_full)
dmi_full = np.reshape(dmi2,nt_full)
ddmi_full = np.reshape(ddmi2,nt_full)

# Compute relative transfer of information (tau) and correlation coefficient (R) and save in files
filename = dir_input + 'Liang_' + sector + '_' + model + '_' + season + '_' + str(n_iter) + 'boot.npy'
if compute_liang == True:
    tau = np.zeros((nvar,nvar))
    R = np.zeros((nvar,nvar))
    error_tau = np.zeros((nvar,nvar))
    error_R = np.zeros((nvar,nvar))
    xx = np.array((ssie_full,psie_full,tas_full,sst_full,sam_full,asl_full,nino_full,dmi_full))
    dx = np.array((dssie_full,dpsie_full,dtas_full,dsst_full,dsam_full,dasl_full,dnino_full,ddmi_full))
    tau,R,error_tau,error_R,noise = compute_liang_nvar(xx,dx,dt,n_iter)
    if save_var == True:
        np.save(filename,[tau,R,error_tau,error_R,noise])
else:
    tau,R,error_tau,error_R,noise = np.load(filename,allow_pickle=True)

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
    filename = dir_fig + 'Liang_' + sector + '_' + model + '_' + season + '_' + str(n_iter) + 'boot.png'
    fig.savefig(filename)