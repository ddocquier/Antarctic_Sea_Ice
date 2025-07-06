#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Figure 7 (A8): Plot Liang index (correlation) from models over whole period (1970-2099) and for different Antarctic sectors as well as observations (1982-2023)

Model large ensembles: EC-Earth3 (SMHI-LENS), CESM2-LE, MPI-ESM1-2-LR, CanESM5, ACCESS-ESM1-5
Liang index computed via compute_liang_antreg_seasons.py

Observations
Liang index computed via compute_liang_antreg_seasons_obs.py

Target variable: Summer sea-ice extent (DJF)

Drivers:
Previous winter/spring sea-ice extent (JAS or OND)
Previous Antarctic mean surface air temperature (<60S; JAS or OND)
Previous Antarctic mean SST (<60S; JAS or OND)
Previous Southern Annular Mode (SAM; JAS or OND)
Previous Amundsen Sea Low (ASL; JAS or OND)
Previous Niño3.4 (JAS or OND)
Previous DMI (JAS or OND)

Last updated: 01/07/2025

@author: David Docquier
"""

# Standard libraries
import numpy as np
import matplotlib.pyplot as plt

# Parameters
season = 'SON' # JAS (previous winter), SON (previous spring), OND (previous spring; default)
nvar = 8 # number of variables (1: SSIE; 2: PSIE; 3: T_2m; 4: SST; 5: SAM; 6: ASL; 7: ENSO; 8: DMI)
n_iter = 1000 # number of bootstrap realizations; default: 1000
conf = 2.57 # 1.96 if 95% confidence interval (normal distribution); 1.65 if 90% and 2.57 if 99%
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
dir_input = '/home/ddocquier/Documents/Papers/My_Papers/RESIST_Antarctic/output/seasons/'
dir_fig = '/home/ddocquier/Documents/Papers/My_Papers/RESIST_Antarctic/LaTeX/'

# Load SMHI-LENS BAS
filename = dir_input + 'Liang_bas_SMHI-LENS_' + season + '_' + str(n_iter) + 'boot.npy'
if season == 'SON':
    tau_bas_ecearth,R_bas_ecearth,error_tau_bas_ecearth,error_R_bas_ecearth = np.load(filename,allow_pickle=True)
else:
    tau_bas_ecearth,R_bas_ecearth,error_tau_bas_ecearth,error_R_bas_ecearth,notused = np.load(filename,allow_pickle=True)

# Load SMHI-LENS WS
filename = dir_input + 'Liang_ws_SMHI-LENS_' + season + '_' + str(n_iter) + 'boot.npy'
if season == 'SON':
    tau_ws_ecearth,R_ws_ecearth,error_tau_ws_ecearth,error_R_ws_ecearth = np.load(filename,allow_pickle=True)
else:
    tau_ws_ecearth,R_ws_ecearth,error_tau_ws_ecearth,error_R_ws_ecearth,notused = np.load(filename,allow_pickle=True)

# Load SMHI-LENS IO
filename = dir_input + 'Liang_io_SMHI-LENS_' + season + '_' + str(n_iter) + 'boot.npy'
if season == 'SON':
    tau_io_ecearth,R_io_ecearth,error_tau_io_ecearth,error_R_io_ecearth = np.load(filename,allow_pickle=True)
else:
    tau_io_ecearth,R_io_ecearth,error_tau_io_ecearth,error_R_io_ecearth,notused = np.load(filename,allow_pickle=True)

# Load SMHI-LENS WPO
filename = dir_input + 'Liang_wpo_SMHI-LENS_' + season + '_' + str(n_iter) + 'boot.npy'
if season == 'SON':
    tau_wpo_ecearth,R_wpo_ecearth,error_tau_wpo_ecearth,error_R_wpo_ecearth = np.load(filename,allow_pickle=True)
else:
    tau_wpo_ecearth,R_wpo_ecearth,error_tau_wpo_ecearth,error_R_wpo_ecearth,notused = np.load(filename,allow_pickle=True)

# Load SMHI-LENS RS
filename = dir_input + 'Liang_rs_SMHI-LENS_' + season + '_' + str(n_iter) + 'boot.npy'
if season == 'SON':
    tau_rs_ecearth,R_rs_ecearth,error_tau_rs_ecearth,error_R_rs_ecearth = np.load(filename,allow_pickle=True)
else:
    tau_rs_ecearth,R_rs_ecearth,error_tau_rs_ecearth,error_R_rs_ecearth,notused = np.load(filename,allow_pickle=True)

# Load CESM2-LE BAS
filename = dir_input + 'Liang_bas_CESM2-LE_' + season + '_' + str(n_iter) + 'boot.npy'
if season == 'SON':
    tau_bas_cesm,R_bas_cesm,error_tau_bas_cesm,error_R_bas_cesm = np.load(filename,allow_pickle=True)
else:
    tau_bas_cesm,R_bas_cesm,error_tau_bas_cesm,error_R_bas_cesm,notused = np.load(filename,allow_pickle=True)

# Load CESM2-LE WS
filename = dir_input + 'Liang_ws_CESM2-LE_' + season + '_' + str(n_iter) + 'boot.npy'
if season == 'SON':
    tau_ws_cesm,R_ws_cesm,error_tau_ws_cesm,error_R_ws_cesm = np.load(filename,allow_pickle=True)
else:
    tau_ws_cesm,R_ws_cesm,error_tau_ws_cesm,error_R_ws_cesm,notused = np.load(filename,allow_pickle=True)

# Load CESM2-LE IO
filename = dir_input + 'Liang_io_CESM2-LE_' + season + '_' + str(n_iter) + 'boot.npy'
if season == 'SON':
    tau_io_cesm,R_io_cesm,error_tau_io_cesm,error_R_io_cesm = np.load(filename,allow_pickle=True)
else:
    tau_io_cesm,R_io_cesm,error_tau_io_cesm,error_R_io_cesm,notused = np.load(filename,allow_pickle=True)

# Load CESM2-LE WPO
filename = dir_input + 'Liang_wpo_CESM2-LE_' + season + '_' + str(n_iter) + 'boot.npy'
if season == 'SON':
    tau_wpo_cesm,R_wpo_cesm,error_tau_wpo_cesm,error_R_wpo_cesm = np.load(filename,allow_pickle=True)
else:
    tau_wpo_cesm,R_wpo_cesm,error_tau_wpo_cesm,error_R_wpo_cesm,notused = np.load(filename,allow_pickle=True)

# Load CESM2-LE RS
filename = dir_input + 'Liang_rs_CESM2-LE_' + season + '_' + str(n_iter) + 'boot.npy'
if season == 'SON':
    tau_rs_cesm,R_rs_cesm,error_tau_rs_cesm,error_R_rs_cesm = np.load(filename,allow_pickle=True)
else:
    tau_rs_cesm,R_rs_cesm,error_tau_rs_cesm,error_R_rs_cesm,notused = np.load(filename,allow_pickle=True)

# Load MPI-ESM1-2-LR BAS
filename = dir_input + 'Liang_bas_MPI-ESM1-2-LR_' + season + '_' + str(n_iter) + 'boot.npy'
if season == 'SON':
    tau_bas_mpi,R_bas_mpi,error_tau_bas_mpi,error_R_bas_mpi = np.load(filename,allow_pickle=True)
else:
    tau_bas_mpi,R_bas_mpi,error_tau_bas_mpi,error_R_bas_mpi,notused = np.load(filename,allow_pickle=True)

# Load MPI-ESM1-2-LR WS
filename = dir_input + 'Liang_ws_MPI-ESM1-2-LR_' + season + '_' + str(n_iter) + 'boot.npy'
if season == 'SON':
    tau_ws_mpi,R_ws_mpi,error_tau_ws_mpi,error_R_ws_mpi = np.load(filename,allow_pickle=True)
else:
    tau_ws_mpi,R_ws_mpi,error_tau_ws_mpi,error_R_ws_mpi,notused = np.load(filename,allow_pickle=True)

# Load MPI-ESM1-2-LR IO
filename = dir_input + 'Liang_io_MPI-ESM1-2-LR_' + season + '_' + str(n_iter) + 'boot.npy'
if season == 'SON':
    tau_io_mpi,R_io_mpi,error_tau_io_mpi,error_R_io_mpi = np.load(filename,allow_pickle=True)
else:
    tau_io_mpi,R_io_mpi,error_tau_io_mpi,error_R_io_mpi,notused = np.load(filename,allow_pickle=True)

# Load MPI-ESM1-2-LR WPO
filename = dir_input + 'Liang_wpo_MPI-ESM1-2-LR_' + season + '_' + str(n_iter) + 'boot.npy'
if season == 'SON':
    tau_wpo_mpi,R_wpo_mpi,error_tau_wpo_mpi,error_R_wpo_mpi = np.load(filename,allow_pickle=True)
else:
    tau_wpo_mpi,R_wpo_mpi,error_tau_wpo_mpi,error_R_wpo_mpi,notused = np.load(filename,allow_pickle=True)

# Load MPI-ESM1-2-LR RS
filename = dir_input + 'Liang_rs_MPI-ESM1-2-LR_' + season + '_' + str(n_iter) + 'boot.npy'
if season == 'SON':
    tau_rs_mpi,R_rs_mpi,error_tau_rs_mpi,error_R_rs_mpi = np.load(filename,allow_pickle=True)
else:
    tau_rs_mpi,R_rs_mpi,error_tau_rs_mpi,error_R_rs_mpi,notused = np.load(filename,allow_pickle=True)

# Load CanESM5 BAS
filename = dir_input + 'Liang_bas_CanESM5_' + season + '_' + str(n_iter) + 'boot.npy'
if season == 'SON':
    tau_bas_canesm,R_bas_canesm,error_tau_bas_canesm,error_R_bas_canesm = np.load(filename,allow_pickle=True)
else:
    tau_bas_canesm,R_bas_canesm,error_tau_bas_canesm,error_R_bas_canesm,notused = np.load(filename,allow_pickle=True)

# Load CanESM5 WS
filename = dir_input + 'Liang_ws_CanESM5_' + season + '_' + str(n_iter) + 'boot.npy'
if season == 'SON':
    tau_ws_canesm,R_ws_canesm,error_tau_ws_canesm,error_R_ws_canesm = np.load(filename,allow_pickle=True)
else:
    tau_ws_canesm,R_ws_canesm,error_tau_ws_canesm,error_R_ws_canesm,notused = np.load(filename,allow_pickle=True)

# Load CanESM5 IO
filename = dir_input + 'Liang_io_CanESM5_' + season + '_' + str(n_iter) + 'boot.npy'
if season == 'SON':
    tau_io_canesm,R_io_canesm,error_tau_io_canesm,error_R_io_canesm = np.load(filename,allow_pickle=True)
else:
    tau_io_canesm,R_io_canesm,error_tau_io_canesm,error_R_io_canesm,notused = np.load(filename,allow_pickle=True)

# Load CanESM5 WPO
filename = dir_input + 'Liang_wpo_CanESM5_' + season + '_' + str(n_iter) + 'boot.npy'
if season == 'SON':
    tau_wpo_canesm,R_wpo_canesm,error_tau_wpo_canesm,error_R_wpo_canesm = np.load(filename,allow_pickle=True)
else:
    tau_wpo_canesm,R_wpo_canesm,error_tau_wpo_canesm,error_R_wpo_canesm,notused = np.load(filename,allow_pickle=True)

# Load CanESM5 RS
filename = dir_input + 'Liang_rs_CanESM5_' + season + '_' + str(n_iter) + 'boot.npy'
if season == 'SON':
    tau_rs_canesm,R_rs_canesm,error_tau_rs_canesm,error_R_rs_canesm = np.load(filename,allow_pickle=True)
else:
    tau_rs_canesm,R_rs_canesm,error_tau_rs_canesm,error_R_rs_canesm,notused = np.load(filename,allow_pickle=True)

# Load ACCESS-ESM1-5 BAS
filename = dir_input + 'Liang_bas_ACCESS-ESM1-5_' + season + '_' + str(n_iter) + 'boot.npy'
if season == 'SON':
    tau_bas_access,R_bas_access,error_tau_bas_access,error_R_bas_access = np.load(filename,allow_pickle=True)
else:
    tau_bas_access,R_bas_access,error_tau_bas_access,error_R_bas_access,notused = np.load(filename,allow_pickle=True)

# Load ACCESS-ESM1-5 WS
filename = dir_input + 'Liang_ws_ACCESS-ESM1-5_' + season + '_' + str(n_iter) + 'boot.npy'
if season == 'SON':
    tau_ws_access,R_ws_access,error_tau_ws_access,error_R_ws_access = np.load(filename,allow_pickle=True)
else:
    tau_ws_access,R_ws_access,error_tau_ws_access,error_R_ws_access,notused = np.load(filename,allow_pickle=True)

# Load ACCESS-ESM1-5 IO
filename = dir_input + 'Liang_io_ACCESS-ESM1-5_' + season + '_' + str(n_iter) + 'boot.npy'
if season == 'SON':
    tau_io_access,R_io_access,error_tau_io_access,error_R_io_access = np.load(filename,allow_pickle=True)
else:
    tau_io_access,R_io_access,error_tau_io_access,error_R_io_access,notused = np.load(filename,allow_pickle=True)

# Load ACCESS-ESM1-5 WPO
filename = dir_input + 'Liang_wpo_ACCESS-ESM1-5_' + season + '_' + str(n_iter) + 'boot.npy'
if season == 'SON':
    tau_wpo_access,R_wpo_access,error_tau_wpo_access,error_R_wpo_access = np.load(filename,allow_pickle=True)
else:
     tau_wpo_access,R_wpo_access,error_tau_wpo_access,error_R_wpo_access,notused = np.load(filename,allow_pickle=True)

# Load ACCESS-ESM1-5 RS
filename = dir_input + 'Liang_rs_ACCESS-ESM1-5_' + season + '_' + str(n_iter) + 'boot.npy'
if season == 'SON':
    tau_rs_access,R_rs_access,error_tau_rs_access,error_R_rs_access = np.load(filename,allow_pickle=True)
else:
    tau_rs_access,R_rs_access,error_tau_rs_access,error_R_rs_access,notused = np.load(filename,allow_pickle=True)

# Compute signficance of SMHI-LENS and CESM2-LE
sig_tau_bas_ecearth = np.zeros((nvar,nvar))
sig_R_bas_ecearth = np.zeros((nvar,nvar))
sig_tau_bas_cesm = np.zeros((nvar,nvar))
sig_R_bas_cesm = np.zeros((nvar,nvar))

sig_tau_ws_ecearth = np.zeros((nvar,nvar))
sig_R_ws_ecearth = np.zeros((nvar,nvar))
sig_tau_ws_cesm = np.zeros((nvar,nvar))
sig_R_ws_cesm = np.zeros((nvar,nvar))

sig_tau_io_ecearth = np.zeros((nvar,nvar))
sig_R_io_ecearth = np.zeros((nvar,nvar))
sig_tau_io_cesm = np.zeros((nvar,nvar))
sig_R_io_cesm = np.zeros((nvar,nvar))

sig_tau_wpo_ecearth = np.zeros((nvar,nvar))
sig_R_wpo_ecearth = np.zeros((nvar,nvar))
sig_tau_wpo_cesm = np.zeros((nvar,nvar))
sig_R_wpo_cesm = np.zeros((nvar,nvar))

sig_tau_rs_ecearth = np.zeros((nvar,nvar))
sig_R_rs_ecearth = np.zeros((nvar,nvar))
sig_tau_rs_cesm = np.zeros((nvar,nvar))
sig_R_rs_cesm = np.zeros((nvar,nvar))
for i in np.arange(nvar):
    for j in np.arange(nvar):
        sig_tau_bas_ecearth[i,j] = compute_sig(tau_bas_ecearth[i,j],error_tau_bas_ecearth[i,j],conf)
        sig_R_bas_ecearth[i,j] = compute_sig(R_bas_ecearth[i,j],error_R_bas_ecearth[i,j],conf)
        sig_tau_bas_cesm[i,j] = compute_sig(tau_bas_cesm[i,j],error_tau_bas_cesm[i,j],conf)
        sig_R_bas_cesm[i,j] = compute_sig(R_bas_cesm[i,j],error_R_bas_cesm[i,j],conf)
        
        sig_tau_ws_ecearth[i,j] = compute_sig(tau_ws_ecearth[i,j],error_tau_ws_ecearth[i,j],conf)
        sig_R_ws_ecearth[i,j] = compute_sig(R_ws_ecearth[i,j],error_R_ws_ecearth[i,j],conf)
        sig_tau_ws_cesm[i,j] = compute_sig(tau_ws_cesm[i,j],error_tau_ws_cesm[i,j],conf)
        sig_R_ws_cesm[i,j] = compute_sig(R_ws_cesm[i,j],error_R_ws_cesm[i,j],conf)
        
        sig_tau_io_ecearth[i,j] = compute_sig(tau_io_ecearth[i,j],error_tau_io_ecearth[i,j],conf)
        sig_R_io_ecearth[i,j] = compute_sig(R_io_ecearth[i,j],error_R_io_ecearth[i,j],conf)
        sig_tau_io_cesm[i,j] = compute_sig(tau_io_cesm[i,j],error_tau_io_cesm[i,j],conf)
        sig_R_io_cesm[i,j] = compute_sig(R_io_cesm[i,j],error_R_io_cesm[i,j],conf)
        
        sig_tau_wpo_ecearth[i,j] = compute_sig(tau_wpo_ecearth[i,j],error_tau_wpo_ecearth[i,j],conf)
        sig_R_wpo_ecearth[i,j] = compute_sig(R_wpo_ecearth[i,j],error_R_wpo_ecearth[i,j],conf)
        sig_tau_wpo_cesm[i,j] = compute_sig(tau_wpo_cesm[i,j],error_tau_wpo_cesm[i,j],conf)
        sig_R_wpo_cesm[i,j] = compute_sig(R_wpo_cesm[i,j],error_R_wpo_cesm[i,j],conf)
        
        sig_tau_rs_ecearth[i,j] = compute_sig(tau_rs_ecearth[i,j],error_tau_rs_ecearth[i,j],conf)
        sig_R_rs_ecearth[i,j] = compute_sig(R_rs_ecearth[i,j],error_R_rs_ecearth[i,j],conf)
        sig_tau_rs_cesm[i,j] = compute_sig(tau_rs_cesm[i,j],error_tau_rs_cesm[i,j],conf)
        sig_R_rs_cesm[i,j] = compute_sig(R_rs_cesm[i,j],error_R_rs_cesm[i,j],conf)
            
# Compute signficance of MPI-ESM1-2-LR
sig_tau_bas_mpi = np.zeros((nvar,nvar))
sig_R_bas_mpi = np.zeros((nvar,nvar))
sig_tau_ws_mpi = np.zeros((nvar,nvar))
sig_R_ws_mpi = np.zeros((nvar,nvar))
sig_tau_io_mpi = np.zeros((nvar,nvar))
sig_R_io_mpi = np.zeros((nvar,nvar))
sig_tau_wpo_mpi = np.zeros((nvar,nvar))
sig_R_wpo_mpi = np.zeros((nvar,nvar))
sig_tau_rs_mpi = np.zeros((nvar,nvar))
sig_R_rs_mpi = np.zeros((nvar,nvar))
for i in np.arange(nvar):
    for j in np.arange(nvar):
        sig_tau_bas_mpi[i,j] = compute_sig(tau_bas_mpi[i,j],error_tau_bas_mpi[i,j],conf)
        sig_R_bas_mpi[i,j] = compute_sig(R_bas_mpi[i,j],error_R_bas_mpi[i,j],conf)
        sig_tau_ws_mpi[i,j] = compute_sig(tau_ws_mpi[i,j],error_tau_ws_mpi[i,j],conf)
        sig_R_ws_mpi[i,j] = compute_sig(R_ws_mpi[i,j],error_R_ws_mpi[i,j],conf)
        sig_tau_io_mpi[i,j] = compute_sig(tau_io_mpi[i,j],error_tau_io_mpi[i,j],conf)
        sig_R_io_mpi[i,j] = compute_sig(R_io_mpi[i,j],error_R_io_mpi[i,j],conf)
        sig_tau_wpo_mpi[i,j] = compute_sig(tau_wpo_mpi[i,j],error_tau_wpo_mpi[i,j],conf)
        sig_R_wpo_mpi[i,j] = compute_sig(R_wpo_mpi[i,j],error_R_wpo_mpi[i,j],conf)
        sig_tau_rs_mpi[i,j] = compute_sig(tau_rs_mpi[i,j],error_tau_rs_mpi[i,j],conf)
        sig_R_rs_mpi[i,j] = compute_sig(R_rs_mpi[i,j],error_R_rs_mpi[i,j],conf)
            
# Compute signficance of CanESM5
sig_tau_bas_canesm = np.zeros((nvar,nvar))
sig_R_bas_canesm = np.zeros((nvar,nvar))
sig_tau_ws_canesm = np.zeros((nvar,nvar))
sig_R_ws_canesm = np.zeros((nvar,nvar))
sig_tau_io_canesm = np.zeros((nvar,nvar))
sig_R_io_canesm = np.zeros((nvar,nvar))
sig_tau_wpo_canesm = np.zeros((nvar,nvar))
sig_R_wpo_canesm = np.zeros((nvar,nvar))
sig_tau_rs_canesm = np.zeros((nvar,nvar))
sig_R_rs_canesm = np.zeros((nvar,nvar))
for i in np.arange(nvar):
    for j in np.arange(nvar):
        sig_tau_bas_canesm[i,j] = compute_sig(tau_bas_canesm[i,j],error_tau_bas_canesm[i,j],conf)
        sig_R_bas_canesm[i,j] = compute_sig(R_bas_canesm[i,j],error_R_bas_canesm[i,j],conf)
        sig_tau_ws_canesm[i,j] = compute_sig(tau_ws_canesm[i,j],error_tau_ws_canesm[i,j],conf)
        sig_R_ws_canesm[i,j] = compute_sig(R_ws_canesm[i,j],error_R_ws_canesm[i,j],conf)
        sig_tau_io_canesm[i,j] = compute_sig(tau_io_canesm[i,j],error_tau_io_canesm[i,j],conf)
        sig_R_io_canesm[i,j] = compute_sig(R_io_canesm[i,j],error_R_io_canesm[i,j],conf)
        sig_tau_wpo_canesm[i,j] = compute_sig(tau_wpo_canesm[i,j],error_tau_wpo_canesm[i,j],conf)
        sig_R_wpo_canesm[i,j] = compute_sig(R_wpo_canesm[i,j],error_R_wpo_canesm[i,j],conf)
        sig_tau_rs_canesm[i,j] = compute_sig(tau_rs_canesm[i,j],error_tau_rs_canesm[i,j],conf)
        sig_R_rs_canesm[i,j] = compute_sig(R_rs_canesm[i,j],error_R_rs_canesm[i,j],conf)

# Compute signficance of ACCESS-ESM1-5
sig_tau_bas_access = np.zeros((nvar,nvar))
sig_R_bas_access = np.zeros((nvar,nvar))
sig_tau_ws_access = np.zeros((nvar,nvar))
sig_R_ws_access = np.zeros((nvar,nvar))
sig_tau_io_access = np.zeros((nvar,nvar))
sig_R_io_access = np.zeros((nvar,nvar))
sig_tau_wpo_access = np.zeros((nvar,nvar))
sig_R_wpo_access = np.zeros((nvar,nvar))
sig_tau_rs_access = np.zeros((nvar,nvar))
sig_R_rs_access = np.zeros((nvar,nvar))
for i in np.arange(nvar):
    for j in np.arange(nvar):
        sig_tau_bas_access[i,j] = compute_sig(tau_bas_access[i,j],error_tau_bas_access[i,j],conf)
        sig_R_bas_access[i,j] = compute_sig(R_bas_access[i,j],error_R_bas_access[i,j],conf)
        sig_tau_ws_access[i,j] = compute_sig(tau_ws_access[i,j],error_tau_ws_access[i,j],conf)
        sig_R_ws_access[i,j] = compute_sig(R_ws_access[i,j],error_R_ws_access[i,j],conf)
        sig_tau_io_access[i,j] = compute_sig(tau_io_access[i,j],error_tau_io_access[i,j],conf)
        sig_R_io_access[i,j] = compute_sig(R_io_access[i,j],error_R_io_access[i,j],conf)
        sig_tau_wpo_access[i,j] = compute_sig(tau_wpo_access[i,j],error_tau_wpo_access[i,j],conf)
        sig_R_wpo_access[i,j] = compute_sig(R_wpo_access[i,j],error_R_wpo_access[i,j],conf)
        sig_tau_rs_access[i,j] = compute_sig(tau_rs_access[i,j],error_tau_rs_access[i,j],conf)
        sig_R_rs_access[i,j] = compute_sig(R_rs_access[i,j],error_R_rs_access[i,j],conf)

# Load observations BAS
filename = dir_input + 'Liang_bas_obs_' + season + '.npy'
tau_bas_obs,R_bas_obs,error_tau_bas_obs,error_R_bas_obs = np.load(filename,allow_pickle=True)

# Load observations WS
filename = dir_input + 'Liang_ws_obs_' + season + '.npy'
tau_ws_obs,R_ws_obs,error_tau_ws_obs,error_R_ws_obs = np.load(filename,allow_pickle=True)

# Load observations IO
filename = dir_input + 'Liang_io_obs_' + season + '.npy'
tau_io_obs,R_io_obs,error_tau_io_obs,error_R_io_obs = np.load(filename,allow_pickle=True)

# Load observations WPO
filename = dir_input + 'Liang_wpo_obs_' + season + '.npy'
tau_wpo_obs,R_wpo_obs,error_tau_wpo_obs,error_R_wpo_obs = np.load(filename,allow_pickle=True)

# Load observations RS
filename = dir_input + 'Liang_rs_obs_' + season + '.npy'
tau_rs_obs,R_rs_obs,error_tau_rs_obs,error_R_rs_obs = np.load(filename,allow_pickle=True)

# Compute significance of obs.
sig_tau_bas_obs = np.zeros((nvar,nvar))
sig_R_bas_obs = np.zeros((nvar,nvar))
sig_tau_ws_obs = np.zeros((nvar,nvar))
sig_R_ws_obs = np.zeros((nvar,nvar))
sig_tau_io_obs = np.zeros((nvar,nvar))
sig_R_io_obs = np.zeros((nvar,nvar))
sig_tau_wpo_obs = np.zeros((nvar,nvar))
sig_R_wpo_obs = np.zeros((nvar,nvar))
sig_tau_rs_obs = np.zeros((nvar,nvar))
sig_R_rs_obs = np.zeros((nvar,nvar))
for i in np.arange(nvar):
    for j in np.arange(nvar):
        sig_tau_bas_obs[i,j] = compute_sig(tau_bas_obs[i,j],error_tau_bas_obs[i,j],conf_obs)
        sig_R_bas_obs[i,j] = compute_sig(R_bas_obs[i,j],error_R_bas_obs[i,j],conf_obs)
        sig_tau_ws_obs[i,j] = compute_sig(tau_ws_obs[i,j],error_tau_ws_obs[i,j],conf_obs)
        sig_R_ws_obs[i,j] = compute_sig(R_ws_obs[i,j],error_R_ws_obs[i,j],conf_obs)
        sig_tau_io_obs[i,j] = compute_sig(tau_io_obs[i,j],error_tau_io_obs[i,j],conf_obs)
        sig_R_io_obs[i,j] = compute_sig(R_io_obs[i,j],error_R_io_obs[i,j],conf_obs)
        sig_tau_wpo_obs[i,j] = compute_sig(tau_wpo_obs[i,j],error_tau_wpo_obs[i,j],conf_obs)
        sig_R_wpo_obs[i,j] = compute_sig(R_wpo_obs[i,j],error_R_wpo_obs[i,j],conf_obs)
        sig_tau_rs_obs[i,j] = compute_sig(tau_rs_obs[i,j],error_tau_rs_obs[i,j],conf_obs)
        sig_R_rs_obs[i,j] = compute_sig(R_rs_obs[i,j],error_R_rs_obs[i,j],conf_obs)

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

# Figure of Liang index (tau)
fig,ax = plt.subplots(3,2,figsize=(18,15))
fig.subplots_adjust(left=0.11,bottom=0.08,right=0.94,top=0.93,hspace=0.4,wspace=0.25)

#############
# Antarctic #
#############

# tau ACCESS
ax[0,0].errorbar(index[0]+0.8,np.abs(tau_access[1,0]),yerr=conf*error_tau_access[1,0],fmt='o',color='orange',markersize=12,label='ACCESS-ESM1.5')
for i in np.arange(1,np.size(index)+1):
    if sig_tau_access[i,0] == 1:
        ax[0,0].plot(index[i-1]+0.8,np.abs(tau_access[i,0]),'ko',markersize=17)
    ax[0,0].errorbar(index[i-1]+0.8,np.abs(tau_access[i,0]),yerr=conf*error_tau_access[i,0],fmt='o',color='orange',markersize=12)
    
# tau CanESM
ax[0,0].errorbar(index[0]+0.9,np.abs(tau_canesm[1,0]),yerr=conf*error_tau_canesm[1,0],fmt='go',markersize=12,label='CanESM5')
for i in np.arange(1,np.size(index)+1):
    if sig_tau_canesm[i,0] == 1:
        ax[0,0].plot(index[i-1]+0.9,np.abs(tau_canesm[i,0]),'ko',markersize=17)
    ax[0,0].errorbar(index[i-1]+0.9,np.abs(tau_canesm[i,0]),yerr=conf*error_tau_canesm[i,0],fmt='go',markersize=12)

# tau CESM
ax[0,0].errorbar(index[0]+1,np.abs(tau_cesm[1,0]),yerr=conf*error_tau_cesm[1,0],fmt='bo',markersize=12,label='CESM2')
for i in np.arange(1,np.size(index)+1):
    if sig_tau_cesm[i,0] == 1:
        ax[0,0].plot(index[i-1]+1,np.abs(tau_cesm[i,0]),'ko',markersize=17)
    ax1=ax[0,0].errorbar(index[i-1]+1,np.abs(tau_cesm[i,0]),yerr=conf*error_tau_cesm[i,0],fmt='bo',markersize=12)

# tau EC-Earth
ax[0,0].errorbar(index[0]+1.1,np.abs(tau_ecearth[1,0]),yerr=conf*error_tau_ecearth[1,0],fmt='ro',markersize=12,label='EC-Earth3')
for i in np.arange(1,np.size(index)+1):
    if sig_tau_ecearth[i,0] == 1:
        ax[0,0].plot(index[i-1]+1.1,np.abs(tau_ecearth[i,0]),'ko',markersize=17)
    ax[0,0].errorbar(index[i-1]+1.1,np.abs(tau_ecearth[i,0]),yerr=conf*error_tau_ecearth[i,0],fmt='ro',markersize=12)

# tau MPI
ax[0,0].errorbar(index[0]+1.2,np.abs(tau_mpi[1,0]),yerr=conf*error_tau_mpi[1,0],fmt='o',color='gray',markersize=12,label='MPI-ESM1.2-LR')
for i in np.arange(1,np.size(index)+1):
    if sig_tau_mpi[i,0] == 1:
        ax[0,0].plot(index[i-1]+1.2,np.abs(tau_mpi[i,0]),'ko',markersize=17)
    ax[0,0].errorbar(index[i-1]+1.2,np.abs(tau_mpi[i,0]),yerr=conf*error_tau_mpi[i,0],fmt='o',color='gray',markersize=12)

# tau obs.
ax1 = ax[0,0].errorbar(index[0]+1.3,np.abs(tau_obs[1,0]),yerr=conf_obs*error_tau_obs[1,0],fmt='x',color='black',markersize=12,label='Obs. (1982-2023)')
ax1[-1][0].set_linestyle('--')
for i in np.arange(1,np.size(index)+1):
    if sig_tau_obs[i,0] == 1:
        ax[0,0].plot(index[i-1]+1.3,np.abs(tau_obs[i,0]),'ko',markersize=17,fillstyle='none')
    ax1 = ax[0,0].errorbar(index[i-1]+1.3,np.abs(tau_obs[i,0]),yerr=conf_obs*error_tau_obs[i,0],fmt='x',color='black',markersize=12)
    ax1[-1][0].set_linestyle('--')

# Labels and legend
ax[0,0].set_ylabel(r'Information transfer $|\tau|$ ($\%$)',fontsize=20)
ax[0,0].tick_params(axis='both',labelsize=16)
ax[0,0].legend(loc='upper right',fontsize=14,shadow=True,frameon=False,ncol=2)
ax[0,0].set_xticks(np.arange(1,np.size(index)+1))
ax[0,0].set_xticklabels(label_names)
ax[0,0].grid(linestyle='--')
ax[0,0].set_ylim(-2,30)
ax[0,0].axes.axhline(y=0,color='k',linestyle='--')
ax[0,0].set_title('a',loc='left',fontsize=25,fontweight='bold')
ax[0,0].set_title('Pan-Antarctic',loc='center',fontsize=25,fontweight='bold')

#######
# BAS #
#######

# tau ACCESS
ax[0,1].errorbar(index[0]+0.8,np.abs(tau_bas_access[1,0]),yerr=conf*error_tau_bas_access[1,0],fmt='o',color='orange',markersize=12,label='ACCESS-ESM1.5')
for i in np.arange(1,np.size(index)+1):
    if sig_tau_bas_access[i,0] == 1:
        ax[0,1].plot(index[i-1]+0.8,np.abs(tau_bas_access[i,0]),'ko',markersize=17)
    ax[0,1].errorbar(index[i-1]+0.8,np.abs(tau_bas_access[i,0]),yerr=conf*error_tau_bas_access[i,0],fmt='o',color='orange',markersize=12)
    
# tau CanESM
ax[0,1].errorbar(index[0]+0.9,np.abs(tau_bas_canesm[1,0]),yerr=conf*error_tau_bas_canesm[1,0],fmt='go',markersize=12,label='CanESM5')
for i in np.arange(1,np.size(index)+1):
    if sig_tau_bas_canesm[i,0] == 1:
        ax[0,1].plot(index[i-1]+0.9,np.abs(tau_bas_canesm[i,0]),'ko',markersize=17)
    ax[0,1].errorbar(index[i-1]+0.9,np.abs(tau_bas_canesm[i,0]),yerr=conf*error_tau_bas_canesm[i,0],fmt='go',markersize=12)
    
# tau CESM
ax[0,1].errorbar(index[0]+1,np.abs(tau_bas_cesm[1,0]),yerr=conf*error_tau_bas_cesm[1,0],fmt='bo',markersize=12,label='CESM2')
for i in np.arange(1,np.size(index)+1):
    if sig_tau_bas_cesm[i,0] == 1:
        ax[0,1].plot(index[i-1]+1,np.abs(tau_bas_cesm[i,0]),'ko',markersize=17)
    ax[0,1].errorbar(index[i-1]+1,np.abs(tau_bas_cesm[i,0]),yerr=conf*error_tau_bas_cesm[i,0],fmt='bo',markersize=12)
 
# tau EC-Earth
ax[0,1].errorbar(index[0]+1.1,np.abs(tau_bas_ecearth[1,0]),yerr=conf*error_tau_bas_ecearth[1,0],fmt='ro',markersize=12,label='EC-Earth3')
for i in np.arange(1,np.size(index)+1):
    if sig_tau_bas_ecearth[i,0] == 1:
        ax[0,1].plot(index[i-1]+1.1,np.abs(tau_bas_ecearth[i,0]),'ko',markersize=17)
    ax[0,1].errorbar(index[i-1]+1.1,np.abs(tau_bas_ecearth[i,0]),yerr=conf*error_tau_bas_ecearth[i,0],fmt='ro',markersize=12)

# tau MPI
ax[0,1].errorbar(index[0]+1.2,np.abs(tau_bas_mpi[1,0]),yerr=conf*error_tau_bas_mpi[1,0],fmt='o',color='gray',markersize=12,label='MPI-ESM1.2-LR')
for i in np.arange(1,np.size(index)+1):
    if sig_tau_bas_mpi[i,0] == 1:
        ax[0,1].plot(index[i-1]+1.2,np.abs(tau_bas_mpi[i,0]),'ko',markersize=17)
    ax[0,1].errorbar(index[i-1]+1.2,np.abs(tau_bas_mpi[i,0]),yerr=conf*error_tau_bas_mpi[i,0],fmt='o',color='gray',markersize=12)

# tau obs.
ax1 = ax[0,1].errorbar(index[0]+1.3,np.abs(tau_bas_obs[1,0]),yerr=conf_obs*error_tau_bas_obs[1,0],fmt='x',color='black',markersize=12,label='Obs. (1982-2023)')
ax1[-1][0].set_linestyle('--')
for i in np.arange(1,np.size(index)+1):
    if sig_tau_bas_obs[i,0] == 1:
        ax[0,1].plot(index[i-1]+1.3,np.abs(tau_bas_obs[i,0]),'ko',markersize=17,fillstyle='none')
    ax1 = ax[0,1].errorbar(index[i-1]+1.3,np.abs(tau_bas_obs[i,0]),yerr=conf_obs*error_tau_bas_obs[i,0],fmt='x',color='black',markersize=12)
    ax1[-1][0].set_linestyle('--')

# Labels and legend
ax[0,1].set_ylabel(r'Information transfer $|\tau|$ ($\%$)',fontsize=20)
ax[0,1].tick_params(axis='both',labelsize=16)
ax[0,1].set_xticks(np.arange(1,np.size(index)+1))
ax[0,1].set_xticklabels(label_names)
ax[0,1].grid(linestyle='--')
ax[0,1].set_ylim(-2,30)
ax[0,1].axes.axhline(y=0,color='k',linestyle='--')
ax[0,1].set_title('b',loc='left',fontsize=25,fontweight='bold')
ax[0,1].set_title('Bellingshausen-Amundsen',loc='center',fontsize=25,fontweight='bold')

######
# WS #
######

# tau ACCESS
ax[1,0].errorbar(index[0]+0.8,np.abs(tau_ws_access[1,0]),yerr=conf*error_tau_ws_access[1,0],fmt='o',color='orange',markersize=12,label='ACCESS-ESM1.5')
for i in np.arange(1,np.size(index)+1):
    if sig_tau_ws_access[i,0] == 1:
        ax[1,0].plot(index[i-1]+0.8,np.abs(tau_ws_access[i,0]),'ko',markersize=17)
    ax[1,0].errorbar(index[i-1]+0.8,np.abs(tau_ws_access[i,0]),yerr=conf*error_tau_ws_access[i,0],fmt='o',color='orange',markersize=12)
    
# tau CanESM
ax[1,0].errorbar(index[0]+0.9,np.abs(tau_ws_canesm[1,0]),yerr=conf*error_tau_ws_canesm[1,0],fmt='go',markersize=12,label='CanESM5')
for i in np.arange(1,np.size(index)+1):
    if sig_tau_ws_canesm[i,0] == 1:
        ax[1,0].plot(index[i-1]+0.9,np.abs(tau_ws_canesm[i,0]),'ko',markersize=17)
    ax[1,0].errorbar(index[i-1]+0.9,np.abs(tau_ws_canesm[i,0]),yerr=conf*error_tau_ws_canesm[i,0],fmt='go',markersize=12)
     
# tau CESM
ax[1,0].errorbar(index[0]+1,np.abs(tau_ws_cesm[1,0]),yerr=conf*error_tau_ws_cesm[1,0],fmt='bo',markersize=12,label='CESM2')
for i in np.arange(1,np.size(index)+1):
    if sig_tau_ws_cesm[i,0] == 1:
        ax[1,0].plot(index[i-1]+1,np.abs(tau_ws_cesm[i,0]),'ko',markersize=17)
    ax[1,0].errorbar(index[i-1]+1,np.abs(tau_ws_cesm[i,0]),yerr=conf*error_tau_ws_cesm[i,0],fmt='bo',markersize=12)
 
# tau EC-Earth
ax[1,0].errorbar(index[0]+1.1,np.abs(tau_ws_ecearth[1,0]),yerr=conf*error_tau_ws_ecearth[1,0],fmt='ro',markersize=12,label='EC-Earth3')
for i in np.arange(1,np.size(index)+1):
    if sig_tau_ws_ecearth[i,0] == 1:
        ax[1,0].plot(index[i-1]+1.1,np.abs(tau_ws_ecearth[i,0]),'ko',markersize=17)
    ax[1,0].errorbar(index[i-1]+1.1,np.abs(tau_ws_ecearth[i,0]),yerr=conf*error_tau_ws_ecearth[i,0],fmt='ro',markersize=12)

# tau MPI
ax[1,0].errorbar(index[0]+1.2,np.abs(tau_ws_mpi[1,0]),yerr=conf*error_tau_ws_mpi[1,0],fmt='o',color='gray',markersize=12,label='MPI-ESM1.2-LR')
for i in np.arange(1,np.size(index)+1):
    if sig_tau_ws_mpi[i,0] == 1:
        ax[1,0].plot(index[i-1]+1.2,np.abs(tau_ws_mpi[i,0]),'ko',markersize=17)
    ax[1,0].errorbar(index[i-1]+1.2,np.abs(tau_ws_mpi[i,0]),yerr=conf*error_tau_ws_mpi[i,0],fmt='o',color='gray',markersize=12)

# tau obs.
ax1 = ax[1,0].errorbar(index[0]+1.3,np.abs(tau_ws_obs[1,0]),yerr=conf_obs*error_tau_ws_obs[1,0],fmt='x',color='black',markersize=12,label='Obs. (1982-2023)')
ax1[-1][0].set_linestyle('--')
for i in np.arange(1,np.size(index)+1):
    if sig_tau_ws_obs[i,0] == 1:
        ax[1,0].plot(index[i-1]+1.3,np.abs(tau_ws_obs[i,0]),'ko',markersize=17,fillstyle='none')
    ax1 = ax[1,0].errorbar(index[i-1]+1.3,np.abs(tau_ws_obs[i,0]),yerr=conf_obs*error_tau_ws_obs[i,0],fmt='x',color='black',markersize=12)
    ax1[-1][0].set_linestyle('--')

# Labels and legend
ax[1,0].set_ylabel(r'Information transfer $|\tau|$ ($\%$)',fontsize=20)
ax[1,0].tick_params(axis='both',labelsize=16)
ax[1,0].set_xticks(np.arange(1,np.size(index)+1))
ax[1,0].set_xticklabels(label_names)
ax[1,0].grid(linestyle='--')
ax[1,0].set_ylim(-2,30)
ax[1,0].axes.axhline(y=0,color='k',linestyle='--')
ax[1,0].set_title('c',loc='left',fontsize=25,fontweight='bold')
ax[1,0].set_title('Weddell Sea',loc='center',fontsize=25,fontweight='bold')

######
# IO #
######

# tau ACCESS
ax[1,1].errorbar(index[0]+0.8,np.abs(tau_io_access[1,0]),yerr=conf*error_tau_io_access[1,0],fmt='o',color='orange',markersize=12,label='ACCESS-ESM1.5')
for i in np.arange(1,np.size(index)+1):
    if sig_tau_io_access[i,0] == 1:
        ax[1,1].plot(index[i-1]+0.8,np.abs(tau_io_access[i,0]),'ko',markersize=17)
    ax[1,1].errorbar(index[i-1]+0.8,np.abs(tau_io_access[i,0]),yerr=conf*error_tau_io_access[i,0],fmt='o',color='orange',markersize=12)
    
# tau CanESM
ax[1,1].errorbar(index[0]+0.9,np.abs(tau_io_canesm[1,0]),yerr=conf*error_tau_io_canesm[1,0],fmt='go',markersize=12,label='CanESM5')
for i in np.arange(1,np.size(index)+1):
    if sig_tau_io_canesm[i,0] == 1:
        ax[1,1].plot(index[i-1]+0.9,np.abs(tau_io_canesm[i,0]),'ko',markersize=17)
    ax[1,1].errorbar(index[i-1]+0.9,np.abs(tau_io_canesm[i,0]),yerr=conf*error_tau_io_canesm[i,0],fmt='go',markersize=12)
    
# tau CESM
ax[1,1].errorbar(index[0]+1,np.abs(tau_io_cesm[1,0]),yerr=conf*error_tau_io_cesm[1,0],fmt='bo',markersize=12,label='CESM2')
for i in np.arange(1,np.size(index)+1):
    if sig_tau_io_cesm[i,0] == 1:
        ax[1,1].plot(index[i-1]+1,np.abs(tau_io_cesm[i,0]),'ko',markersize=17)
    ax[1,1].errorbar(index[i-1]+1,np.abs(tau_io_cesm[i,0]),yerr=conf*error_tau_io_cesm[i,0],fmt='bo',markersize=12)
 
# tau EC-Earth
ax[1,1].errorbar(index[0]+1.1,np.abs(tau_io_ecearth[1,0]),yerr=conf*error_tau_io_ecearth[1,0],fmt='ro',markersize=12,label='EC-Earth3')
for i in np.arange(1,np.size(index)+1):
    if sig_tau_io_ecearth[i,0] == 1:
        ax[1,1].plot(index[i-1]+1.1,np.abs(tau_io_ecearth[i,0]),'ko',markersize=17)
    ax[1,1].errorbar(index[i-1]+1.1,np.abs(tau_io_ecearth[i,0]),yerr=conf*error_tau_io_ecearth[i,0],fmt='ro',markersize=12)

# tau MPI
ax[1,1].errorbar(index[0]+1.2,np.abs(tau_io_mpi[1,0]),yerr=conf*error_tau_io_mpi[1,0],fmt='o',color='gray',markersize=12,label='MPI-ESM1.2-LR')
for i in np.arange(1,np.size(index)+1):
    if sig_tau_io_mpi[i,0] == 1:
        ax[1,1].plot(index[i-1]+1.2,np.abs(tau_io_mpi[i,0]),'ko',markersize=17)
    ax[1,1].errorbar(index[i-1]+1.2,np.abs(tau_io_mpi[i,0]),yerr=conf*error_tau_io_mpi[i,0],fmt='o',color='gray',markersize=12)

# tau obs.
ax1 = ax[1,1].errorbar(index[0]+1.3,np.abs(tau_io_obs[1,0]),yerr=conf_obs*error_tau_io_obs[1,0],fmt='x',color='black',markersize=12,label='Obs. (1982-2023)')
ax1[-1][0].set_linestyle('--')
for i in np.arange(1,np.size(index)+1):
    if sig_tau_io_obs[i,0] == 1:
        ax[1,1].plot(index[i-1]+1.3,np.abs(tau_io_obs[i,0]),'ko',markersize=17,fillstyle='none')
    ax1 = ax[1,1].errorbar(index[i-1]+1.3,np.abs(tau_io_obs[i,0]),yerr=conf_obs*error_tau_io_obs[i,0],fmt='x',color='black',markersize=12)
    ax1[-1][0].set_linestyle('--')

# Labels and legend
ax[1,1].set_ylabel(r'Information transfer $|\tau|$ ($\%$)',fontsize=20)
ax[1,1].tick_params(axis='both',labelsize=16)
ax[1,1].set_xticks(np.arange(1,np.size(index)+1))
ax[1,1].set_xticklabels(label_names)
ax[1,1].grid(linestyle='--')
ax[1,1].set_ylim(-2,30)
ax[1,1].axes.axhline(y=0,color='k',linestyle='--')
ax[1,1].set_title('d',loc='left',fontsize=25,fontweight='bold')
ax[1,1].set_title('Indian Ocean',loc='center',fontsize=25,fontweight='bold')

#######
# WPO #
#######

# tau ACCESS
ax[2,0].errorbar(index[0]+0.8,np.abs(tau_wpo_access[1,0]),yerr=conf*error_tau_wpo_access[1,0],fmt='o',color='orange',markersize=12,label='ACCESS-ESM1.5')
for i in np.arange(1,np.size(index)+1):
    if sig_tau_wpo_access[i,0] == 1:
        ax[2,0].plot(index[i-1]+0.8,np.abs(tau_wpo_access[i,0]),'ko',markersize=17)
    ax[2,0].errorbar(index[i-1]+0.8,np.abs(tau_wpo_access[i,0]),yerr=conf*error_tau_wpo_access[i,0],fmt='o',color='orange',markersize=12)
    
# tau CanESM
ax[2,0].errorbar(index[0]+0.9,np.abs(tau_wpo_canesm[1,0]),yerr=conf*error_tau_wpo_canesm[1,0],fmt='go',markersize=12,label='CanESM5')
for i in np.arange(1,np.size(index)+1):
    if sig_tau_wpo_canesm[i,0] == 1:
        ax[2,0].plot(index[i-1]+0.9,np.abs(tau_wpo_canesm[i,0]),'ko',markersize=17)
    ax[2,0].errorbar(index[i-1]+0.9,np.abs(tau_wpo_canesm[i,0]),yerr=conf*error_tau_wpo_canesm[i,0],fmt='go',markersize=12)
    
# tau CESM
ax[2,0].errorbar(index[0]+1,np.abs(tau_wpo_cesm[1,0]),yerr=conf*error_tau_wpo_cesm[1,0],fmt='bo',markersize=12,label='CESM2')
for i in np.arange(1,np.size(index)+1):
    if sig_tau_wpo_cesm[i,0] == 1:
        ax[2,0].plot(index[i-1]+1,np.abs(tau_wpo_cesm[i,0]),'ko',markersize=17)
    ax[2,0].errorbar(index[i-1]+1,np.abs(tau_wpo_cesm[i,0]),yerr=conf*error_tau_wpo_cesm[i,0],fmt='bo',markersize=12)
 
# tau EC-Earth
ax[2,0].errorbar(index[0]+1.1,np.abs(tau_wpo_ecearth[1,0]),yerr=conf*error_tau_wpo_ecearth[1,0],fmt='ro',markersize=12,label='EC-Earth3')
for i in np.arange(1,np.size(index)+1):
    if sig_tau_wpo_ecearth[i,0] == 1:
        ax[2,0].plot(index[i-1]+1.1,np.abs(tau_wpo_ecearth[i,0]),'ko',markersize=17)
    ax[2,0].errorbar(index[i-1]+1.1,np.abs(tau_wpo_ecearth[i,0]),yerr=conf*error_tau_wpo_ecearth[i,0],fmt='ro',markersize=12)

# tau MPI
ax[2,0].errorbar(index[0]+1.2,np.abs(tau_wpo_mpi[1,0]),yerr=conf*error_tau_wpo_mpi[1,0],fmt='o',color='gray',markersize=12,label='MPI-ESM1.2-LR')
for i in np.arange(1,np.size(index)+1):
    if sig_tau_wpo_mpi[i,0] == 1:
        ax[2,0].plot(index[i-1]+1.2,np.abs(tau_wpo_mpi[i,0]),'ko',markersize=17)
    ax[2,0].errorbar(index[i-1]+1.2,np.abs(tau_wpo_mpi[i,0]),yerr=conf*error_tau_wpo_mpi[i,0],fmt='o',color='gray',markersize=12)

# tau obs.
ax1 = ax[2,0].errorbar(index[0]+1.3,np.abs(tau_wpo_obs[1,0]),yerr=conf_obs*error_tau_wpo_obs[1,0],fmt='x',color='black',markersize=12,label='Obs. (1982-2023)')
ax1[-1][0].set_linestyle('--')
for i in np.arange(1,np.size(index)+1):
    if sig_tau_wpo_obs[i,0] == 1:
        ax[2,0].plot(index[i-1]+1.3,np.abs(tau_wpo_obs[i,0]),'ko',markersize=17,fillstyle='none')
    ax1 = ax[2,0].errorbar(index[i-1]+1.3,np.abs(tau_wpo_obs[i,0]),yerr=conf_obs*error_tau_wpo_obs[i,0],fmt='x',color='black',markersize=12)
    ax1[-1][0].set_linestyle('--')

# Labels and legend
ax[2,0].set_ylabel(r'Information transfer $|\tau|$ ($\%$)',fontsize=20)
ax[2,0].tick_params(axis='both',labelsize=16)
ax[2,0].set_xticks(np.arange(1,np.size(index)+1))
ax[2,0].set_xticklabels(label_names)
ax[2,0].grid(linestyle='--')
ax[2,0].set_ylim(-2,30)
ax[2,0].axes.axhline(y=0,color='k',linestyle='--')
ax[2,0].set_title('e',loc='left',fontsize=25,fontweight='bold')
ax[2,0].set_title('Western Pacific Ocean',loc='center',fontsize=25,fontweight='bold')

######
# RS #
######

# tau ACCESS
ax[2,1].errorbar(index[0]+0.8,np.abs(tau_rs_access[1,0]),yerr=conf*error_tau_rs_access[1,0],fmt='o',color='orange',markersize=12,label='ACCESS-ESM1.5')
for i in np.arange(1,np.size(index)+1):
    if sig_tau_rs_access[i,0] == 1:
        ax[2,1].plot(index[i-1]+0.8,np.abs(tau_rs_access[i,0]),'ko',markersize=17)
    ax[2,1].errorbar(index[i-1]+0.8,np.abs(tau_rs_access[i,0]),yerr=conf*error_tau_rs_access[i,0],fmt='o',color='orange',markersize=12)
    
# tau CanESM
ax[2,1].errorbar(index[0]+0.9,np.abs(tau_rs_canesm[1,0]),yerr=conf*error_tau_rs_canesm[1,0],fmt='go',markersize=12,label='CanESM5')
for i in np.arange(1,np.size(index)+1):
    if sig_tau_rs_canesm[i,0] == 1:
        ax[2,1].plot(index[i-1]+0.9,np.abs(tau_rs_canesm[i,0]),'ko',markersize=17)
    ax[2,1].errorbar(index[i-1]+0.9,np.abs(tau_rs_canesm[i,0]),yerr=conf*error_tau_rs_canesm[i,0],fmt='go',markersize=12)
    
# tau CESM
ax[2,1].errorbar(index[0]+1,np.abs(tau_rs_cesm[1,0]),yerr=conf*error_tau_rs_cesm[1,0],fmt='bo',markersize=12,label='CESM2')
for i in np.arange(1,np.size(index)+1):
    if sig_tau_rs_cesm[i,0] == 1:
        ax[2,1].plot(index[i-1]+1,np.abs(tau_rs_cesm[i,0]),'ko',markersize=17)
    ax[2,1].errorbar(index[i-1]+1,np.abs(tau_rs_cesm[i,0]),yerr=conf*error_tau_rs_cesm[i,0],fmt='bo',markersize=12)
 
# tau EC-Earth
ax[2,1].errorbar(index[0]+1.1,np.abs(tau_rs_ecearth[1,0]),yerr=conf*error_tau_rs_ecearth[1,0],fmt='ro',markersize=12,label='EC-Earth3')
for i in np.arange(1,np.size(index)+1):
    if sig_tau_rs_ecearth[i,0] == 1:
        ax[2,1].plot(index[i-1]+1.1,np.abs(tau_rs_ecearth[i,0]),'ko',markersize=17)
    ax[2,1].errorbar(index[i-1]+1.1,np.abs(tau_rs_ecearth[i,0]),yerr=conf*error_tau_rs_ecearth[i,0],fmt='ro',markersize=12)

# tau MPI
ax[2,1].errorbar(index[0]+1.2,np.abs(tau_rs_mpi[1,0]),yerr=conf*error_tau_rs_mpi[1,0],fmt='o',color='gray',markersize=12,label='MPI-ESM1.2-LR')
for i in np.arange(1,np.size(index)+1):
    if sig_tau_rs_mpi[i,0] == 1:
        ax[2,1].plot(index[i-1]+1.2,np.abs(tau_rs_mpi[i,0]),'ko',markersize=17)
    ax[2,1].errorbar(index[i-1]+1.2,np.abs(tau_rs_mpi[i,0]),yerr=conf*error_tau_rs_mpi[i,0],fmt='o',color='gray',markersize=12)

# tau obs.
ax1 = ax[2,1].errorbar(index[0]+1.3,np.abs(tau_rs_obs[1,0]),yerr=conf_obs*error_tau_rs_obs[1,0],fmt='x',color='black',markersize=12,label='Obs. (1982-2023)')
ax1[-1][0].set_linestyle('--')
for i in np.arange(1,np.size(index)+1):
    if sig_tau_rs_obs[i,0] == 1:
        ax[2,1].plot(index[i-1]+1.3,np.abs(tau_rs_obs[i,0]),'ko',markersize=17,fillstyle='none')
    ax1 = ax[2,1].errorbar(index[i-1]+1.3,np.abs(tau_rs_obs[i,0]),yerr=conf_obs*error_tau_rs_obs[i,0],fmt='x',color='black',markersize=12)
    ax1[-1][0].set_linestyle('--')

# Labels and legend
ax[2,1].set_ylabel(r'Information transfer $|\tau|$ ($\%$)',fontsize=20)
ax[2,1].tick_params(axis='both',labelsize=16)
ax[2,1].legend(loc='upper right',fontsize=14,shadow=True,frameon=False)
ax[2,1].set_xticks(np.arange(1,np.size(index)+1))
ax[2,1].set_xticklabels(label_names)
ax[2,1].grid(linestyle='--')
ax[2,1].set_ylim(-2,30)
ax[2,1].axes.axhline(y=0,color='k',linestyle='--')
ax[2,1].set_title('f',loc='left',fontsize=25,fontweight='bold')
ax[2,1].set_title('Ross Sea',loc='center',fontsize=25,fontweight='bold')

# Save figure
if save_fig == True:
    if season == 'OND':
        filename = dir_fig + 'fig7.pdf'
    if season == 'SON':
        filename = dir_fig + 'fig7b.pdf'
    fig.savefig(filename)


# Figure of correlations (R)
fig,ax = plt.subplots(3,2,figsize=(18,15))
fig.subplots_adjust(left=0.11,bottom=0.08,right=0.94,top=0.93,hspace=0.4,wspace=0.25)

#############
# Antarctic #
#############

# R ACCESS
ax[0,0].errorbar(index[0]+0.8,R_access[1,0],yerr=conf*error_R_access[1,0],fmt='o',color='orange',markersize=12,label='ACCESS-ESM1.5')
for i in np.arange(1,np.size(index)+1):
    if sig_R_access[i,0] == 1:
        ax[0,0].plot(index[i-1]+0.8,R_access[i,0],'ko',markersize=17)
    ax[0,0].errorbar(index[i-1]+0.8,R_access[i,0],yerr=conf*error_R_access[i,0],fmt='o',color='orange',markersize=12)
    
# R CanESM
ax[0,0].errorbar(index[0]+0.9,R_canesm[1,0],yerr=conf*error_R_canesm[1,0],fmt='go',markersize=12,label='CanESM5')
for i in np.arange(1,np.size(index)+1):
    if sig_R_canesm[i,0] == 1:
        ax[0,0].plot(index[i-1]+0.9,R_canesm[i,0],'ko',markersize=17)
    ax[0,0].errorbar(index[i-1]+0.9,R_canesm[i,0],yerr=conf*error_R_canesm[i,0],fmt='go',markersize=12)

# R CESM
ax[0,0].errorbar(index[0]+1,R_cesm[1,0],yerr=conf*error_R_cesm[1,0],fmt='bo',markersize=12,label='CESM2')
for i in np.arange(1,np.size(index)+1):
    if sig_R_cesm[i,0] == 1:
        ax[0,0].plot(index[i-1]+1,R_cesm[i,0],'ko',markersize=17)
    ax1=ax[0,0].errorbar(index[i-1]+1,R_cesm[i,0],yerr=conf*error_R_cesm[i,0],fmt='bo',markersize=12)

# R EC-Earth
ax[0,0].errorbar(index[0]+1.1,R_ecearth[1,0],yerr=conf*error_R_ecearth[1,0],fmt='ro',markersize=12,label='EC-Earth3')
for i in np.arange(1,np.size(index)+1):
    if sig_R_ecearth[i,0] == 1:
        ax[0,0].plot(index[i-1]+1.1,R_ecearth[i,0],'ko',markersize=17)
    ax[0,0].errorbar(index[i-1]+1.1,R_ecearth[i,0],yerr=conf*error_R_ecearth[i,0],fmt='ro',markersize=12)

# R MPI
ax[0,0].errorbar(index[0]+1.2,R_mpi[1,0],yerr=conf*error_R_mpi[1,0],fmt='o',color='gray',markersize=12,label='MPI-ESM1.2-LR')
for i in np.arange(1,np.size(index)+1):
    if sig_R_mpi[i,0] == 1:
        ax[0,0].plot(index[i-1]+1.2,R_mpi[i,0],'ko',markersize=17)
    ax[0,0].errorbar(index[i-1]+1.2,R_mpi[i,0],yerr=conf*error_R_mpi[i,0],fmt='o',color='gray',markersize=12)

# R obs.
ax1 = ax[0,0].errorbar(index[0]+1.3,R_obs[1,0],yerr=conf_obs*error_R_obs[1,0],fmt='x',color='black',markersize=12,label='Obs. (1982-2023)')
ax1[-1][0].set_linestyle('--')
for i in np.arange(1,np.size(index)+1):
    if sig_R_obs[i,0] == 1:
        ax[0,0].plot(index[i-1]+1.3,R_obs[i,0],'ko',markersize=17,fillstyle='none')
    ax1 = ax[0,0].errorbar(index[i-1]+1.3,R_obs[i,0],yerr=conf_obs*error_R_obs[i,0],fmt='x',color='black',markersize=12)
    ax1[-1][0].set_linestyle('--')

# Labels and legend
ax[0,0].set_ylabel('Correlation coefficient $R$',fontsize=20)
ax[0,0].tick_params(axis='both',labelsize=16)
ax[0,0].legend(loc='upper right',fontsize=14,shadow=True,frameon=False,ncol=2)
ax[0,0].set_xticks(np.arange(1,np.size(index)+1))
ax[0,0].set_xticklabels(label_names)
ax[0,0].grid(linestyle='--')
ax[0,0].set_ylim(-1,1)
ax[0,0].axes.axhline(y=0,color='k',linestyle='--')
ax[0,0].set_title('a',loc='left',fontsize=25,fontweight='bold')
ax[0,0].set_title('Pan-Antarctic',loc='center',fontsize=25,fontweight='bold')

#######
# BAS #
#######

# R ACCESS
ax[0,1].errorbar(index[0]+0.8,R_bas_access[1,0],yerr=conf*error_R_bas_access[1,0],fmt='o',color='orange',markersize=12,label='ACCESS-ESM1.5')
for i in np.arange(1,np.size(index)+1):
    if sig_R_bas_access[i,0] == 1:
        ax[0,1].plot(index[i-1]+0.8,R_bas_access[i,0],'ko',markersize=17)
    ax[0,1].errorbar(index[i-1]+0.8,R_bas_access[i,0],yerr=conf*error_R_bas_access[i,0],fmt='o',color='orange',markersize=12)
    
# R CanESM
ax[0,1].errorbar(index[0]+0.9,R_bas_canesm[1,0],yerr=conf*error_R_bas_canesm[1,0],fmt='go',markersize=12,label='CanESM5')
for i in np.arange(1,np.size(index)+1):
    if sig_R_bas_canesm[i,0] == 1:
        ax[0,1].plot(index[i-1]+0.9,R_bas_canesm[i,0],'ko',markersize=17)
    ax[0,1].errorbar(index[i-1]+0.9,R_bas_canesm[i,0],yerr=conf*error_R_bas_canesm[i,0],fmt='go',markersize=12)
    
# R CESM
ax[0,1].errorbar(index[0]+1,R_bas_cesm[1,0],yerr=conf*error_R_bas_cesm[1,0],fmt='bo',markersize=12,label='CESM2')
for i in np.arange(1,np.size(index)+1):
    if sig_R_bas_cesm[i,0] == 1:
        ax[0,1].plot(index[i-1]+1,R_bas_cesm[i,0],'ko',markersize=17)
    ax[0,1].errorbar(index[i-1]+1,R_bas_cesm[i,0],yerr=conf*error_R_bas_cesm[i,0],fmt='bo',markersize=12)
 
# R EC-Earth
ax[0,1].errorbar(index[0]+1.1,R_bas_ecearth[1,0],yerr=conf*error_R_bas_ecearth[1,0],fmt='ro',markersize=12,label='EC-Earth3')
for i in np.arange(1,np.size(index)+1):
    if sig_R_bas_ecearth[i,0] == 1:
        ax[0,1].plot(index[i-1]+1.1,R_bas_ecearth[i,0],'ko',markersize=17)
    ax[0,1].errorbar(index[i-1]+1.1,R_bas_ecearth[i,0],yerr=conf*error_R_bas_ecearth[i,0],fmt='ro',markersize=12)

# R MPI
ax[0,1].errorbar(index[0]+1.2,R_bas_mpi[1,0],yerr=conf*error_R_bas_mpi[1,0],fmt='o',color='gray',markersize=12,label='MPI-ESM1.2-LR')
for i in np.arange(1,np.size(index)+1):
    if sig_R_bas_mpi[i,0] == 1:
        ax[0,1].plot(index[i-1]+1.2,R_bas_mpi[i,0],'ko',markersize=17)
    ax[0,1].errorbar(index[i-1]+1.2,R_bas_mpi[i,0],yerr=conf*error_R_bas_mpi[i,0],fmt='o',color='gray',markersize=12)

# R obs.
ax1 = ax[0,1].errorbar(index[0]+1.3,R_bas_obs[1,0],yerr=conf_obs*error_R_bas_obs[1,0],fmt='x',color='black',markersize=12,label='Obs. (1982-2023)')
ax1[-1][0].set_linestyle('--')
for i in np.arange(1,np.size(index)+1):
    if sig_R_bas_obs[i,0] == 1:
        ax[0,1].plot(index[i-1]+1.3,R_bas_obs[i,0],'ko',markersize=17,fillstyle='none')
    ax1 = ax[0,1].errorbar(index[i-1]+1.3,R_bas_obs[i,0],yerr=conf_obs*error_R_bas_obs[i,0],fmt='x',color='black',markersize=12)
    ax1[-1][0].set_linestyle('--')

# Labels and legend
ax[0,1].set_ylabel('Correlation coefficient $R$',fontsize=20)
ax[0,1].tick_params(axis='both',labelsize=16)
ax[0,1].set_xticks(np.arange(1,np.size(index)+1))
ax[0,1].set_xticklabels(label_names)
ax[0,1].grid(linestyle='--')
ax[0,1].set_ylim(-1,1)
ax[0,1].axes.axhline(y=0,color='k',linestyle='--')
ax[0,1].set_title('b',loc='left',fontsize=25,fontweight='bold')
ax[0,1].set_title('Bellingshausen-Amundsen',loc='center',fontsize=25,fontweight='bold')

######
# WS #
######

# R ACCESS
ax[1,0].errorbar(index[0]+0.8,R_ws_access[1,0],yerr=conf*error_R_ws_access[1,0],fmt='o',color='orange',markersize=12,label='ACCESS-ESM1.5')
for i in np.arange(1,np.size(index)+1):
    if sig_R_ws_access[i,0] == 1:
        ax[1,0].plot(index[i-1]+0.8,R_ws_access[i,0],'ko',markersize=17)
    ax[1,0].errorbar(index[i-1]+0.8,R_ws_access[i,0],yerr=conf*error_R_ws_access[i,0],fmt='o',color='orange',markersize=12)
    
# R CanESM
ax[1,0].errorbar(index[0]+0.9,R_ws_canesm[1,0],yerr=conf*error_R_ws_canesm[1,0],fmt='go',markersize=12,label='CanESM5')
for i in np.arange(1,np.size(index)+1):
    if sig_R_ws_canesm[i,0] == 1:
        ax[1,0].plot(index[i-1]+0.9,R_ws_canesm[i,0],'ko',markersize=17)
    ax[1,0].errorbar(index[i-1]+0.9,R_ws_canesm[i,0],yerr=conf*error_R_ws_canesm[i,0],fmt='go',markersize=12)
     
# R CESM
ax[1,0].errorbar(index[0]+1,R_ws_cesm[1,0],yerr=conf*error_R_ws_cesm[1,0],fmt='bo',markersize=12,label='CESM2')
for i in np.arange(1,np.size(index)+1):
    if sig_R_ws_cesm[i,0] == 1:
        ax[1,0].plot(index[i-1]+1,R_ws_cesm[i,0],'ko',markersize=17)
    ax[1,0].errorbar(index[i-1]+1,R_ws_cesm[i,0],yerr=conf*error_R_ws_cesm[i,0],fmt='bo',markersize=12)
 
# R EC-Earth
ax[1,0].errorbar(index[0]+1.1,R_ws_ecearth[1,0],yerr=conf*error_R_ws_ecearth[1,0],fmt='ro',markersize=12,label='EC-Earth3')
for i in np.arange(1,np.size(index)+1):
    if sig_R_ws_ecearth[i,0] == 1:
        ax[1,0].plot(index[i-1]+1.1,R_ws_ecearth[i,0],'ko',markersize=17)
    ax[1,0].errorbar(index[i-1]+1.1,R_ws_ecearth[i,0],yerr=conf*error_R_ws_ecearth[i,0],fmt='ro',markersize=12)

# R MPI
ax[1,0].errorbar(index[0]+1.2,R_ws_mpi[1,0],yerr=conf*error_R_ws_mpi[1,0],fmt='o',color='gray',markersize=12,label='MPI-ESM1.2-LR')
for i in np.arange(1,np.size(index)+1):
    if sig_R_ws_mpi[i,0] == 1:
        ax[1,0].plot(index[i-1]+1.2,R_ws_mpi[i,0],'ko',markersize=17)
    ax[1,0].errorbar(index[i-1]+1.2,R_ws_mpi[i,0],yerr=conf*error_R_ws_mpi[i,0],fmt='o',color='gray',markersize=12)

# R obs.
ax1 = ax[1,0].errorbar(index[0]+1.3,R_ws_obs[1,0],yerr=conf_obs*error_R_ws_obs[1,0],fmt='x',color='black',markersize=12,label='Obs. (1982-2023)')
ax1[-1][0].set_linestyle('--')
for i in np.arange(1,np.size(index)+1):
    if sig_R_ws_obs[i,0] == 1:
        ax[1,0].plot(index[i-1]+1.3,R_ws_obs[i,0],'ko',markersize=17,fillstyle='none')
    ax1 = ax[1,0].errorbar(index[i-1]+1.3,R_ws_obs[i,0],yerr=conf_obs*error_R_ws_obs[i,0],fmt='x',color='black',markersize=12)
    ax1[-1][0].set_linestyle('--')

# Labels and legend
ax[1,0].set_ylabel('Correlation coefficient $R$',fontsize=20)
ax[1,0].tick_params(axis='both',labelsize=16)
ax[1,0].set_xticks(np.arange(1,np.size(index)+1))
ax[1,0].set_xticklabels(label_names)
ax[1,0].grid(linestyle='--')
ax[1,0].set_ylim(-1,1)
ax[1,0].axes.axhline(y=0,color='k',linestyle='--')
ax[1,0].set_title('c',loc='left',fontsize=25,fontweight='bold')
ax[1,0].set_title('Weddell Sea',loc='center',fontsize=25,fontweight='bold')

######
# IO #
######

# R ACCESS
ax[1,1].errorbar(index[0]+0.8,R_io_access[1,0],yerr=conf*error_R_io_access[1,0],fmt='o',color='orange',markersize=12,label='ACCESS-ESM1.5')
for i in np.arange(1,np.size(index)+1):
    if sig_R_io_access[i,0] == 1:
        ax[1,1].plot(index[i-1]+0.8,R_io_access[i,0],'ko',markersize=17)
    ax[1,1].errorbar(index[i-1]+0.8,R_io_access[i,0],yerr=conf*error_R_io_access[i,0],fmt='o',color='orange',markersize=12)
    
# R CanESM
ax[1,1].errorbar(index[0]+0.9,R_io_canesm[1,0],yerr=conf*error_R_io_canesm[1,0],fmt='go',markersize=12,label='CanESM5')
for i in np.arange(1,np.size(index)+1):
    if sig_R_io_canesm[i,0] == 1:
        ax[1,1].plot(index[i-1]+0.9,R_io_canesm[i,0],'ko',markersize=17)
    ax[1,1].errorbar(index[i-1]+0.9,R_io_canesm[i,0],yerr=conf*error_R_io_canesm[i,0],fmt='go',markersize=12)
    
# R CESM
ax[1,1].errorbar(index[0]+1,R_io_cesm[1,0],yerr=conf*error_R_io_cesm[1,0],fmt='bo',markersize=12,label='CESM2')
for i in np.arange(1,np.size(index)+1):
    if sig_R_io_cesm[i,0] == 1:
        ax[1,1].plot(index[i-1]+1,R_io_cesm[i,0],'ko',markersize=17)
    ax[1,1].errorbar(index[i-1]+1,R_io_cesm[i,0],yerr=conf*error_R_io_cesm[i,0],fmt='bo',markersize=12)
 
# R EC-Earth
ax[1,1].errorbar(index[0]+1.1,R_io_ecearth[1,0],yerr=conf*error_R_io_ecearth[1,0],fmt='ro',markersize=12,label='EC-Earth3')
for i in np.arange(1,np.size(index)+1):
    if sig_R_io_ecearth[i,0] == 1:
        ax[1,1].plot(index[i-1]+1.1,R_io_ecearth[i,0],'ko',markersize=17)
    ax[1,1].errorbar(index[i-1]+1.1,R_io_ecearth[i,0],yerr=conf*error_R_io_ecearth[i,0],fmt='ro',markersize=12)

# R MPI
ax[1,1].errorbar(index[0]+1.2,R_io_mpi[1,0],yerr=conf*error_R_io_mpi[1,0],fmt='o',color='gray',markersize=12,label='MPI-ESM1.2-LR')
for i in np.arange(1,np.size(index)+1):
    if sig_R_io_mpi[i,0] == 1:
        ax[1,1].plot(index[i-1]+1.2,R_io_mpi[i,0],'ko',markersize=17)
    ax[1,1].errorbar(index[i-1]+1.2,R_io_mpi[i,0],yerr=conf*error_R_io_mpi[i,0],fmt='o',color='gray',markersize=12)

# R obs.
ax1 = ax[1,1].errorbar(index[0]+1.3,R_io_obs[1,0],yerr=conf_obs*error_R_io_obs[1,0],fmt='x',color='black',markersize=12,label='Obs. (1982-2023)')
ax1[-1][0].set_linestyle('--')
for i in np.arange(1,np.size(index)+1):
    if sig_R_io_obs[i,0] == 1:
        ax[1,1].plot(index[i-1]+1.3,R_io_obs[i,0],'ko',markersize=17,fillstyle='none')
    ax1 = ax[1,1].errorbar(index[i-1]+1.3,R_io_obs[i,0],yerr=conf_obs*error_R_io_obs[i,0],fmt='x',color='black',markersize=12)
    ax1[-1][0].set_linestyle('--')

# Labels and legend
ax[1,1].set_ylabel('Correlation coefficient $R$',fontsize=20)
ax[1,1].tick_params(axis='both',labelsize=16)
ax[1,1].set_xticks(np.arange(1,np.size(index)+1))
ax[1,1].set_xticklabels(label_names)
ax[1,1].grid(linestyle='--')
ax[1,1].set_ylim(-1,1)
ax[1,1].axes.axhline(y=0,color='k',linestyle='--')
ax[1,1].set_title('d',loc='left',fontsize=25,fontweight='bold')
ax[1,1].set_title('Indian Ocean',loc='center',fontsize=25,fontweight='bold')

#######
# WPO #
#######

# R ACCESS
ax[2,0].errorbar(index[0]+0.8,R_wpo_access[1,0],yerr=conf*error_R_wpo_access[1,0],fmt='o',color='orange',markersize=12,label='ACCESS-ESM1.5')
for i in np.arange(1,np.size(index)+1):
    if sig_R_wpo_access[i,0] == 1:
        ax[2,0].plot(index[i-1]+0.8,R_wpo_access[i,0],'ko',markersize=17)
    ax[2,0].errorbar(index[i-1]+0.8,R_wpo_access[i,0],yerr=conf*error_R_wpo_access[i,0],fmt='o',color='orange',markersize=12)
    
# R CanESM
ax[2,0].errorbar(index[0]+0.9,R_wpo_canesm[1,0],yerr=conf*error_R_wpo_canesm[1,0],fmt='go',markersize=12,label='CanESM5')
for i in np.arange(1,np.size(index)+1):
    if sig_R_wpo_canesm[i,0] == 1:
        ax[2,0].plot(index[i-1]+0.9,R_wpo_canesm[i,0],'ko',markersize=17)
    ax[2,0].errorbar(index[i-1]+0.9,R_wpo_canesm[i,0],yerr=conf*error_R_wpo_canesm[i,0],fmt='go',markersize=12)
    
# R CESM
ax[2,0].errorbar(index[0]+1,R_wpo_cesm[1,0],yerr=conf*error_R_wpo_cesm[1,0],fmt='bo',markersize=12,label='CESM2')
for i in np.arange(1,np.size(index)+1):
    if sig_R_wpo_cesm[i,0] == 1:
        ax[2,0].plot(index[i-1]+1,R_wpo_cesm[i,0],'ko',markersize=17)
    ax[2,0].errorbar(index[i-1]+1,R_wpo_cesm[i,0],yerr=conf*error_R_wpo_cesm[i,0],fmt='bo',markersize=12)
 
# R EC-Earth
ax[2,0].errorbar(index[0]+1.1,R_wpo_ecearth[1,0],yerr=conf*error_R_wpo_ecearth[1,0],fmt='ro',markersize=12,label='EC-Earth3')
for i in np.arange(1,np.size(index)+1):
    if sig_R_wpo_ecearth[i,0] == 1:
        ax[2,0].plot(index[i-1]+1.1,R_wpo_ecearth[i,0],'ko',markersize=17)
    ax[2,0].errorbar(index[i-1]+1.1,R_wpo_ecearth[i,0],yerr=conf*error_R_wpo_ecearth[i,0],fmt='ro',markersize=12)

# R MPI
ax[2,0].errorbar(index[0]+1.2,R_wpo_mpi[1,0],yerr=conf*error_R_wpo_mpi[1,0],fmt='o',color='gray',markersize=12,label='MPI-ESM1.2-LR')
for i in np.arange(1,np.size(index)+1):
    if sig_R_wpo_mpi[i,0] == 1:
        ax[2,0].plot(index[i-1]+1.2,R_wpo_mpi[i,0],'ko',markersize=17)
    ax[2,0].errorbar(index[i-1]+1.2,R_wpo_mpi[i,0],yerr=conf*error_R_wpo_mpi[i,0],fmt='o',color='gray',markersize=12)

# R obs.
ax1 = ax[2,0].errorbar(index[0]+1.3,R_wpo_obs[1,0],yerr=conf_obs*error_R_wpo_obs[1,0],fmt='x',color='black',markersize=12,label='Obs. (1982-2023)')
ax1[-1][0].set_linestyle('--')
for i in np.arange(1,np.size(index)+1):
    if sig_R_wpo_obs[i,0] == 1:
        ax[2,0].plot(index[i-1]+1.3,R_wpo_obs[i,0],'ko',markersize=17,fillstyle='none')
    ax1 = ax[2,0].errorbar(index[i-1]+1.3,R_wpo_obs[i,0],yerr=conf_obs*error_R_wpo_obs[i,0],fmt='x',color='black',markersize=12)
    ax1[-1][0].set_linestyle('--')

# Labels and legend
ax[2,0].set_ylabel('Correlation coefficient $R$',fontsize=20)
ax[2,0].tick_params(axis='both',labelsize=16)
ax[2,0].set_xticks(np.arange(1,np.size(index)+1))
ax[2,0].set_xticklabels(label_names)
ax[2,0].grid(linestyle='--')
ax[2,0].set_ylim(-1,1)
ax[2,0].axes.axhline(y=0,color='k',linestyle='--')
ax[2,0].set_title('e',loc='left',fontsize=25,fontweight='bold')
ax[2,0].set_title('Western Pacific Ocean',loc='center',fontsize=25,fontweight='bold')

######
# RS #
######

# R ACCESS
ax[2,1].errorbar(index[0]+0.8,R_rs_access[1,0],yerr=conf*error_R_rs_access[1,0],fmt='o',color='orange',markersize=12,label='ACCESS-ESM1.5')
for i in np.arange(1,np.size(index)+1):
    if sig_R_rs_access[i,0] == 1:
        ax[2,1].plot(index[i-1]+0.8,R_rs_access[i,0],'ko',markersize=17)
    ax[2,1].errorbar(index[i-1]+0.8,R_rs_access[i,0],yerr=conf*error_R_rs_access[i,0],fmt='o',color='orange',markersize=12)
    
# R CanESM
ax[2,1].errorbar(index[0]+0.9,R_rs_canesm[1,0],yerr=conf*error_R_rs_canesm[1,0],fmt='go',markersize=12,label='CanESM5')
for i in np.arange(1,np.size(index)+1):
    if sig_R_rs_canesm[i,0] == 1:
        ax[2,1].plot(index[i-1]+0.9,R_rs_canesm[i,0],'ko',markersize=17)
    ax[2,1].errorbar(index[i-1]+0.9,R_rs_canesm[i,0],yerr=conf*error_R_rs_canesm[i,0],fmt='go',markersize=12)
    
# R CESM
ax[2,1].errorbar(index[0]+1,R_rs_cesm[1,0],yerr=conf*error_R_rs_cesm[1,0],fmt='bo',markersize=12,label='CESM2')
for i in np.arange(1,np.size(index)+1):
    if sig_R_rs_cesm[i,0] == 1:
        ax[2,1].plot(index[i-1]+1,R_rs_cesm[i,0],'ko',markersize=17)
    ax[2,1].errorbar(index[i-1]+1,R_rs_cesm[i,0],yerr=conf*error_R_rs_cesm[i,0],fmt='bo',markersize=12)
 
# R EC-Earth
ax[2,1].errorbar(index[0]+1.1,R_rs_ecearth[1,0],yerr=conf*error_R_rs_ecearth[1,0],fmt='ro',markersize=12,label='EC-Earth3')
for i in np.arange(1,np.size(index)+1):
    if sig_R_rs_ecearth[i,0] == 1:
        ax[2,1].plot(index[i-1]+1.1,R_rs_ecearth[i,0],'ko',markersize=17)
    ax[2,1].errorbar(index[i-1]+1.1,R_rs_ecearth[i,0],yerr=conf*error_R_rs_ecearth[i,0],fmt='ro',markersize=12)

# R MPI
ax[2,1].errorbar(index[0]+1.2,R_rs_mpi[1,0],yerr=conf*error_R_rs_mpi[1,0],fmt='o',color='gray',markersize=12,label='MPI-ESM1.2-LR')
for i in np.arange(1,np.size(index)+1):
    if sig_R_rs_mpi[i,0] == 1:
        ax[2,1].plot(index[i-1]+1.2,R_rs_mpi[i,0],'ko',markersize=17)
    ax[2,1].errorbar(index[i-1]+1.2,R_rs_mpi[i,0],yerr=conf*error_R_rs_mpi[i,0],fmt='o',color='gray',markersize=12)

# R obs.
ax1 = ax[2,1].errorbar(index[0]+1.3,R_rs_obs[1,0],yerr=conf_obs*error_R_rs_obs[1,0],fmt='x',color='black',markersize=12,label='Obs. (1982-2023)')
ax1[-1][0].set_linestyle('--')
for i in np.arange(1,np.size(index)+1):
    if sig_R_rs_obs[i,0] == 1:
        ax[2,1].plot(index[i-1]+1.3,R_rs_obs[i,0],'ko',markersize=17,fillstyle='none')
    ax1 = ax[2,1].errorbar(index[i-1]+1.3,R_rs_obs[i,0],yerr=conf_obs*error_R_rs_obs[i,0],fmt='x',color='black',markersize=12)
    ax1[-1][0].set_linestyle('--')

# Labels and legend
ax[2,1].set_ylabel('Correlation coefficient $R$',fontsize=20)
ax[2,1].tick_params(axis='both',labelsize=16)
ax[2,1].set_xticks(np.arange(1,np.size(index)+1))
ax[2,1].set_xticklabels(label_names)
ax[2,1].grid(linestyle='--')
ax[2,1].set_ylim(-1,1)
ax[2,1].axes.axhline(y=0,color='k',linestyle='--')
ax[2,1].set_title('f',loc='left',fontsize=25,fontweight='bold')
ax[2,1].set_title('Ross Sea',loc='center',fontsize=25,fontweight='bold')

# Save figure
if save_fig == True:
    if season == 'OND':
        filename = dir_fig + 'fig_a8.pdf'
    if season == 'SON':
        filename = dir_fig + 'fig_a8b.pdf'
    fig.savefig(filename)