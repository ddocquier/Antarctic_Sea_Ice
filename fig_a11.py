#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fig. A11: Plot Liang index as a function of mean initial state and variability over 1982-2023

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

Last updated: 01/07/2025

@author: David Docquier
"""

# Standard libraries
import numpy as np
import matplotlib.pyplot as plt

# Parameters
first_year = 1982 # starting year included in the computation
last_year = 2023 # last year included in the computation
ind_first_year = int(first_year-1970)
ind_last_year = int(last_year-1970+1)
season = 'OND' # JAS (previous winter), OND (previous spring)
nvar = 8 # number of variables (1: SSIE; 2: PSIE; 3: T_2m; 4: SST; 5: SAM; 6: ASL; 7: ENSO; 8: DMI)
n_iter = 1000 # number of bootstrap realizations; default: 1000
conf = 2.57 # 1.96 if 95% confidence interval (normal distribution); 1.65 if 90% and 2.57 if 99%
save_fig = True

# Working directories
dir_input = '/home/ddocquier/Documents/Papers/My_Papers/RESIST_Antarctic/output/seasons/'
dir_fig = '/home/ddocquier/Documents/Papers/My_Papers/RESIST_Antarctic/LaTeX/'

# Load Liang SMHI-LENS
filename = dir_input + 'Liang_SMHI-LENS_' + season + '_' + str(n_iter) + 'boot_1982-2023.npy'
tau_ecearth,R_ecearth,error_tau_ecearth,error_R_ecearth = np.load(filename,allow_pickle=True)

# Load Liang CESM2-LE
filename = dir_input + 'Liang_CESM2-LE_' + season + '_' + str(n_iter) + 'boot_1982-2023.npy'
tau_cesm,R_cesm,error_tau_cesm,error_R_cesm = np.load(filename,allow_pickle=True)

# Load Liang MPI-ESM1-2-LR
filename = dir_input + 'Liang_MPI-ESM1-2-LR_' + season + '_' + str(n_iter) + 'boot_1982-2023.npy'
tau_mpi,R_mpi,error_tau_mpi,error_R_mpi = np.load(filename,allow_pickle=True)

# Load Liang CanESM5
filename = dir_input + 'Liang_CanESM5_' + season + '_' + str(n_iter) + 'boot_1982-2023.npy'
tau_canesm,R_canesm,error_tau_canesm,error_R_canesm = np.load(filename,allow_pickle=True)

# Load Liang ACCESS-ESM1-5
filename = dir_input + 'Liang_ACCESS-ESM1-5_' + season + '_' + str(n_iter) + 'boot_1982-2023.npy'
tau_access,R_access,error_tau_access,error_R_access = np.load(filename,allow_pickle=True)

# Load Liang Observations
filename = dir_input + 'Liang_obs_' + season + '.npy'
tau_obs,R_obs,error_tau_obs,error_R_obs = np.load(filename,allow_pickle=True)

# Load EC-Earth3
filename = dir_input + 'EC-Earth3_Antarctic_timeseries_' + season + '.npy'
sie_summer_ecearth,sie_previous_ecearth,tas_ecearth,sst_ecearth,sam_ecearth,asl_ecearth,nino34_ecearth,dmi_ecearth = np.load(filename,allow_pickle=True)

# Load CESM2
filename = dir_input + 'CESM2_Antarctic_timeseries_' + season + '.npy'
sie_summer_cesm,sie_previous_cesm,tas_cesm,sst_cesm,sam_cesm,asl_cesm,nino34_cesm,dmi_cesm = np.load(filename,allow_pickle=True)
#
# Load MPI-ESM1-2-LR
filename = dir_input + 'MPI-ESM1-2-LR_Antarctic_timeseries_' + season + '.npy'
sie_summer_mpi,sie_previous_mpi,tas_mpi,sst_mpi,sam_mpi,asl_mpi,nino34_mpi,dmi_mpi = np.load(filename,allow_pickle=True)

# Load CanESM5
filename = dir_input + 'CanESM5_Antarctic_timeseries_' + season + '.npy'
sie_summer_canesm,sie_previous_canesm,tas_canesm,sst_canesm,sam_canesm,asl_canesm,nino34_canesm,dmi_canesm = np.load(filename,allow_pickle=True)

# Load ACCESS-ESM1-5
filename = dir_input + 'ACCESS-ESM1-5_Antarctic_timeseries_' + season + '.npy'
sie_summer_access,sie_previous_access,tas_access,sst_access,sam_access,asl_access,nino34_access,dmi_access = np.load(filename,allow_pickle=True)

# Load observations
filename = dir_input + 'Obs_Antarctic_timeseries_' + season + '.npy'
sie_summer_obs,sie_previous_obs,tas_obs,sst_obs,sam_obs,asl_obs,nino34_obs,dmi_obs = np.load(filename,allow_pickle=True)

# Take 1982-2023 for all EC-Earth3 variables
sie_summer_ecearth = np.roll(sie_summer_ecearth,-1,axis=1) # shift summer SIE to the left (so that it lags other variables)
sie_summer_ecearth = sie_summer_ecearth[:,ind_first_year:ind_last_year]
sie_previous_ecearth = sie_previous_ecearth[:,ind_first_year:ind_last_year]
tas_ecearth = tas_ecearth[:,ind_first_year:ind_last_year]
sst_ecearth = sst_ecearth[:,ind_first_year:ind_last_year]
sam_ecearth = sam_ecearth[:,ind_first_year:ind_last_year]
asl_ecearth = asl_ecearth[:,ind_first_year:ind_last_year]
nino34_ecearth = nino34_ecearth[:,ind_first_year:ind_last_year]
dmi_ecearth = dmi_ecearth[:,ind_first_year:ind_last_year]

# Take 1982-2023 for all CESM2 variables
sie_summer_cesm = np.roll(sie_summer_cesm,-1,axis=1) # shift summer SIE to the left (so that it lags other variables)
sie_summer_cesm = sie_summer_cesm[:,ind_first_year:ind_last_year]
sie_previous_cesm = sie_previous_cesm[:,ind_first_year:ind_last_year]
tas_cesm = tas_cesm[:,ind_first_year:ind_last_year]
sst_cesm = sst_cesm[:,ind_first_year:ind_last_year]
sam_cesm = sam_cesm[:,ind_first_year:ind_last_year]
asl_cesm = asl_cesm[:,ind_first_year:ind_last_year]
nino34_cesm = nino34_cesm[:,ind_first_year:ind_last_year]
dmi_cesm = dmi_cesm[:,ind_first_year:ind_last_year]

# Take 1982-2023 for all MPI-ESM1.2-LR variables
sie_summer_mpi = np.roll(sie_summer_mpi,-1,axis=1) # shift summer SIE to the left (so that it lags other variables)
sie_summer_mpi = sie_summer_mpi[:,ind_first_year:ind_last_year]
sie_previous_mpi = sie_previous_mpi[:,ind_first_year:ind_last_year]
tas_mpi = tas_mpi[:,ind_first_year:ind_last_year]
sst_mpi = sst_mpi[:,ind_first_year:ind_last_year]
sam_mpi = sam_mpi[:,ind_first_year:ind_last_year]
asl_mpi = asl_mpi[:,ind_first_year:ind_last_year]
nino34_mpi = nino34_mpi[:,ind_first_year:ind_last_year]
dmi_mpi = dmi_mpi[:,ind_first_year:ind_last_year]

# Take 1982-2023 for all CanESM5 variables
sie_summer_canesm = np.roll(sie_summer_canesm,-1,axis=1) # shift summer SIE to the left (so that it lags other variables)
sie_summer_canesm = sie_summer_canesm[:,ind_first_year:ind_last_year]
sie_previous_canesm = sie_previous_canesm[:,ind_first_year:ind_last_year]
tas_canesm = tas_canesm[:,ind_first_year:ind_last_year]
sst_canesm = sst_canesm[:,ind_first_year:ind_last_year]
sam_canesm = sam_canesm[:,ind_first_year:ind_last_year]
asl_canesm = asl_canesm[:,ind_first_year:ind_last_year]
nino34_canesm = nino34_canesm[:,ind_first_year:ind_last_year]
dmi_canesm = dmi_canesm[:,ind_first_year:ind_last_year]

# Take 1982-2023 for all ACCESS-ESM1-5 variables
sie_summer_access = np.roll(sie_summer_access,-1,axis=1) # shift summer SIE to the left (so that it lags other variables)
sie_summer_access = sie_summer_access[:,ind_first_year:ind_last_year]
sie_previous_access = sie_previous_access[:,ind_first_year:ind_last_year]
tas_access = tas_access[:,ind_first_year:ind_last_year]
sst_access = sst_access[:,ind_first_year:ind_last_year]
sam_access = sam_access[:,ind_first_year:ind_last_year]
asl_access = asl_access[:,ind_first_year:ind_last_year]
nino34_access = nino34_access[:,ind_first_year:ind_last_year]
dmi_access = dmi_access[:,ind_first_year:ind_last_year]

# Take 1982-2023 for all observational variables
sie_summer_obs = sie_summer_obs[4::] # DJF 1982-1983 - DJF 2023-2024
sie_previous_obs = sie_previous_obs[3::] # JAS/OND 1982 - JAS/OND 2023
sam_obs = sam_obs[12::]
asl_obs = asl_obs[12::]
nino34_obs = nino34_obs[12::]
dmi_obs = dmi_obs[12::]

# Compute mean state EC-Earth3
sie_previous_ecearth_mean = np.nanmean(sie_previous_ecearth)
tas_ecearth_mean = np.nanmean(tas_ecearth)
sst_ecearth_mean = np.nanmean(sst_ecearth)
sam_ecearth_mean = np.nanmean(sam_ecearth)
asl_ecearth_mean = np.nanmean(asl_ecearth)
nino34_ecearth_mean = np.nanmean(nino34_ecearth)
dmi_ecearth_mean = np.nanmean(dmi_ecearth)

# Compute mean state CESM2
sie_previous_cesm_mean = np.nanmean(sie_previous_cesm)
tas_cesm_mean = np.nanmean(tas_cesm)
sst_cesm_mean = np.nanmean(sst_cesm)
sam_cesm_mean = np.nanmean(sam_cesm)
asl_cesm_mean = np.nanmean(asl_cesm)
nino34_cesm_mean = np.nanmean(nino34_cesm)
dmi_cesm_mean = np.nanmean(dmi_cesm)

# Compute mean state MPI-ESM1-2-LR
sie_previous_mpi_mean = np.nanmean(sie_previous_mpi)
tas_mpi_mean = np.nanmean(tas_mpi)
sst_mpi_mean = np.nanmean(sst_mpi)
sam_mpi_mean = np.nanmean(sam_mpi)
asl_mpi_mean = np.nanmean(asl_mpi)
nino34_mpi_mean = np.nanmean(nino34_mpi)
dmi_mpi_mean = np.nanmean(dmi_mpi)

# Compute mean state CanESM5
sie_previous_canesm_mean = np.nanmean(sie_previous_canesm)
tas_canesm_mean = np.nanmean(tas_canesm)
sst_canesm_mean = np.nanmean(sst_canesm)
sam_canesm_mean = np.nanmean(sam_canesm)
asl_canesm_mean = np.nanmean(asl_canesm)
nino34_canesm_mean = np.nanmean(nino34_canesm)
dmi_canesm_mean = np.nanmean(dmi_canesm)

# Compute mean state ACCESS-ESM1-5
sie_previous_access_mean = np.nanmean(sie_previous_access)
tas_access_mean = np.nanmean(tas_access)
sst_access_mean = np.nanmean(sst_access)
sam_access_mean = np.nanmean(sam_access)
asl_access_mean = np.nanmean(asl_access)
nino34_access_mean = np.nanmean(nino34_access)
dmi_access_mean = np.nanmean(dmi_access)

# Compute mean state Obs.
sie_previous_obs_mean = np.nanmean(sie_previous_obs)
tas_obs_mean = np.nanmean(tas_obs)
sst_obs_mean = np.nanmean(sst_obs)
sam_obs_mean = np.nanmean(sam_obs)
asl_obs_mean = np.nanmean(asl_obs)
nino34_obs_mean = np.nanmean(nino34_obs)
dmi_obs_mean = np.nanmean(dmi_obs)

# Compute variability EC-Earth3
sie_previous_ecearth_var = np.nanstd(sie_previous_ecearth)
tas_ecearth_var = np.nanstd(tas_ecearth)
sst_ecearth_var = np.nanstd(sst_ecearth)
sam_ecearth_var = np.nanstd(sam_ecearth)
asl_ecearth_var = np.nanstd(asl_ecearth)
nino34_ecearth_var = np.nanstd(nino34_ecearth)
dmi_ecearth_var = np.nanstd(dmi_ecearth)

# Compute variability CESM2
sie_previous_cesm_var = np.nanstd(sie_previous_cesm)
tas_cesm_var = np.nanstd(tas_cesm)
sst_cesm_var = np.nanstd(sst_cesm)
sam_cesm_var = np.nanstd(sam_cesm)
asl_cesm_var = np.nanstd(asl_cesm)
nino34_cesm_var = np.nanstd(nino34_cesm)
dmi_cesm_var = np.nanstd(dmi_cesm)

# Compute variability MPI
sie_previous_mpi_var = np.nanstd(sie_previous_mpi)
tas_mpi_var = np.nanstd(tas_mpi)
sst_mpi_var = np.nanstd(sst_mpi)
sam_mpi_var = np.nanstd(sam_mpi)
asl_mpi_var = np.nanstd(asl_mpi)
nino34_mpi_var = np.nanstd(nino34_mpi)
dmi_mpi_var = np.nanstd(dmi_mpi)

# Compute variability CanESM5
sie_previous_canesm_var = np.nanstd(sie_previous_canesm)
tas_canesm_var = np.nanstd(tas_canesm)
sst_canesm_var = np.nanstd(sst_canesm)
sam_canesm_var = np.nanstd(sam_canesm)
asl_canesm_var = np.nanstd(asl_canesm)
nino34_canesm_var = np.nanstd(nino34_canesm)
dmi_canesm_var = np.nanstd(dmi_canesm)

# Compute variability ACCESS
sie_previous_access_var = np.nanstd(sie_previous_access)
tas_access_var = np.nanstd(tas_access)
sst_access_var = np.nanstd(sst_access)
sam_access_var = np.nanstd(sam_access)
asl_access_var = np.nanstd(asl_access)
nino34_access_var = np.nanstd(nino34_access)
dmi_access_var = np.nanstd(dmi_access)

# Compute variability Obs.
sie_previous_obs_var = np.nanstd(sie_previous_obs)
tas_obs_var = np.nanstd(tas_obs)
sst_obs_var = np.nanstd(sst_obs)
sam_obs_var = np.nanstd(sam_obs)
asl_obs_var = np.nanstd(asl_obs)
nino34_obs_var = np.nanstd(nino34_obs)
dmi_obs_var = np.nanstd(dmi_obs)


# Scatter plot of information transfer as function of initial mean state and variability
fig,ax = plt.subplots(2,3,figsize=(24,14))
fig.subplots_adjust(left=0.06,bottom=0.08,right=0.95,top=0.95,wspace=0.3,hspace=0.3)

# Previous SIE
ax[0,0].errorbar(sie_previous_access_mean,np.abs(tau_access[1,0]),yerr=conf*error_tau_access[1,0],fmt='o',color='orange',markersize=16,label='ACCESS-ESM1.5')
ax[0,0].errorbar(sie_previous_canesm_mean,np.abs(tau_canesm[1,0]),yerr=conf*error_tau_canesm[1,0],fmt='o',color='green',markersize=16,label='CanESM5')
ax[0,0].errorbar(sie_previous_cesm_mean,np.abs(tau_cesm[1,0]),yerr=conf*error_tau_cesm[1,0],fmt='o',color='blue',markersize=16,label='CESM2')
ax[0,0].errorbar(sie_previous_ecearth_mean,np.abs(tau_ecearth[1,0]),yerr=conf*error_tau_ecearth[1,0],fmt='o',color='red',markersize=16,label='EC-Earth3')
ax[0,0].errorbar(sie_previous_mpi_mean,np.abs(tau_mpi[1,0]),yerr=conf*error_tau_mpi[1,0],fmt='o',color='gray',markersize=16,label='MPI-ESM1.2-LR')
ax[0,0].errorbar(sie_previous_obs_mean,np.abs(tau_obs[1,0]),yerr=conf*error_tau_obs[1,0],fmt='o',color='black',markersize=16,label='Obs. (1982-2023)')
ax[0,0].set_xlabel('Mean previous SIE (10$^6$ km$^2$)',fontsize=24)
ax[0,0].set_ylabel(r'$|\tau_{PSIE \longrightarrow SSIE}|$ ($\%$)',fontsize=24)
ax[0,0].tick_params(axis='both',labelsize=18)
ax[0,0].set_ylim([-1,25])
ax[0,0].grid(linestyle='--')
ax[0,0].legend(loc='lower left',fontsize=16,shadow=True,frameon=False)
ax[0,0].set_title('a',loc='left',fontsize=25,fontweight='bold')

# T2m
ax[0,1].errorbar(tas_access_mean,np.abs(tau_access[2,0]),yerr=conf*error_tau_access[2,0],fmt='o',color='orange',markersize=16,label='ACCESS-ESM1.5')
ax[0,1].errorbar(tas_canesm_mean,np.abs(tau_canesm[2,0]),yerr=conf*error_tau_canesm[2,0],fmt='o',color='green',markersize=16,label='CanESM5')
ax[0,1].errorbar(tas_cesm_mean,np.abs(tau_cesm[2,0]),yerr=conf*error_tau_cesm[2,0],fmt='o',color='blue',markersize=16,label='CESM2')
ax[0,1].errorbar(tas_ecearth_mean,np.abs(tau_ecearth[2,0]),yerr=conf*error_tau_ecearth[2,0],fmt='o',color='red',markersize=16,label='EC-Earth3')
ax[0,1].errorbar(tas_mpi_mean,np.abs(tau_mpi[2,0]),yerr=conf*error_tau_mpi[2,0],fmt='o',color='gray',markersize=16,label='MPI-ESM1.2-LR')
ax[0,1].errorbar(tas_obs_mean,np.abs(tau_obs[2,0]),yerr=conf*error_tau_obs[2,0],fmt='o',color='black',markersize=16,label='Obs. (1982-2023)')
ax[0,1].set_xlabel('Mean surf. air temperature ($^\circ$C)',fontsize=24)
ax[0,1].set_ylabel(r'$|\tau_{T_{2m} \longrightarrow SSIE}|$ ($\%$)',fontsize=24)
ax[0,1].tick_params(axis='both',labelsize=18)
ax[0,1].set_ylim([-1,25])
ax[0,1].grid(linestyle='--')
ax[0,1].set_title('b',loc='left',fontsize=25,fontweight='bold')

# SST
ax[0,2].errorbar(sst_access_mean,np.abs(tau_access[3,0]),yerr=conf*error_tau_access[3,0],fmt='o',color='orange',markersize=16,label='ACCESS-ESM1.5')
ax[0,2].errorbar(sst_canesm_mean,np.abs(tau_canesm[3,0]),yerr=conf*error_tau_canesm[3,0],fmt='o',color='green',markersize=16,label='CanESM5')
ax[0,2].errorbar(sst_cesm_mean,np.abs(tau_cesm[3,0]),yerr=conf*error_tau_cesm[3,0],fmt='o',color='blue',markersize=16,label='CESM2')
ax[0,2].errorbar(sst_ecearth_mean,np.abs(tau_ecearth[3,0]),yerr=conf*error_tau_ecearth[3,0],fmt='o',color='red',markersize=16,label='EC-Earth3')
ax[0,2].errorbar(sst_mpi_mean,np.abs(tau_mpi[3,0]),yerr=conf*error_tau_mpi[3,0],fmt='o',color='gray',markersize=16,label='MPI-ESM1.2-LR')
ax[0,2].errorbar(sst_obs_mean,np.abs(tau_obs[3,0]),yerr=conf*error_tau_obs[3,0],fmt='o',color='black',markersize=16,label='Obs. (1982-2023)')
ax[0,2].set_xlabel('Mean SST ($^\circ$C)',fontsize=24)
ax[0,2].set_ylabel(r'$|\tau_{SST \longrightarrow SSIE}|$ ($\%$)',fontsize=24)
ax[0,2].tick_params(axis='both',labelsize=18)
ax[0,2].set_ylim([-1,25])
ax[0,2].grid(linestyle='--')
ax[0,2].set_title('c',loc='left',fontsize=25,fontweight='bold')

# Previous SIE
ax[1,0].errorbar(sie_previous_access_var,np.abs(tau_access[1,0]),yerr=conf*error_tau_access[1,0],fmt='o',color='orange',markersize=16,label='ACCESS-ESM1.5')
ax[1,0].errorbar(sie_previous_canesm_var,np.abs(tau_canesm[1,0]),yerr=conf*error_tau_canesm[1,0],fmt='o',color='green',markersize=16,label='CanESM5')
ax[1,0].errorbar(sie_previous_cesm_var,np.abs(tau_cesm[1,0]),yerr=conf*error_tau_cesm[1,0],fmt='o',color='blue',markersize=16,label='CESM2')
ax[1,0].errorbar(sie_previous_ecearth_var,np.abs(tau_ecearth[1,0]),yerr=conf*error_tau_ecearth[1,0],fmt='o',color='red',markersize=16,label='EC-Earth3')
ax[1,0].errorbar(sie_previous_mpi_var,np.abs(tau_mpi[1,0]),yerr=conf*error_tau_mpi[1,0],fmt='o',color='gray',markersize=16,label='MPI-ESM1.2-LR')
ax[1,0].errorbar(sie_previous_obs_var,np.abs(tau_obs[1,0]),yerr=conf*error_tau_obs[1,0],fmt='o',color='black',markersize=16,label='Obs. (1982-2023)')
ax[1,0].set_xlabel('Variability in previous SIE (10$^6$ km$^2$)',fontsize=24)
ax[1,0].set_ylabel(r'$\|\tau_{PSIE \longrightarrow SSIE}\|$ ($\%$)',fontsize=24)
ax[1,0].tick_params(axis='both',labelsize=18)
ax[1,0].set_ylim([-1,25])
ax[1,0].grid(linestyle='--')
ax[1,0].legend(loc='lower right',fontsize=16,shadow=True,frameon=False)
ax[1,0].set_title('d',loc='left',fontsize=25,fontweight='bold')

# T2m
ax[1,1].errorbar(tas_access_var,np.abs(tau_access[2,0]),yerr=conf*error_tau_access[2,0],fmt='o',color='orange',markersize=16,label='ACCESS-ESM1.5')
ax[1,1].errorbar(tas_canesm_var,np.abs(tau_canesm[2,0]),yerr=conf*error_tau_canesm[2,0],fmt='o',color='green',markersize=16,label='CanESM5')
ax[1,1].errorbar(tas_cesm_var,np.abs(tau_cesm[2,0]),yerr=conf*error_tau_cesm[2,0],fmt='o',color='blue',markersize=16,label='CESM2')
ax[1,1].errorbar(tas_ecearth_var,np.abs(tau_ecearth[2,0]),yerr=conf*error_tau_ecearth[2,0],fmt='o',color='red',markersize=16,label='EC-Earth3')
ax[1,1].errorbar(tas_mpi_var,np.abs(tau_mpi[2,0]),yerr=conf*error_tau_mpi[2,0],fmt='o',color='gray',markersize=16,label='MPI-ESM1.2-LR')
ax[1,1].errorbar(tas_obs_var,np.abs(tau_obs[2,0]),yerr=conf*error_tau_obs[2,0],fmt='o',color='black',markersize=16,label='Obs. (1982-2023)')
ax[1,1].set_xlabel('Variability in surf. air temperature ($^\circ$C)',fontsize=24)
ax[1,1].set_ylabel(r'$\|\tau_{T_{2m} \longrightarrow SSIE}\|$ ($\%$)',fontsize=24)
ax[1,1].tick_params(axis='both',labelsize=18)
ax[1,1].set_ylim([-1,25])
ax[1,1].grid(linestyle='--')
ax[1,1].set_title('e',loc='left',fontsize=25,fontweight='bold')

# SST
ax[1,2].errorbar(sst_access_var,np.abs(tau_access[3,0]),yerr=conf*error_tau_access[3,0],fmt='o',color='orange',markersize=16,label='ACCESS-ESM1.5')
ax[1,2].errorbar(sst_canesm_var,np.abs(tau_canesm[3,0]),yerr=conf*error_tau_canesm[3,0],fmt='o',color='green',markersize=16,label='CanESM5')
ax[1,2].errorbar(sst_cesm_var,np.abs(tau_cesm[3,0]),yerr=conf*error_tau_cesm[3,0],fmt='o',color='blue',markersize=16,label='CESM2')
ax[1,2].errorbar(sst_ecearth_var,np.abs(tau_ecearth[3,0]),yerr=conf*error_tau_ecearth[3,0],fmt='o',color='red',markersize=16,label='EC-Earth3')
ax[1,2].errorbar(sst_mpi_var,np.abs(tau_mpi[3,0]),yerr=conf*error_tau_mpi[3,0],fmt='o',color='gray',markersize=16,label='MPI-ESM1.2-LR')
ax[1,2].errorbar(sst_obs_var,np.abs(tau_obs[3,0]),yerr=conf*error_tau_obs[3,0],fmt='o',color='black',markersize=16,label='Obs. (1982-2023)')
ax[1,2].set_xlabel('Variability in SST ($^\circ$C)',fontsize=24)
ax[1,2].set_ylabel(r'$\|\tau_{SST \longrightarrow SSIE}\|$ ($\%$)',fontsize=24)
ax[1,2].tick_params(axis='both',labelsize=18)
ax[1,2].set_ylim([-1,25])
ax[1,2].grid(linestyle='--')
ax[1,2].set_title('f',loc='left',fontsize=25,fontweight='bold')

# Save figure
if save_fig == True:
    fig.savefig(dir_fig + 'fig_a11.pdf')