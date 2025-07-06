#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute Liang index over observational period (1982-2023) and over all model members

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

Last updated: 01/07/2025

@author: David Docquier
"""

# Import libraries
import numpy as np
import sys

# Import my functions
sys.path.append('/home/ddocquier/Documents/Codes/Liang/')
from function_liang_nvar_dx import compute_liang_nvar

# Parameters
season = 'OND' # MAM, AMJ, MJJ, JJA, JAS, ASO, SON, OND / default: OND (spring) or JAS (winter)
model = 'SMHI-LENS' # SMHI-LENS; CESM2-LE; MPI-ESM1-2-LR; CanESM5; ACCESS-ESM1-5
if model == 'SMHI-LENS':
    string_model = 'EC-Earth3'
elif model == 'CESM2-LE':
    string_model = 'CESM2'
else:
    string_model = model
nvar = 8 # number of variables (1: SSIE; 2: PSIE; 3: T_2m; 4: SST; 5: SAM; 6: ASL; 7: ENSO; 8: DMI)
first_year = 1982 # starting year included in the computation
last_year = 2023 # last year included in the computation
dt = 1 # time step (years)
n_iter = 1000 # number of bootstrap realizations
conf = 2.57 # 1.96 if 95% confidence interval (normal distribution); 1.65 if 90% and 2.57 if 99%
compute_liang = True # True compute Liang index; False: load existing value
save_var = True

# Interpolate NaN values
def interpolate_nan(array_like):
    array = array_like.copy()
    nans = np.isnan(array)
    
    def get_x(a):
        return a.nonzero()[0]

    array[nans] = np.interp(get_x(nans), get_x(~nans), array[~nans])

    return array

# Working directories
dir_input = '/home/ddocquier/Documents/Papers/My_Papers/RESIST_Antarctic/output/seasons/'

# Number of members
if model == 'ACCESS-ESM1-5':
    n_members = 40
else:
    n_members = 50

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

# Take years of interest
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
    
# Compute dx/dt (tendency) of detrended data for Liang index
dsie_summer2 = np.zeros((n_members,nyears_new))
dsie_previous2 = np.zeros((n_members,nyears_new))
dtas2 = np.zeros((n_members,nyears_new))
dsam2 = np.zeros((n_members,nyears_new))
dsst2 = np.zeros((n_members,nyears_new))
dasl2 = np.zeros((n_members,nyears_new))
dnino2 = np.zeros((n_members,nyears_new))
ddmi2 = np.zeros((n_members,nyears_new))
for m in np.arange(n_members):
    dsie_summer2[m,0:nyears_new-1] = (sie_summer2[m,1:nyears_new] - sie_summer2[m,0:nyears_new-1]) / dt
    dsie_previous2[m,0:nyears_new-1] = (sie_previous2[m,1:nyears_new] - sie_previous2[m,0:nyears_new-1]) / dt
    dtas2[m,0:nyears_new-1] = (tas2[m,1:nyears_new] - tas2[m,0:nyears_new-1]) / dt
    dsam2[m,0:nyears_new-1] = (sam2[m,1:nyears_new] - sam2[m,0:nyears_new-1]) / dt
    dsst2[m,0:nyears_new-1] = (sst2[m,1:nyears_new] - sst2[m,0:nyears_new-1]) / dt
    dasl2[m,0:nyears_new-1] = (asl2[m,1:nyears_new] - asl2[m,0:nyears_new-1]) / dt
    dnino2[m,0:nyears_new-1] = (nino2[m,1:nyears_new] - nino2[m,0:nyears_new-1]) / dt
    ddmi2[m,0:nyears_new-1] = (dmi2[m,1:nyears_new] - dmi2[m,0:nyears_new-1]) / dt

# Concatenate all members and create 1 single time series for each variable
nt_full = nyears_new * n_members
sie_summer_full = np.reshape(sie_summer2,nt_full)
dsie_summer_full = np.reshape(dsie_summer2,nt_full)
sie_previous_full = np.reshape(sie_previous2,nt_full)
dsie_previous_full = np.reshape(dsie_previous2,nt_full)
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
filename = dir_input + 'Liang_' + model + '_' + season + '_' + str(n_iter) + 'boot_' + str(first_year) + '-' + str(last_year) + '.npy'
if compute_liang == True:
    tau = np.zeros((nvar,nvar))
    R = np.zeros((nvar,nvar))
    error_tau = np.zeros((nvar,nvar))
    error_R = np.zeros((nvar,nvar))
    xx = np.array((sie_summer_full,sie_previous_full,tas_full,sst_full,sam_full,asl_full,nino_full,dmi_full))
    dx = np.array((dsie_summer_full,dsie_previous_full,dtas_full,dsst_full,dsam_full,dasl_full,dnino_full,ddmi_full))
    tau,R,error_tau,error_R = compute_liang_nvar(xx,dx,dt,n_iter)
    if save_var == True:
        np.save(filename,[tau,R,error_tau,error_R])
else:
    tau,R,error_tau,error_R = np.load(filename,allow_pickle=True)