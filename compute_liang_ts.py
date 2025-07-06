#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Time evolution of rate of information transfer

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
from function_liang_nvar import compute_liang_nvar

# Parameters
range_years = 10 # range of years (default: 10 years)
season = 'OND' # MAM, AMJ, MJJ, JJA, JAS, ASO, SON, OND / default: OND (spring) or JAS (winter)
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
ind_last_year = int(last_year-1970+1)
nyears = ind_last_year
sie_summer2 = np.zeros((n_members,nyears))
sie_previous2 = np.zeros((n_members,nyears))
tas2 = np.zeros((n_members,nyears))
sst2 = np.zeros((n_members,nyears))
sam2 = np.zeros((n_members,nyears))
asl2 = np.zeros((n_members,nyears))
nino2 = np.zeros((n_members,nyears))
dmi2 = np.zeros((n_members,nyears))
for m in np.arange(n_members):
    sie_summer2[m,:] = sie_summer[m,0:ind_last_year]
    sie_previous2[m,:] = sie_previous[m,0:ind_last_year]
    tas2[m,:] = tas[m,0:ind_last_year]
    sst2[m,:] = sst[m,0:ind_last_year]
    sam2[m,:] = sam[m,0:ind_last_year]
    asl2[m,:] = asl[m,0:ind_last_year]
    nino2[m,:] = nino[m,0:ind_last_year]
    dmi2[m,:] = dmi[m,0:ind_last_year]

# Filename
filename = dir_input + 'Liang_' + model + '_' + season + '_' + str(n_iter) + 'boot_ts.npy'

# Compute relative transfers of information (tau) and correlation coefficient (R) and their errors using function_liang_nvar
nt = int(nyears / range_years)
tau = np.zeros((nt,nvar,nvar))
R = np.zeros((nt,nvar,nvar))
error_tau = np.zeros((nt,nvar,nvar))
error_R = np.zeros((nt,nvar,nvar))
for t in np.arange(nt):
    print(t)
    if t == (nt - 1):
        sie_summer3 = sie_summer2[:,t*range_years::]
        sie_previous3 = sie_previous2[:,t*range_years::]
        tas3 = tas2[:,t*range_years::]
        sst3 = sst2[:,t*range_years::]
        sam3 = sam2[:,t*range_years::]
        asl3 = asl2[:,t*range_years::]
        nino3 = nino2[:,t*range_years::]
        dmi3 = dmi2[:,t*range_years::]
    else:
        sie_summer3 = sie_summer2[:,t*range_years:t*range_years+range_years]
        sie_previous3 = sie_previous2[:,t*range_years:t*range_years+range_years]
        tas3 = tas2[:,t*range_years:t*range_years+range_years]
        sst3 = sst2[:,t*range_years:t*range_years+range_years]
        sam3 = sam2[:,t*range_years:t*range_years+range_years]
        asl3 = asl2[:,t*range_years:t*range_years+range_years]
        nino3 = nino2[:,t*range_years:t*range_years+range_years]
        dmi3 = dmi2[:,t*range_years:t*range_years+range_years]
    n = np.size(sie_summer3)
    sie_summer4 = np.reshape(sie_summer3,n)
    sie_previous4 = np.reshape(sie_previous3,n)
    tas4 = np.reshape(tas3,n)
    sst4 = np.reshape(sst3,n)
    sam4 = np.reshape(sam3,n)
    asl4 = np.reshape(asl3,n)
    nino4 = np.reshape(nino3,n)
    dmi4 = np.reshape(dmi3,n)
    
    xx = np.array((sie_summer4,sie_previous4,tas4,sst4,sam4,asl4,nino4,dmi4))
    notused,tau[t,:,:],R[t,:,:],notused,error_tau[t,:,:],error_R[t,:,:] = compute_liang_nvar(xx,dt,n_iter)

# Save variables
if save_var == True:
    np.save(filename,[tau,R,error_tau,error_R])