#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Save time series of original variables - Separate models

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

# Options
season = 'OND' # MAM, AMJ, MJJ, JJA, JAS, ASO, SON, OND / default: OND (spring) or JAS (winter)
model = 'ACCESS-ESM1-5' # SMHI-LENS; CESM2-LE; MPI-ESM1-2-LR; CanESM5; ACCESS-ESM1-5
if model == 'SMHI-LENS':
    string_model = 'EC-Earth3'
elif model == 'CESM2-LE':
    string_model = 'CESM2'
else:
    string_model = model
scenario = 'ssp370' # ssp119, ssp370, ssp585
nmy = 12 # number of months in a year
save_var = True

# Working directories
if model == 'SMHI-LENS':
    dir_input = '/home/ddocquier/Documents/Models/' + model + '/input/'
else:
    dir_input = '/home/ddocquier/Documents/Models/' + model + '/'
dir_output = '/home/ddocquier/Documents/Papers/My_Papers/RESIST_Antarctic/output/seasons/'

# Number of members
if model == 'SMHI-LENS' or model == 'CESM2-LE' or model == 'CanESM5' or model == 'MPI-ESM1-2-LR':
    n_members = 50 # number of members
if model == 'ACCESS-ESM1-5':
    n_members = 40

# Load monthly sea-ice extent from model (1970-2014)
filename = dir_input + 'AntSIE_' + model + '_historical.npy'
sie_hist = np.load(filename,allow_pickle=True)
if model == 'CESM2-LE':
    sie_hist = sie_hist[:,120::,:]
nyears_hist = np.size(sie_hist,1)

# Load monthly sea-ice extent from model (2015-2100)
filename = dir_input + 'AntSIE_' + model + '_' + scenario + '.npy'
sie_ssp = np.load(filename,allow_pickle=True)
nyears_ssp = np.size(sie_ssp,1)
    
# Load monthly mean surface air temperature from model (1970-2014)
if model == 'SMHI-LENS':
    filename = dir_input + 'tas_Ocean_Antarctic_' + model + '_historical.npy'
else:
    filename = dir_input + 'tas_mon_Ocean_Antarctic_' + model + '_historical.npy'
tas_hist = np.load(filename,allow_pickle=True)

# Load monthly mean surface air temperature from model (2015-2100)
if model == 'SMHI-LENS':
    filename = dir_input + 'tas_Ocean_Antarctic_' + model + '_' + scenario + '.npy'
else:
    filename = dir_input + 'tas_mon_Ocean_Antarctic_' + model + '_' + scenario + '.npy'
tas_ssp = np.load(filename,allow_pickle=True)

# Load monthly mean SST from model (1970-2014)
if model == 'SMHI-LENS':
    filename = dir_input + 'sst_Antarctic_' + model + '_historical.npy'
else:
    filename = dir_input + 'sst_mon_Antarctic_' + model + '_historical.npy'
sst_hist = np.load(filename,allow_pickle=True)

# Load monthly mean SST from model (2015-2100)
if model == 'SMHI-LENS':
    filename = dir_input + 'sst_Antarctic_' + model + '_' + scenario + '.npy'
else:
    filename = dir_input + 'sst_mon_Antarctic_' + model + '_' + scenario + '.npy'
sst_ssp = np.load(filename,allow_pickle=True)

# Load SAM index from model (1970-2014)
filename = dir_input + 'SAM_' + model + '_historical.npy'
sam_hist = np.load(filename,allow_pickle=True)

# Load SAM index from model (2015-2100)
filename = dir_input + 'SAM_' + model + '_' + str(scenario) + '.npy'
sam_ssp = np.load(filename,allow_pickle=True)

# Load ASL actual central pressure from model (1970-2014)
filename = dir_input + 'ASL_' + model + '_historical.npy'
asl_hist = np.load(filename,allow_pickle=True)[2]

# Load ASL actual central pressure from model (2015-2100)
filename = dir_input + 'ASL_' + model + '_' + str(scenario) + '.npy'
asl_ssp = np.load(filename,allow_pickle=True)[2]

# Load Nino3.4
filename = dir_input + 'Nino34_' + model + '_1970-2100.npy'
nino34 = np.load(filename,allow_pickle=True)

# Load DMI
filename = dir_input + 'DMI_' + model + '_1970-2100.npy'
dmi = np.load(filename,allow_pickle=True)

# Concatenate historical and future periods (1970-2100)
nyears = nyears_hist + nyears_ssp
sie = np.zeros((n_members,nyears,nmy))
tas = np.zeros((n_members,nyears,nmy))
sst = np.zeros((n_members,nyears,nmy))
sam = np.zeros((n_members,nyears,nmy))
asl = np.zeros((n_members,nyears,nmy))
for m in np.arange(n_members):
    for i in np.arange(nmy):
        sie[m,:,i] = np.concatenate((sie_hist[m,:,i],sie_ssp[m,:,i]))
        tas[m,:,i] = np.concatenate((tas_hist[m,:,i],tas_ssp[m,:,i]))
        sst[m,:,i] = np.concatenate((sst_hist[m,:,i],sst_ssp[m,:,i]))
        sam[m,:,i] = np.concatenate((sam_hist[m,:,i],sam_ssp[m,:,i]))
        asl[m,:,i] = np.concatenate((asl_hist[m,:,i],asl_ssp[m,:,i]))
       
# Summer (DJF) Antarctic SIE
sie_summer = np.zeros((n_members,nyears))
for m in np.arange(n_members):
    sie_summer[m,0] = np.nanmean(sie[m,0,0:2])
    for year in np.arange(1,nyears):
        sie_conc_summer = np.concatenate((sie[m,year-1,11:12],sie[m,year,0:2])) # concatenate DJF
        sie_summer[m,year] = np.nanmean(sie_conc_summer) # mean DJF

# Drivers during previous months
sie_previous = np.zeros((n_members,nyears))
tas_previous = np.zeros((n_members,nyears))
sst_previous = np.zeros((n_members,nyears))
sam_previous = np.zeros((n_members,nyears))
asl_previous = np.zeros((n_members,nyears))
nino34_previous = np.zeros((n_members,nyears))
dmi_previous = np.zeros((n_members,nyears))
if season == 'MAM':
    index_start = 2
    index_end = 5
elif season == 'AMJ':
    index_start = 3
    index_end = 6
elif season == 'MJJ':
    index_start = 4
    index_end = 7
elif season == 'JJA':
    index_start = 5
    index_end = 8
elif season == 'JAS':
    index_start = 6
    index_end = 9
elif season == 'ASO':
    index_start = 7
    index_end = 10
elif season == 'SON':
    index_start = 8
    index_end = 11
elif season == 'OND':
    index_start = 9
    index_end = 12
for m in np.arange(n_members):
    for year in np.arange(nyears):
        sie_previous[m,year] = np.nanmean(sie[m,year,index_start:index_end])
        tas_previous[m,year] = np.nanmean(tas[m,year,index_start:index_end])
        sst_previous[m,year] = np.nanmean(sst[m,year,index_start:index_end])
        sam_previous[m,year] = np.nanmean(sam[m,year,index_start:index_end])
        asl_previous[m,year] = np.nanmean(asl[m,year,index_start:index_end])
        nino34_previous[m,year] = np.nanmean(nino34[m,year,index_start:index_end])
        dmi_previous[m,year] = np.nanmean(dmi[m,year,index_start:index_end])

# Save time series
filename = dir_output + string_model + '_Antarctic_timeseries_' + season + '.npy'
if save_var == True:
    np.save(filename,[sie_summer,sie_previous,tas_previous,sst_previous,sam_previous,asl_previous,nino34_previous,dmi_previous])
