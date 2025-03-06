#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Figures A5-A7: Time series of drivers (SIE, T2m, SST) in the previous season for different Antarctic sectors

Large ensembles: EC-Earth3 (SMHI-LENS), CESM2-LE, MPI-ESM1-2-LR, CanESM5, ACCESS-ESM1-5
Time series saved via save_timeseries.py for total Antarctic

Observations
Time series saved via save_timeseries_obs.py for total Antarctic

Last updated: 12/02/2025

@author: David Docquier
"""

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
np.bool = np.bool_

# Options
driver = 'sst_mon' # SIE, tas_mon, sst_mon, mld_mon, tauv
season = 'OND' # JAS (previous winter), OND (previous spring; default)
nmy = 12 # number of months in a year
save_fig = True
save_var = True

# String driver
if driver == 'SIE':
    string_driver = driver
    string_driver_plot = 'SIE (10$^6$ km$^2$)'
elif driver == 'tas_mon':
    string_driver = 'T2m'
    string_driver_plot = 'T$_{2m}$ ($^\circ$C)'
elif driver == 'sst_mon':
    string_driver = 'SST'
    string_driver_plot = 'SST ($^\circ$C)'
elif driver == 'mld_mon':
    string_driver = 'MLD'
    string_driver_plot = 'MLD (m)'
elif driver == 'tauv_mon':
    string_driver = 'tauv'
    string_driver_plot = r'$\tau_v$ (Pa)'

# Indices for season
if season == 'JAS':
    index_start = 6
    index_end = 9
elif season == 'OND':
    index_start = 9
    index_end = 12
    
# Interpolate NaN values
def interpolate_nan(array_like):
    array = array_like.copy()
    nans = np.isnan(array)
    
    def get_x(a):
        return a.nonzero()[0]

    array[nans] = np.interp(get_x(nans), get_x(~nans), array[~nans])

    return array

# Working directories
dir_input = '/home/dadocq/Documents/Models/'
dir_input2 = '/home/dadocq/Documents/Papers/My_Papers/RESIST_Antarctic/output/seasons/'
dir_osisaf = '/home/dadocq/Documents/Observations/OSISAF/' # Observed sea-ice extent
dir_hadcrut5 = '/home/dadocq/Documents/Observations/HadCRUT5/' # Observed air temperature
dir_ostia = '/home/dadocq/Documents/Observations/OSTIA/' # Observed SST
dir_output = dir_input2
dir_fig = '/home/dadocq/Documents/Papers/My_Papers/RESIST_Antarctic/LaTeX/'

# Load Antarctic EC-Earth3
filename = dir_input2 + 'EC-Earth3_Antarctic_timeseries_' + season + '.npy'
sie_summer_ecearth,sie_previous_ecearth,tas_ecearth,sst_ecearth,notused,notused,notused,notused = np.load(filename,allow_pickle=True)

# Load Antarctic CESM2
filename = dir_input2 + 'CESM2_Antarctic_timeseries_' + season + '.npy'
sie_summer_cesm,sie_previous_cesm,tas_cesm,sst_cesm,notused,notused,notused,notused = np.load(filename,allow_pickle=True)
#
# Load Antarctic MPI-ESM1-2-LR
filename = dir_input2 + 'MPI-ESM1-2-LR_Antarctic_timeseries_' + season + '.npy'
sie_summer_mpi,sie_previous_mpi,tas_mpi,sst_mpi,notused,notused,notused,notused = np.load(filename,allow_pickle=True)

# Load Antarctic CanESM5
filename = dir_input2 + 'CanESM5_Antarctic_timeseries_' + season + '.npy'
sie_summer_canesm,sie_previous_canesm,tas_canesm,sst_canesm,notused,notused,notused,notused = np.load(filename,allow_pickle=True)

# Load Antarctic ACCESS-ESM1-5
filename = dir_input2 + 'ACCESS-ESM1-5_Antarctic_timeseries_' + season + '.npy'
sie_summer_access,sie_previous_access,tas_access,sst_access,notused,notused,notused,notused = np.load(filename,allow_pickle=True)

# Load Antarctic Obs.
filename = dir_input2 + 'Obs_Antarctic_timeseries_' + season + '.npy'
sie_summer_obs,sie_previous_obs,tas_obs,sst_obs,notused,notused,notused,notused = np.load(filename,allow_pickle=True)

# Store Antarctic driver
if driver == 'SIE':
    driver_antarctic_ecearth = sie_previous_ecearth
    driver_antarctic_cesm = sie_previous_cesm
    driver_antarctic_mpi = sie_previous_mpi
    driver_antarctic_canesm = sie_previous_canesm
    driver_antarctic_access = sie_previous_access
    driver_antarctic_obs = sie_previous_obs
elif driver == 'tas_mon':
    driver_antarctic_ecearth = tas_ecearth
    driver_antarctic_cesm = tas_cesm
    driver_antarctic_mpi = tas_mpi
    driver_antarctic_canesm = tas_canesm
    driver_antarctic_access = tas_access
    driver_antarctic_obs = tas_obs
elif driver == 'sst_mon':
    driver_antarctic_ecearth = sst_ecearth
    driver_antarctic_cesm = sst_cesm
    driver_antarctic_mpi = sst_mpi
    driver_antarctic_canesm = sst_canesm
    driver_antarctic_access = sst_access
    driver_antarctic_obs = sst_obs

# Load regional driver from ACCESS during historical period (1970-2014)
filename = dir_input + 'ACCESS-ESM1-5/' + driver + '_AntReg_ACCESS-ESM1-5_historical.npy'
driver_access_bas_hist,driver_access_ws_hist,driver_access_io_hist,driver_access_wpo_hist,driver_access_rs_hist = np.load(filename,allow_pickle=True)
n_members_access = np.size(driver_access_bas_hist,0)
nyears_hist = np.size(driver_access_bas_hist,1)

# Load regional driver from ACCESS during future period (2015-2100)
filename = dir_input + 'ACCESS-ESM1-5/' + driver + '_AntReg_ACCESS-ESM1-5_ssp370.npy'
driver_access_bas_ssp,driver_access_ws_ssp,driver_access_io_ssp,driver_access_wpo_ssp,driver_access_rs_ssp = np.load(filename,allow_pickle=True)
nyears_ssp = np.size(driver_access_bas_ssp,1)

# Load regional driver from CanESM during historical period (1970-2014)
filename = dir_input + 'CanESM5/' + driver + '_AntReg_CanESM5_historical.npy'
driver_canesm_bas_hist,driver_canesm_ws_hist,driver_canesm_io_hist,driver_canesm_wpo_hist,driver_canesm_rs_hist = np.load(filename,allow_pickle=True)
n_members_canesm = np.size(driver_canesm_bas_hist,0)

# Load regional driver from CanESM during future period (2015-2100)
filename = dir_input + 'CanESM5/' + driver + '_AntReg_CanESM5_ssp370.npy'
driver_canesm_bas_ssp,driver_canesm_ws_ssp,driver_canesm_io_ssp,driver_canesm_wpo_ssp,driver_canesm_rs_ssp = np.load(filename,allow_pickle=True)

# Load regional driver from MPI during historical period (1970-2014)
filename = dir_input + 'MPI-ESM1-2-LR/' + driver + '_AntReg_MPI-ESM1-2-LR_historical.npy'
driver_mpi_bas_hist,driver_mpi_ws_hist,driver_mpi_io_hist,driver_mpi_wpo_hist,driver_mpi_rs_hist = np.load(filename,allow_pickle=True)
n_members_mpi = np.size(driver_mpi_bas_hist,0)

# Load regional driver from MPI during future period (2015-2100)
filename = dir_input + 'MPI-ESM1-2-LR/' + driver + '_AntReg_MPI-ESM1-2-LR_ssp370.npy'
driver_mpi_bas_ssp,driver_mpi_ws_ssp,driver_mpi_io_ssp,driver_mpi_wpo_ssp,driver_mpi_rs_ssp = np.load(filename,allow_pickle=True)

# Load regional driver from CESM during historical period (1970-2014)
filename = dir_input + 'CESM2-LE/' + driver + '_AntReg_CESM2-LE_historical.npy'
driver_cesm_bas_hist,driver_cesm_ws_hist,driver_cesm_io_hist,driver_cesm_wpo_hist,driver_cesm_rs_hist = np.load(filename,allow_pickle=True)
n_members_cesm = np.size(driver_cesm_bas_hist,0)

# Load regional driver from CESM during future period (2015-2100)
filename = dir_input + 'CESM2-LE/' + driver + '_AntReg_CESM2-LE_ssp370.npy'
driver_cesm_bas_ssp,driver_cesm_ws_ssp,driver_cesm_io_ssp,driver_cesm_wpo_ssp,driver_cesm_rs_ssp = np.load(filename,allow_pickle=True)

# Load regional driver from EC-Earth3 during historical period (1970-2014)
filename = dir_input + 'SMHI-LENS/input/' + driver + '_AntReg_SMHI-LENS_historical.npy'
driver_ecearth_bas_hist,driver_ecearth_ws_hist,driver_ecearth_io_hist,driver_ecearth_wpo_hist,driver_ecearth_rs_hist = np.load(filename,allow_pickle=True)
n_members_ecearth = np.size(driver_ecearth_bas_hist,0)

# Load regional driver from EC-Earth3 during future period (2015-2100)
filename = dir_input + 'SMHI-LENS/input/' + driver + '_AntReg_SMHI-LENS_ssp370.npy'
driver_ecearth_bas_ssp,driver_ecearth_ws_ssp,driver_ecearth_io_ssp,driver_ecearth_wpo_ssp,driver_ecearth_rs_ssp = np.load(filename,allow_pickle=True)

# Concatenate historical and future periods ACCESS (1970-2100)
nyears = nyears_hist + nyears_ssp
driver_access_bas = np.zeros((n_members_access,nyears,nmy))
driver_access_ws = np.zeros((n_members_access,nyears,nmy))
driver_access_io = np.zeros((n_members_access,nyears,nmy))
driver_access_wpo = np.zeros((n_members_access,nyears,nmy))
driver_access_rs = np.zeros((n_members_access,nyears,nmy))
for m in np.arange(n_members_access):
    for i in np.arange(nmy):
        driver_access_bas[m,:,i] = np.concatenate((driver_access_bas_hist[m,:,i],driver_access_bas_ssp[m,:,i]))
        driver_access_ws[m,:,i] = np.concatenate((driver_access_ws_hist[m,:,i],driver_access_ws_ssp[m,:,i]))
        driver_access_io[m,:,i] = np.concatenate((driver_access_io_hist[m,:,i],driver_access_io_ssp[m,:,i]))
        driver_access_wpo[m,:,i] = np.concatenate((driver_access_wpo_hist[m,:,i],driver_access_wpo_ssp[m,:,i]))
        driver_access_rs[m,:,i] = np.concatenate((driver_access_rs_hist[m,:,i],driver_access_rs_ssp[m,:,i]))

# Concatenate historical and future periods CanESM (1970-2100)
driver_canesm_bas = np.zeros((n_members_canesm,nyears,nmy))
driver_canesm_ws = np.zeros((n_members_canesm,nyears,nmy))
driver_canesm_io = np.zeros((n_members_canesm,nyears,nmy))
driver_canesm_wpo = np.zeros((n_members_canesm,nyears,nmy))
driver_canesm_rs = np.zeros((n_members_canesm,nyears,nmy))
for m in np.arange(n_members_canesm):
    for i in np.arange(nmy):
        driver_canesm_bas[m,:,i] = np.concatenate((driver_canesm_bas_hist[m,:,i],driver_canesm_bas_ssp[m,:,i]))
        driver_canesm_ws[m,:,i] = np.concatenate((driver_canesm_ws_hist[m,:,i],driver_canesm_ws_ssp[m,:,i]))
        driver_canesm_io[m,:,i] = np.concatenate((driver_canesm_io_hist[m,:,i],driver_canesm_io_ssp[m,:,i]))
        driver_canesm_wpo[m,:,i] = np.concatenate((driver_canesm_wpo_hist[m,:,i],driver_canesm_wpo_ssp[m,:,i]))
        driver_canesm_rs[m,:,i] = np.concatenate((driver_canesm_rs_hist[m,:,i],driver_canesm_rs_ssp[m,:,i]))
        
# Concatenate historical and future periods MPI (1970-2100)
driver_mpi_bas = np.zeros((n_members_mpi,nyears,nmy))
driver_mpi_ws = np.zeros((n_members_mpi,nyears,nmy))
driver_mpi_io = np.zeros((n_members_mpi,nyears,nmy))
driver_mpi_wpo = np.zeros((n_members_mpi,nyears,nmy))
driver_mpi_rs = np.zeros((n_members_mpi,nyears,nmy))
for m in np.arange(n_members_mpi):
    for i in np.arange(nmy):
        driver_mpi_bas[m,:,i] = np.concatenate((driver_mpi_bas_hist[m,:,i],driver_mpi_bas_ssp[m,:,i]))
        driver_mpi_ws[m,:,i] = np.concatenate((driver_mpi_ws_hist[m,:,i],driver_mpi_ws_ssp[m,:,i]))
        driver_mpi_io[m,:,i] = np.concatenate((driver_mpi_io_hist[m,:,i],driver_mpi_io_ssp[m,:,i]))
        driver_mpi_wpo[m,:,i] = np.concatenate((driver_mpi_wpo_hist[m,:,i],driver_mpi_wpo_ssp[m,:,i]))
        driver_mpi_rs[m,:,i] = np.concatenate((driver_mpi_rs_hist[m,:,i],driver_mpi_rs_ssp[m,:,i]))
        
# Concatenate historical and future periods CESM (1970-2100)
driver_cesm_bas = np.zeros((n_members_cesm,nyears,nmy))
driver_cesm_ws = np.zeros((n_members_cesm,nyears,nmy))
driver_cesm_io = np.zeros((n_members_cesm,nyears,nmy))
driver_cesm_wpo = np.zeros((n_members_cesm,nyears,nmy))
driver_cesm_rs = np.zeros((n_members_cesm,nyears,nmy))
for m in np.arange(n_members_cesm):
    for i in np.arange(nmy):
        driver_cesm_bas[m,:,i] = np.concatenate((driver_cesm_bas_hist[m,:,i],driver_cesm_bas_ssp[m,:,i]))
        driver_cesm_ws[m,:,i] = np.concatenate((driver_cesm_ws_hist[m,:,i],driver_cesm_ws_ssp[m,:,i]))
        driver_cesm_io[m,:,i] = np.concatenate((driver_cesm_io_hist[m,:,i],driver_cesm_io_ssp[m,:,i]))
        driver_cesm_wpo[m,:,i] = np.concatenate((driver_cesm_wpo_hist[m,:,i],driver_cesm_wpo_ssp[m,:,i]))
        driver_cesm_rs[m,:,i] = np.concatenate((driver_cesm_rs_hist[m,:,i],driver_cesm_rs_ssp[m,:,i]))
        
# Concatenate historical and future periods EC-Earth3 (1970-2100)
driver_ecearth_bas = np.zeros((n_members_ecearth,nyears,nmy))
driver_ecearth_ws = np.zeros((n_members_ecearth,nyears,nmy))
driver_ecearth_io = np.zeros((n_members_ecearth,nyears,nmy))
driver_ecearth_wpo = np.zeros((n_members_ecearth,nyears,nmy))
driver_ecearth_rs = np.zeros((n_members_ecearth,nyears,nmy))
for m in np.arange(n_members_ecearth):
    for i in np.arange(nmy):
        driver_ecearth_bas[m,:,i] = np.concatenate((driver_ecearth_bas_hist[m,:,i],driver_ecearth_bas_ssp[m,:,i]))
        driver_ecearth_ws[m,:,i] = np.concatenate((driver_ecearth_ws_hist[m,:,i],driver_ecearth_ws_ssp[m,:,i]))
        driver_ecearth_io[m,:,i] = np.concatenate((driver_ecearth_io_hist[m,:,i],driver_ecearth_io_ssp[m,:,i]))
        driver_ecearth_wpo[m,:,i] = np.concatenate((driver_ecearth_wpo_hist[m,:,i],driver_ecearth_wpo_ssp[m,:,i]))
        driver_ecearth_rs[m,:,i] = np.concatenate((driver_ecearth_rs_hist[m,:,i],driver_ecearth_rs_ssp[m,:,i]))

# Compute seasonal mean - ACCESS
driver_period_access_bas = np.zeros((n_members_access,nyears))
driver_period_access_ws = np.zeros((n_members_access,nyears))
driver_period_access_io = np.zeros((n_members_access,nyears))
driver_period_access_wpo = np.zeros((n_members_access,nyears))
driver_period_access_rs = np.zeros((n_members_access,nyears))
for m in np.arange(n_members_access):
    for year in np.arange(nyears):
        driver_period_access_bas[m,year] = np.nanmean(driver_access_bas[m,year,index_start:index_end])
        driver_period_access_ws[m,year] = np.nanmean(driver_access_ws[m,year,index_start:index_end])
        driver_period_access_io[m,year] = np.nanmean(driver_access_io[m,year,index_start:index_end])
        driver_period_access_wpo[m,year] = np.nanmean(driver_access_wpo[m,year,index_start:index_end])
        driver_period_access_rs[m,year] = np.nanmean(driver_access_rs[m,year,index_start:index_end])

# Compute seasonal mean - CanESM
driver_period_canesm_bas = np.zeros((n_members_canesm,nyears))
driver_period_canesm_ws = np.zeros((n_members_canesm,nyears))
driver_period_canesm_io = np.zeros((n_members_canesm,nyears))
driver_period_canesm_wpo = np.zeros((n_members_canesm,nyears))
driver_period_canesm_rs = np.zeros((n_members_canesm,nyears))
for m in np.arange(n_members_canesm):
    for year in np.arange(nyears):
        driver_period_canesm_bas[m,year] = np.nanmean(driver_canesm_bas[m,year,index_start:index_end])
        driver_period_canesm_ws[m,year] = np.nanmean(driver_canesm_ws[m,year,index_start:index_end])
        driver_period_canesm_io[m,year] = np.nanmean(driver_canesm_io[m,year,index_start:index_end])
        driver_period_canesm_wpo[m,year] = np.nanmean(driver_canesm_wpo[m,year,index_start:index_end])
        driver_period_canesm_rs[m,year] = np.nanmean(driver_canesm_rs[m,year,index_start:index_end])
        
# Compute seasonal mean - MPI
driver_period_mpi_bas = np.zeros((n_members_mpi,nyears))
driver_period_mpi_ws = np.zeros((n_members_mpi,nyears))
driver_period_mpi_io = np.zeros((n_members_mpi,nyears))
driver_period_mpi_wpo = np.zeros((n_members_mpi,nyears))
driver_period_mpi_rs = np.zeros((n_members_mpi,nyears))
for m in np.arange(n_members_mpi):
    for year in np.arange(nyears):
        driver_period_mpi_bas[m,year] = np.nanmean(driver_mpi_bas[m,year,index_start:index_end])
        driver_period_mpi_ws[m,year] = np.nanmean(driver_mpi_ws[m,year,index_start:index_end])
        driver_period_mpi_io[m,year] = np.nanmean(driver_mpi_io[m,year,index_start:index_end])
        driver_period_mpi_wpo[m,year] = np.nanmean(driver_mpi_wpo[m,year,index_start:index_end])
        driver_period_mpi_rs[m,year] = np.nanmean(driver_mpi_rs[m,year,index_start:index_end])
    
# Compute seasonal mean - CESM
driver_period_cesm_bas = np.zeros((n_members_cesm,nyears))
driver_period_cesm_ws = np.zeros((n_members_cesm,nyears))
driver_period_cesm_io = np.zeros((n_members_cesm,nyears))
driver_period_cesm_wpo = np.zeros((n_members_cesm,nyears))
driver_period_cesm_rs = np.zeros((n_members_cesm,nyears))
for m in np.arange(n_members_cesm):
    for year in np.arange(nyears):
        driver_period_cesm_bas[m,year] = np.nanmean(driver_cesm_bas[m,year,index_start:index_end])
        driver_period_cesm_ws[m,year] = np.nanmean(driver_cesm_ws[m,year,index_start:index_end])
        driver_period_cesm_io[m,year] = np.nanmean(driver_cesm_io[m,year,index_start:index_end])
        driver_period_cesm_wpo[m,year] = np.nanmean(driver_cesm_wpo[m,year,index_start:index_end])
        driver_period_cesm_rs[m,year] = np.nanmean(driver_cesm_rs[m,year,index_start:index_end])

# Compute seasonal mean - EC-Earth
driver_period_ecearth_bas = np.zeros((n_members_ecearth,nyears))
driver_period_ecearth_ws = np.zeros((n_members_ecearth,nyears))
driver_period_ecearth_io = np.zeros((n_members_ecearth,nyears))
driver_period_ecearth_wpo = np.zeros((n_members_ecearth,nyears))
driver_period_ecearth_rs = np.zeros((n_members_ecearth,nyears))
for m in np.arange(n_members_ecearth):
    for year in np.arange(nyears):
        driver_period_ecearth_bas[m,year] = np.nanmean(driver_ecearth_bas[m,year,index_start:index_end])
        driver_period_ecearth_ws[m,year] = np.nanmean(driver_ecearth_ws[m,year,index_start:index_end])
        driver_period_ecearth_io[m,year] = np.nanmean(driver_ecearth_io[m,year,index_start:index_end])
        driver_period_ecearth_wpo[m,year] = np.nanmean(driver_ecearth_wpo[m,year,index_start:index_end])
        driver_period_ecearth_rs[m,year] = np.nanmean(driver_ecearth_rs[m,year,index_start:index_end])

# Load Observations
if driver == 'SIE':
    
    # Load BAS SIE from OSI SAF (OSI-420) 1979-2023
    filename = dir_osisaf + 'sh_indices/osisaf_bell_sie_monthly.nc'
    fh = Dataset(filename, mode='r')
    sie_obs_init = fh.variables['sie'][:]
    fh.close()
    sie_obs_init2 = sie_obs_init[12:552] # 1979-2023
    sie_obs_init2[sie_obs_init2<0.] = np.nan
    nyears_obs = 45
    sie_obs_bas = np.zeros((nyears_obs,nmy))
    for mon in np.arange(nmy):
        sie_obs_bas[:,mon] = sie_obs_init2[mon::12]
        sie_obs_bas[:,mon] = interpolate_nan(sie_obs_bas[:,mon])
    driver_period_obs_bas = np.nanmean(sie_obs_bas[:,index_start:index_end],axis=1)
    
    # Load WS SIE from OSI SAF (OSI-420) 1979-2023
    filename = dir_osisaf + 'sh_indices/osisaf_wedd_sie_monthly.nc'
    fh = Dataset(filename, mode='r')
    sie_obs_init = fh.variables['sie'][:]
    fh.close()
    sie_obs_init2 = sie_obs_init[12:552] # 1979-2023
    sie_obs_init2[sie_obs_init2<0.] = np.nan
    nyears_obs = 45
    sie_obs_ws = np.zeros((nyears_obs,nmy))
    for mon in np.arange(nmy):
        sie_obs_ws[:,mon] = sie_obs_init2[mon::12]
        sie_obs_ws[:,mon] = interpolate_nan(sie_obs_ws[:,mon])
    driver_period_obs_ws = np.nanmean(sie_obs_ws[:,index_start:index_end],axis=1)
    
    # Load IO SIE from OSI SAF (OSI-420) 1979-2023
    filename = dir_osisaf + 'sh_indices/osisaf_indi_sie_monthly.nc'
    fh = Dataset(filename, mode='r')
    sie_obs_init = fh.variables['sie'][:]
    fh.close()
    sie_obs_init2 = sie_obs_init[12:552] # 1979-2023
    sie_obs_init2[sie_obs_init2<0.] = np.nan
    nyears_obs = 45
    sie_obs_io = np.zeros((nyears_obs,nmy))
    for mon in np.arange(nmy):
        sie_obs_io[:,mon] = sie_obs_init2[mon::12]
        sie_obs_io[:,mon] = interpolate_nan(sie_obs_io[:,mon])
    driver_period_obs_io = np.nanmean(sie_obs_io[:,index_start:index_end],axis=1)
    
    # Load WPO SIE from OSI SAF (OSI-420) 1979-2023
    filename = dir_osisaf + 'sh_indices/osisaf_wpac_sie_monthly.nc'
    fh = Dataset(filename, mode='r')
    sie_obs_init = fh.variables['sie'][:]
    fh.close()
    sie_obs_init2 = sie_obs_init[12:552] # 1979-2023
    sie_obs_init2[sie_obs_init2<0.] = np.nan
    nyears_obs = 45
    sie_obs_wpo = np.zeros((nyears_obs,nmy))
    for mon in np.arange(nmy):
        sie_obs_wpo[:,mon] = sie_obs_init2[mon::12]
        sie_obs_wpo[:,mon] = interpolate_nan(sie_obs_wpo[:,mon])
    driver_period_obs_wpo = np.nanmean(sie_obs_wpo[:,index_start:index_end],axis=1)
    
    # Load RS SIE from OSI SAF (OSI-420) 1979-2023
    filename = dir_osisaf + 'sh_indices/osisaf_ross_sie_monthly.nc'
    fh = Dataset(filename, mode='r')
    sie_obs_init = fh.variables['sie'][:]
    fh.close()
    sie_obs_init2 = sie_obs_init[12:552] # 1979-2023
    sie_obs_init2[sie_obs_init2<0.] = np.nan
    nyears_obs = 45
    sie_obs_rs = np.zeros((nyears_obs,nmy))
    for mon in np.arange(nmy):
        sie_obs_rs[:,mon] = sie_obs_init2[mon::12]
        sie_obs_rs[:,mon] = interpolate_nan(sie_obs_rs[:,mon])
    driver_period_obs_rs = np.nanmean(sie_obs_rs[:,index_start:index_end],axis=1)

elif driver == 'tas_mon':
    
    # Load monthly mean surface temperature anomalies from HadCRUT5 1970-2023
    # https://crudata.uea.ac.uk/cru/data//temperature/
    filename = dir_hadcrut5 + 'HadCRUT.5.0.2.0.analysis.anomalies.ensemble_mean.nc'
    fh = Dataset(filename, mode='r')
    tas_obs_ano = fh.variables['tas_mean'][:] # 1850-2023
    start_mon_hadcrut5 = int((1970-1850)*12) # 1970
    end_mon_hadcrut5 = int((2023-1850+1)*12) # 2023
    tas_obs_ano = tas_obs_ano[start_mon_hadcrut5:end_mon_hadcrut5,:,:] # 1970-2023
    lon_init = fh.variables['longitude'][:]
    lat_init = fh.variables['latitude'][:]
    lon_hadcrut5,lat_hadcrut5 = np.meshgrid(lon_init,lat_init)
    fh.close()
    nm_obs = np.size(tas_obs_ano,0)
    
    # Load monthly seasonal cycle of absolute surface temperature from HadCRUT5 (average over 1961-1990)
    # https://crudata.uea.ac.uk/cru/data//temperature/
    filename = dir_hadcrut5 + 'absolute_v5.nc'
    fh = Dataset(filename, mode='r')
    tas_obs_cycle = fh.variables['tem'][:]
    fh.close()
    
    # Load grid area HadCRUT5
    filename = dir_hadcrut5 + 'gridarea_hadcrut5.nc'
    fh = Dataset(filename, mode='r')
    grid_area_hadcrut5 = fh.variables['cell_area'][:] 
    fh.close()
    
    # Load SST HadSST4 (for land-sea mask)
    filename = dir_hadcrut5 + 'HadSST.4.0.1.0_median.nc'
    fh = Dataset(filename, mode='r')
    sst_hadsst4 = fh.variables['tos'][-3,:,:] # only February 2022 (minimum SIE)
    fh.close()
    
    # Compute mean surface temperature anomalies from HadCRUT5 1970-2023
    tas_bas_ano = np.zeros(nm_obs)
    tas_ws_ano = np.zeros(nm_obs)
    tas_io_ano = np.zeros(nm_obs)
    tas_wpo_ano = np.zeros(nm_obs)
    tas_rs_ano = np.zeros(nm_obs)
    index1 = (lat_hadcrut5 <= -60.) * (lon_hadcrut5 >= 160.) * (lon_hadcrut5 <= 180.)
    index2 = (lat_hadcrut5 <= -60.) * (lon_hadcrut5 >= -180.) * (lon_hadcrut5 <= -130.)
    for mon in np.arange(nm_obs):
        tas_obs_ano_masked = np.ma.MaskedArray(tas_obs_ano[mon,:,:],mask=np.isnan(tas_obs_ano[mon,:,:]))
        tas_bas_ano[mon] = np.average(tas_obs_ano_masked * (lat_hadcrut5 <= -60.) * (lon_hadcrut5 >= -130.) * (lon_hadcrut5 <= -60.) * np.isfinite(sst_hadsst4),weights=grid_area_hadcrut5 * (lat_hadcrut5 <= -60.) * (lon_hadcrut5 >= -130.) * (lon_hadcrut5 <= -60.) * np.isfinite(sst_hadsst4))
        tas_ws_ano[mon] = np.average(tas_obs_ano_masked * (lat_hadcrut5 <= -60.) * (lon_hadcrut5 >= -60.) * (lon_hadcrut5 <= 20.) * np.isfinite(sst_hadsst4),weights=grid_area_hadcrut5 * (lat_hadcrut5 <= -60.) * (lon_hadcrut5 >= -60.) * (lon_hadcrut5 <= 20.) * np.isfinite(sst_hadsst4))
        tas_io_ano[mon] = np.average(tas_obs_ano_masked * (lat_hadcrut5 <= -60.) * (lon_hadcrut5 >= 20.) * (lon_hadcrut5 <= 90.) * np.isfinite(sst_hadsst4),weights=grid_area_hadcrut5 * (lat_hadcrut5 <= -60.) * (lon_hadcrut5 >= 20.) * (lon_hadcrut5 <= 90.) * np.isfinite(sst_hadsst4))
        tas_wpo_ano[mon] = np.average(tas_obs_ano_masked * (lat_hadcrut5 <= -60.) * (lon_hadcrut5 >= 90.) * (lon_hadcrut5 <= 160.) * np.isfinite(sst_hadsst4),weights=grid_area_hadcrut5 * (lat_hadcrut5 <= -60.) * (lon_hadcrut5 >= 90.) * (lon_hadcrut5 <= 160.) * np.isfinite(sst_hadsst4))
        tas_rs_ano[mon] = np.average(tas_obs_ano_masked * (index1 | index2) * np.isfinite(sst_hadsst4),weights=grid_area_hadcrut5 * (index1 | index2) * np.isfinite(sst_hadsst4))

    # Compute mean absolute surface temperature from HadCRUT5 (average over 1961-1990)
    tas_bas_cycle = np.zeros(nmy)
    tas_ws_cycle = np.zeros(nmy)
    tas_io_cycle = np.zeros(nmy)
    tas_wpo_cycle = np.zeros(nmy)
    tas_rs_cycle = np.zeros(nmy)
    for mon in np.arange(nmy):
        tas_bas_cycle[mon] = np.average(tas_obs_cycle[mon,:,:] * (lat_hadcrut5 <= -60.) * (lon_hadcrut5 >= -130.) * (lon_hadcrut5 <= -60.) * np.isfinite(sst_hadsst4),weights=grid_area_hadcrut5 * (lat_hadcrut5 <= -60.) * (lon_hadcrut5 >= -130.) * (lon_hadcrut5 <= -60.) * np.isfinite(sst_hadsst4))
        tas_ws_cycle[mon] = np.average(tas_obs_cycle[mon,:,:] * (lat_hadcrut5 <= -60.) * (lon_hadcrut5 >= -60.) * (lon_hadcrut5 <= 20.) * np.isfinite(sst_hadsst4),weights=grid_area_hadcrut5 * (lat_hadcrut5 <= -60.) * (lon_hadcrut5 >= -60.) * (lon_hadcrut5 <= 20.) * np.isfinite(sst_hadsst4))
        tas_io_cycle[mon] = np.average(tas_obs_cycle[mon,:,:] * (lat_hadcrut5 <= -60.) * (lon_hadcrut5 >= 20.) * (lon_hadcrut5 <= 90.) * np.isfinite(sst_hadsst4),weights=grid_area_hadcrut5 * (lat_hadcrut5 <= -60.) * (lon_hadcrut5 >= 20.) * (lon_hadcrut5 <= 90.) * np.isfinite(sst_hadsst4))
        tas_wpo_cycle[mon] = np.average(tas_obs_cycle[mon,:,:] * (lat_hadcrut5 <= -60.) * (lon_hadcrut5 >= 90.) * (lon_hadcrut5 <= 160.) * np.isfinite(sst_hadsst4),weights=grid_area_hadcrut5 * (lat_hadcrut5 <= -60.) * (lon_hadcrut5 >= 90.) * (lon_hadcrut5 <= 160.) * np.isfinite(sst_hadsst4))
        tas_rs_cycle[mon] = np.average(tas_obs_cycle[mon,:,:] * (index1 | index2) * np.isfinite(sst_hadsst4),weights=grid_area_hadcrut5 * (index1 | index2) * np.isfinite(sst_hadsst4))

    # Compute seasonal mean surface temperature anomalies from HadCRUT5 1970-2023
    nyears_obs = int(nm_obs/nmy)
    tas_bas_ano_mon = np.zeros((nyears_obs,nmy))
    tas_ws_ano_mon = np.zeros((nyears_obs,nmy))
    tas_io_ano_mon = np.zeros((nyears_obs,nmy))
    tas_wpo_ano_mon = np.zeros((nyears_obs,nmy))
    tas_rs_ano_mon = np.zeros((nyears_obs,nmy))
    for mon in np.arange(nmy):
        tas_bas_ano_mon[:,mon] = tas_bas_ano[mon::12]
        tas_ws_ano_mon[:,mon] = tas_ws_ano[mon::12]
        tas_io_ano_mon[:,mon] = tas_io_ano[mon::12]
        tas_wpo_ano_mon[:,mon] = tas_wpo_ano[mon::12]
        tas_rs_ano_mon[:,mon] = tas_rs_ano[mon::12]
    tas_bas_ano_seasonalmean = np.nanmean(tas_bas_ano_mon[:,index_start:index_end],axis=1)
    tas_ws_ano_seasonalmean = np.nanmean(tas_ws_ano_mon[:,index_start:index_end],axis=1)
    tas_io_ano_seasonalmean = np.nanmean(tas_io_ano_mon[:,index_start:index_end],axis=1)
    tas_wpo_ano_seasonalmean = np.nanmean(tas_wpo_ano_mon[:,index_start:index_end],axis=1)
    tas_rs_ano_seasonalmean = np.nanmean(tas_rs_ano_mon[:,index_start:index_end],axis=1)
    
    # Compute seasonal mean absolute surface temperature from HadCRUT5 (average over 1961-1990)
    tas_bas_cycle_seasonalmean = np.nanmean(tas_bas_cycle[index_start:index_end])
    tas_ws_cycle_seasonalmean = np.nanmean(tas_ws_cycle[index_start:index_end])
    tas_io_cycle_seasonalmean = np.nanmean(tas_io_cycle[index_start:index_end])
    tas_wpo_cycle_seasonalmean = np.nanmean(tas_wpo_cycle[index_start:index_end])
    tas_rs_cycle_seasonalmean = np.nanmean(tas_rs_cycle[index_start:index_end])
    
    # Compute seasonal mean absolute surface temperature from HadCRUT5 1970-2023
    driver_period_obs_bas = tas_bas_ano_seasonalmean + tas_bas_cycle_seasonalmean
    driver_period_obs_ws = tas_ws_ano_seasonalmean + tas_ws_cycle_seasonalmean
    driver_period_obs_io = tas_io_ano_seasonalmean + tas_io_cycle_seasonalmean
    driver_period_obs_wpo = tas_wpo_ano_seasonalmean + tas_wpo_cycle_seasonalmean
    driver_period_obs_rs = tas_rs_ano_seasonalmean + tas_rs_cycle_seasonalmean

elif driver == 'sst_mon':
    
    # Load SST OSTIA 1982-2023 and convert into degC
    filename = dir_ostia + 'OSTIA_SST_SecZ83.lat-80to-60.1982to2023_1m.nc'
    fh = Dataset(filename, mode='r')
    sst_obs_init = fh.variables['sst'][:]
    fh.close()
    sst_obs_init = sst_obs_init - 273.15
    nyears_obs = int(2023-1982+1)
    sst_obs_init2 = np.zeros((nyears_obs,nmy,6))
    for mon in np.arange(nmy):
        sst_obs_init2[:,mon,:] = sst_obs_init[mon::12,:]
    
    # Compute seasonal mean SST from OSTIA 1982-2023
    driver_period_obs_bas = np.nanmean(sst_obs_init2[:,index_start:index_end,1],axis=1)
    driver_period_obs_io = np.nanmean(sst_obs_init2[:,index_start:index_end,2],axis=1)
    driver_period_obs_rs = np.nanmean(sst_obs_init2[:,index_start:index_end,3],axis=1)
    driver_period_obs_wpo = np.nanmean(sst_obs_init2[:,index_start:index_end,4],axis=1)
    driver_period_obs_ws = np.nanmean(sst_obs_init2[:,index_start:index_end,5],axis=1)
        
# Save time series
if save_var == True:
    filename_ecearth = dir_output + driver + '_EC-Earth3_timeseries_' + season + '.npy'
    filename_cesm = dir_output + driver + '_CESM2_timeseries_' + season + '.npy'
    filename_mpi = dir_output + driver + '_MPI-ESM1-2-LR_timeseries_' + season + '.npy'
    filename_canesm = dir_output + driver + '_CanESM5_timeseries_' + season + '.npy'
    filename_access = dir_output + driver + '_ACCESS-ESM1-5_timeseries_' + season + '.npy'
    filename_obs = dir_output + driver + '_Obs_timeseries_' + season + '.npy'
    np.save(filename_ecearth,[driver_period_ecearth_bas,driver_period_ecearth_ws,driver_period_ecearth_io,driver_period_ecearth_wpo,driver_period_ecearth_rs])
    np.save(filename_cesm,[driver_period_cesm_bas,driver_period_cesm_ws,driver_period_cesm_io,driver_period_cesm_wpo,driver_period_cesm_rs])
    np.save(filename_mpi,[driver_period_mpi_bas,driver_period_mpi_ws,driver_period_mpi_io,driver_period_mpi_wpo,driver_period_mpi_rs])
    np.save(filename_canesm,[driver_period_canesm_bas,driver_period_canesm_ws,driver_period_canesm_io,driver_period_canesm_wpo,driver_period_canesm_rs])
    np.save(filename_access,[driver_period_access_bas,driver_period_access_ws,driver_period_access_io,driver_period_access_wpo,driver_period_access_rs])
    np.save(filename_obs,[driver_period_obs_bas,driver_period_obs_ws,driver_period_obs_io,driver_period_obs_wpo,driver_period_obs_rs])
   
# Plot options
xrange = np.arange(11,141,20)
name_xticks = ['1980','2000','2020','2040','2060','2080','2100']
if driver == 'SIE':
    nt_obs = 10
elif driver == 'tas_mon':
    nt_obs = 1
elif driver == 'sst_mon':
    nt_obs = 13
    
# Time series of original variables
fig,ax = plt.subplots(3,2,figsize=(24,24))
fig.subplots_adjust(left=0.08,bottom=0.05,right=0.95,top=0.95,hspace=0.2,wspace=0.2)

# Antarctic
if driver == 'mld_mon' or driver == 'tauv_mon':
    plt.delaxes(ax[0,0])
else:
    ax[0,0].plot(np.arange(np.size(driver_antarctic_access,1))+1,np.nanmean(driver_antarctic_access,axis=0),'-',color='orange',linewidth=4,label='ACCESS-ESM1.5')
    ax[0,0].plot(np.arange(np.size(driver_antarctic_canesm,1))+1,np.nanmean(driver_antarctic_canesm,axis=0),'-',color='green',linewidth=4,label='CanESM5')
    ax[0,0].plot(np.arange(np.size(driver_antarctic_cesm,1))+1,np.nanmean(driver_antarctic_cesm,axis=0),'-',color='blue',linewidth=4,label='CESM2')
    ax[0,0].plot(np.arange(np.size(driver_antarctic_ecearth,1))+1,np.nanmean(driver_antarctic_ecearth,axis=0),'-',color='red',linewidth=4,label='EC-Earth3')
    ax[0,0].plot(np.arange(np.size(driver_antarctic_mpi,1))+1,np.nanmean(driver_antarctic_mpi,axis=0),'-',color='gray',linewidth=4,label='MPI-ESM1.2-LR')
    ax[0,0].plot(np.arange(np.size(driver_antarctic_obs))+nt_obs,driver_antarctic_obs,'k.-',linewidth=2,label='Observations')
    ax[0,0].fill_between(np.arange(np.size(driver_antarctic_ecearth,1))+1,np.nanmin(driver_antarctic_ecearth,axis=0),np.nanmax(driver_antarctic_ecearth,axis=0),color='red',alpha=0.1)
    ax[0,0].fill_between(np.arange(np.size(driver_antarctic_cesm,1))+1,np.nanmin(driver_antarctic_cesm,axis=0),np.nanmax(driver_antarctic_cesm,axis=0),color='blue',alpha=0.1)
    ax[0,0].fill_between(np.arange(np.size(driver_antarctic_mpi,1))+1,np.nanmin(driver_antarctic_mpi,axis=0),np.nanmax(driver_antarctic_mpi,axis=0),color='gray',alpha=0.1)
    ax[0,0].fill_between(np.arange(np.size(driver_antarctic_canesm,1))+1,np.nanmin(driver_antarctic_canesm,axis=0),np.nanmax(driver_antarctic_canesm,axis=0),color='green',alpha=0.1)
    ax[0,0].fill_between(np.arange(np.size(driver_antarctic_access,1))+1,np.nanmin(driver_antarctic_access,axis=0),np.nanmax(driver_antarctic_access,axis=0),color='orange',alpha=0.1)
    ax[0,0].set_ylabel('Total Antarctic ' + string_driver_plot,fontsize=26)
    ax[0,0].set_xticks(xrange)
    ax[0,0].set_xticklabels(name_xticks)
    ax[0,0].tick_params(labelsize=20)
    ax[0,0].grid(linestyle='--')
    if driver == 'SIE':
        ax[0,0].axis([-1, 133, -0.5, 24])
        ax[0,0].legend(loc='lower left',fontsize=22,shadow=True,frameon=False,ncol=2)
    elif driver == 'sst_mon':
        ax[0,0].axis([-1, 133, -2, 3])
        ax[0,0].legend(loc='upper left',fontsize=22,shadow=True,frameon=False,ncol=2)
    elif driver == 'tas_mon':
        if season == 'OND':
            ax[0,0].axis([-1, 133, -10, 4])
        else:
            ax[0,0].axis([-1, 133, -22, 2])
        ax[0,0].legend(loc='upper left',fontsize=22,shadow=True,frameon=False,ncol=2)
    ax[0,0].set_title('a',loc='left',fontsize=30,fontweight='bold')

# Bellingshausen-Amundsen Seas
ax[0,1].plot(np.arange(np.size(driver_period_access_bas,1))+1,np.nanmean(driver_period_access_bas,axis=0),'-',color='orange',linewidth=4,label='ACCESS-ESM1.5')
ax[0,1].plot(np.arange(np.size(driver_period_canesm_bas,1))+1,np.nanmean(driver_period_canesm_bas,axis=0),'-',color='green',linewidth=4,label='CanESM5')
ax[0,1].plot(np.arange(np.size(driver_period_cesm_bas,1))+1,np.nanmean(driver_period_cesm_bas,axis=0),'-',color='blue',linewidth=4,label='CESM2')
ax[0,1].plot(np.arange(np.size(driver_period_ecearth_bas,1))+1,np.nanmean(driver_period_ecearth_bas,axis=0),'-',color='red',linewidth=4,label='EC-Earth3')
ax[0,1].plot(np.arange(np.size(driver_period_mpi_bas,1))+1,np.nanmean(driver_period_mpi_bas,axis=0),'-',color='gray',linewidth=4,label='MPI-ESM1.2-LR')
ax[0,1].plot(np.arange(np.size(driver_period_obs_bas))+nt_obs,driver_period_obs_bas,'k.-',linewidth=2,label='Observations')
ax[0,1].fill_between(np.arange(np.size(driver_period_ecearth_bas,1))+1,np.nanmin(driver_period_ecearth_bas,axis=0),np.nanmax(driver_period_ecearth_bas,axis=0),color='red',alpha=0.1)
ax[0,1].fill_between(np.arange(np.size(driver_period_cesm_bas,1))+1,np.nanmin(driver_period_cesm_bas,axis=0),np.nanmax(driver_period_cesm_bas,axis=0),color='blue',alpha=0.1)
ax[0,1].fill_between(np.arange(np.size(driver_period_mpi_bas,1))+1,np.nanmin(driver_period_mpi_bas,axis=0),np.nanmax(driver_period_mpi_bas,axis=0),color='gray',alpha=0.1)
ax[0,1].fill_between(np.arange(np.size(driver_period_canesm_bas,1))+1,np.nanmin(driver_period_canesm_bas,axis=0),np.nanmax(driver_period_canesm_bas,axis=0),color='green',alpha=0.1)
ax[0,1].fill_between(np.arange(np.size(driver_period_access_bas,1))+1,np.nanmin(driver_period_access_bas,axis=0),np.nanmax(driver_period_access_bas,axis=0),color='orange',alpha=0.1)
ax[0,1].set_ylabel('Bellingshausen-Amundsen ' + string_driver_plot,fontsize=26)
ax[0,1].set_xticks(xrange)
ax[0,1].set_xticklabels(name_xticks)
ax[0,1].tick_params(labelsize=20)
ax[0,1].grid(linestyle='--')
if driver == 'SIE':
    ax[0,1].axis([-1, 133, -0.2, 6])
    ax[0,1].legend(loc='upper right',fontsize=22,shadow=True,frameon=False,ncol=2)
elif driver == 'sst_mon':
    ax[0,1].axis([-1, 133, -2, 5])
    ax[0,1].legend(loc='upper left',fontsize=22,shadow=True,frameon=False,ncol=2)
elif driver == 'tas_mon':
    if season == 'OND':
        ax[0,1].axis([-1, 133, -10, 6])
    else:
        ax[0,1].axis([-1, 133, -22, 4])
    ax[0,1].legend(loc='lower right',fontsize=22,shadow=True,frameon=False,ncol=2)
ax[0,1].set_title('b',loc='left',fontsize=30,fontweight='bold')

# Weddell Sea
ax[1,0].plot(np.arange(np.size(driver_period_access_ws,1))+1,np.nanmean(driver_period_access_ws,axis=0),'-',color='orange',linewidth=4,label='ACCESS-ESM1.5')
ax[1,0].plot(np.arange(np.size(driver_period_canesm_ws,1))+1,np.nanmean(driver_period_canesm_ws,axis=0),'-',color='green',linewidth=4,label='CanESM5')
ax[1,0].plot(np.arange(np.size(driver_period_cesm_ws,1))+1,np.nanmean(driver_period_cesm_ws,axis=0),'-',color='blue',linewidth=4,label='CESM2')
ax[1,0].plot(np.arange(np.size(driver_period_ecearth_ws,1))+1,np.nanmean(driver_period_ecearth_ws,axis=0),'-',color='red',linewidth=4,label='EC-Earth3')
ax[1,0].plot(np.arange(np.size(driver_period_mpi_ws,1))+1,np.nanmean(driver_period_mpi_ws,axis=0),'-',color='gray',linewidth=4,label='MPI-ESM1.2-LR')
ax[1,0].plot(np.arange(np.size(driver_period_obs_ws))+nt_obs,driver_period_obs_ws,'k.-',linewidth=2,label='Observations')
ax[1,0].fill_between(np.arange(np.size(driver_period_ecearth_ws,1))+1,np.nanmin(driver_period_ecearth_ws,axis=0),np.nanmax(driver_period_ecearth_ws,axis=0),color='red',alpha=0.1)
ax[1,0].fill_between(np.arange(np.size(driver_period_cesm_ws,1))+1,np.nanmin(driver_period_cesm_ws,axis=0),np.nanmax(driver_period_cesm_ws,axis=0),color='blue',alpha=0.1)
ax[1,0].fill_between(np.arange(np.size(driver_period_mpi_ws,1))+1,np.nanmin(driver_period_mpi_ws,axis=0),np.nanmax(driver_period_mpi_ws,axis=0),color='gray',alpha=0.1)
ax[1,0].fill_between(np.arange(np.size(driver_period_canesm_ws,1))+1,np.nanmin(driver_period_canesm_ws,axis=0),np.nanmax(driver_period_canesm_ws,axis=0),color='green',alpha=0.1)
ax[1,0].fill_between(np.arange(np.size(driver_period_access_ws,1))+1,np.nanmin(driver_period_access_ws,axis=0),np.nanmax(driver_period_access_ws,axis=0),color='orange',alpha=0.1)
ax[1,0].set_ylabel('Weddell ' + string_driver_plot,fontsize=26)
ax[1,0].set_xticks(xrange)
ax[1,0].set_xticklabels(name_xticks)
ax[1,0].tick_params(labelsize=20)
ax[1,0].grid(linestyle='--')
if driver == 'SIE':
    ax[1,0].axis([-1, 133, -0.2, 8])
    ax[1,0].legend(loc='lower left',fontsize=22,shadow=True,frameon=False,ncol=2)
elif driver == 'sst_mon':
    ax[1,0].axis([-1, 133, -2, 3])
    ax[1,0].legend(loc='upper left',fontsize=22,shadow=True,frameon=False,ncol=2)
elif driver == 'tas_mon':
    if season == 'OND':
        ax[1,0].axis([-1, 133, -10, 4])
    else:
        ax[1,0].axis([-1, 133, -26, 0])
    ax[1,0].legend(loc='upper left',fontsize=22,shadow=True,frameon=False,ncol=2)
ax[1,0].set_title('c',loc='left',fontsize=30,fontweight='bold')

# Indian Ocean
ax[1,1].plot(np.arange(np.size(driver_period_access_io,1))+1,np.nanmean(driver_period_access_io,axis=0),'-',color='orange',linewidth=4,label='ACCESS-ESM1.5')
ax[1,1].plot(np.arange(np.size(driver_period_canesm_io,1))+1,np.nanmean(driver_period_canesm_io,axis=0),'-',color='green',linewidth=4,label='CanESM5')
ax[1,1].plot(np.arange(np.size(driver_period_cesm_io,1))+1,np.nanmean(driver_period_cesm_io,axis=0),'-',color='blue',linewidth=4,label='CESM2')
ax[1,1].plot(np.arange(np.size(driver_period_ecearth_io,1))+1,np.nanmean(driver_period_ecearth_io,axis=0),'-',color='red',linewidth=4,label='EC-Earth3')
ax[1,1].plot(np.arange(np.size(driver_period_mpi_io,1))+1,np.nanmean(driver_period_mpi_io,axis=0),'-',color='gray',linewidth=4,label='MPI-ESM1.2-LR')
ax[1,1].plot(np.arange(np.size(driver_period_obs_io))+nt_obs,driver_period_obs_io,'k.-',linewidth=2,label='Observations')
ax[1,1].fill_between(np.arange(np.size(driver_period_ecearth_io,1))+1,np.nanmin(driver_period_ecearth_io,axis=0),np.nanmax(driver_period_ecearth_io,axis=0),color='red',alpha=0.1)
ax[1,1].fill_between(np.arange(np.size(driver_period_cesm_io,1))+1,np.nanmin(driver_period_cesm_io,axis=0),np.nanmax(driver_period_cesm_io,axis=0),color='blue',alpha=0.1)
ax[1,1].fill_between(np.arange(np.size(driver_period_mpi_io,1))+1,np.nanmin(driver_period_mpi_io,axis=0),np.nanmax(driver_period_mpi_io,axis=0),color='gray',alpha=0.1)
ax[1,1].fill_between(np.arange(np.size(driver_period_canesm_io,1))+1,np.nanmin(driver_period_canesm_io,axis=0),np.nanmax(driver_period_canesm_io,axis=0),color='green',alpha=0.1)
ax[1,1].fill_between(np.arange(np.size(driver_period_access_io,1))+1,np.nanmin(driver_period_access_io,axis=0),np.nanmax(driver_period_access_io,axis=0),color='orange',alpha=0.1)
ax[1,1].set_ylabel('Indian Ocean ' + string_driver_plot,fontsize=26)
ax[1,1].set_xticks(xrange)
ax[1,1].set_xticklabels(name_xticks)
ax[1,1].tick_params(labelsize=20)
ax[1,1].grid(linestyle='--')
if driver == 'SIE':
    ax[1,1].axis([-1, 133, -0.2, 6])
    ax[1,1].legend(loc='upper right',fontsize=22,shadow=True,frameon=False,ncol=2)
elif driver == 'sst_mon':
    ax[1,1].axis([-1, 133, -2, 3])
    ax[1,1].legend(loc='upper left',fontsize=22,shadow=True,frameon=False,ncol=2)
elif driver == 'tas_mon':
    if season == 'OND':
        ax[1,1].axis([-1, 133, -10, 4])
        ax[1,1].legend(loc='upper left',fontsize=22,shadow=True,frameon=False,ncol=2)
    else:
        ax[1,1].axis([-1, 133, -26, 0])
        ax[1,1].legend(loc='lower right',fontsize=22,shadow=True,frameon=False,ncol=2)
ax[1,1].set_title('d',loc='left',fontsize=30,fontweight='bold')

# Western Pacific Ocean
ax[2,0].plot(np.arange(np.size(driver_period_access_wpo,1))+1,np.nanmean(driver_period_access_wpo,axis=0),'-',color='orange',linewidth=4,label='ACCESS-ESM1.5')
ax[2,0].plot(np.arange(np.size(driver_period_canesm_wpo,1))+1,np.nanmean(driver_period_canesm_wpo,axis=0),'-',color='green',linewidth=4,label='CanESM5')
ax[2,0].plot(np.arange(np.size(driver_period_cesm_wpo,1))+1,np.nanmean(driver_period_cesm_wpo,axis=0),'-',color='blue',linewidth=4,label='CESM2')
ax[2,0].plot(np.arange(np.size(driver_period_ecearth_wpo,1))+1,np.nanmean(driver_period_ecearth_wpo,axis=0),'-',color='red',linewidth=4,label='EC-Earth3')
ax[2,0].plot(np.arange(np.size(driver_period_mpi_wpo,1))+1,np.nanmean(driver_period_mpi_wpo,axis=0),'-',color='gray',linewidth=4,label='MPI-ESM1.2-LR')
ax[2,0].plot(np.arange(np.size(driver_period_obs_wpo))+nt_obs,driver_period_obs_wpo,'k.-',linewidth=2,label='Observations')
ax[2,0].fill_between(np.arange(np.size(driver_period_ecearth_wpo,1))+1,np.nanmin(driver_period_ecearth_wpo,axis=0),np.nanmax(driver_period_ecearth_wpo,axis=0),color='red',alpha=0.1)
ax[2,0].fill_between(np.arange(np.size(driver_period_cesm_wpo,1))+1,np.nanmin(driver_period_cesm_wpo,axis=0),np.nanmax(driver_period_cesm_wpo,axis=0),color='blue',alpha=0.1)
ax[2,0].fill_between(np.arange(np.size(driver_period_mpi_wpo,1))+1,np.nanmin(driver_period_mpi_wpo,axis=0),np.nanmax(driver_period_mpi_wpo,axis=0),color='gray',alpha=0.1)
ax[2,0].fill_between(np.arange(np.size(driver_period_canesm_wpo,1))+1,np.nanmin(driver_period_canesm_wpo,axis=0),np.nanmax(driver_period_canesm_wpo,axis=0),color='green',alpha=0.1)
ax[2,0].fill_between(np.arange(np.size(driver_period_access_wpo,1))+1,np.nanmin(driver_period_access_wpo,axis=0),np.nanmax(driver_period_access_wpo,axis=0),color='orange',alpha=0.1)
ax[2,0].set_ylabel('Western Pacific Ocean ' + string_driver_plot,fontsize=26)
ax[2,0].set_xticks(xrange)
ax[2,0].set_xticklabels(name_xticks)
ax[2,0].tick_params(labelsize=20)
ax[2,0].grid(linestyle='--')
if driver == 'SIE':
    ax[2,0].axis([-1, 133, -0.2, 6])
    ax[2,0].legend(loc='upper right',fontsize=22,shadow=True,frameon=False,ncol=2)
elif driver == 'sst_mon':
    ax[2,0].axis([-1, 133, -2, 3])
    ax[2,0].legend(loc='upper left',fontsize=22,shadow=True,frameon=False,ncol=2)
elif driver == 'tas_mon':
    if season == 'OND':
        ax[2,0].axis([-1, 133, -12, 4])
        ax[2,0].legend(loc='upper left',fontsize=22,shadow=True,frameon=False,ncol=2)
    else:
        ax[2,0].axis([-1, 133, -22, 0])
        ax[2,0].legend(loc='lower right',fontsize=22,shadow=True,frameon=False,ncol=2)
ax[2,0].set_title('e',loc='left',fontsize=30,fontweight='bold')

# Ross Sea
ax[2,1].plot(np.arange(np.size(driver_period_access_rs,1))+1,np.nanmean(driver_period_access_rs,axis=0),'-',color='orange',linewidth=4,label='ACCESS-ESM1.5')
ax[2,1].plot(np.arange(np.size(driver_period_canesm_rs,1))+1,np.nanmean(driver_period_canesm_rs,axis=0),'-',color='green',linewidth=4,label='CanESM5')
ax[2,1].plot(np.arange(np.size(driver_period_cesm_rs,1))+1,np.nanmean(driver_period_cesm_rs,axis=0),'-',color='blue',linewidth=4,label='CESM2')
ax[2,1].plot(np.arange(np.size(driver_period_ecearth_bas,1))+1,np.nanmean(driver_period_ecearth_rs,axis=0),'-',color='red',linewidth=4,label='EC-Earth3')
ax[2,1].plot(np.arange(np.size(driver_period_mpi_rs,1))+1,np.nanmean(driver_period_mpi_rs,axis=0),'-',color='gray',linewidth=4,label='MPI-ESM1.2-LR')
ax[2,1].plot(np.arange(np.size(driver_period_obs_rs))+nt_obs,driver_period_obs_rs,'k.-',linewidth=2,label='Observations')
ax[2,1].fill_between(np.arange(np.size(driver_period_ecearth_rs,1))+1,np.nanmin(driver_period_ecearth_rs,axis=0),np.nanmax(driver_period_ecearth_rs,axis=0),color='red',alpha=0.1)
ax[2,1].fill_between(np.arange(np.size(driver_period_cesm_rs,1))+1,np.nanmin(driver_period_cesm_rs,axis=0),np.nanmax(driver_period_cesm_rs,axis=0),color='blue',alpha=0.1)
ax[2,1].fill_between(np.arange(np.size(driver_period_mpi_rs,1))+1,np.nanmin(driver_period_mpi_rs,axis=0),np.nanmax(driver_period_mpi_rs,axis=0),color='gray',alpha=0.1)
ax[2,1].fill_between(np.arange(np.size(driver_period_canesm_rs,1))+1,np.nanmin(driver_period_canesm_rs,axis=0),np.nanmax(driver_period_canesm_rs,axis=0),color='green',alpha=0.1)
ax[2,1].fill_between(np.arange(np.size(driver_period_access_rs,1))+1,np.nanmin(driver_period_access_rs,axis=0),np.nanmax(driver_period_access_rs,axis=0),color='orange',alpha=0.1)
ax[2,1].set_ylabel('Ross ' + string_driver_plot,fontsize=26)
ax[2,1].set_xticks(xrange)
ax[2,1].set_xticklabels(name_xticks)
ax[2,1].tick_params(labelsize=20)
ax[2,1].grid(linestyle='--')
if driver == 'SIE':
    ax[2,1].axis([-1, 133, -0.2, 6])
    ax[2,1].legend(loc='lower left',fontsize=22,shadow=True,frameon=False,ncol=2)
elif driver == 'sst_mon':
    ax[2,1].axis([-1, 133, -2, 3])
    ax[2,1].legend(loc='upper left',fontsize=22,shadow=True,frameon=False,ncol=2)
elif driver == 'tas_mon':
    if season == 'OND':
        ax[2,1].axis([-1, 133, -10, 4])
    else:
        ax[2,1].axis([-1, 133, -22, 0])
    ax[2,1].legend(loc='upper left',fontsize=22,shadow=True,frameon=False,ncol=2)
ax[2,1].set_title('f',loc='left',fontsize=30,fontweight='bold')

# Save Fig.
if save_fig == True:
    if driver == 'SIE':
        fig.savefig(dir_fig + 'fig_a5.pdf')
    elif driver == 'tas_mon':
        fig.savefig(dir_fig + 'fig_a6.pdf')
    elif driver == 'sst_mon':
        fig.savefig(dir_fig + 'fig_a7.pdf')