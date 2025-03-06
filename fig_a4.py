#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Figure A4: Time series of summer SIE (target) for different Antarctic sectors

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
nmy = 12 # number of months in a year
save_fig = True
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
dir_input = '/home/dadocq/Documents/Models/'
dir_input2 = '/home/dadocq/Documents/Papers/My_Papers/RESIST_Antarctic/output/seasons/'
dir_osisaf = '/home/dadocq/Documents/Observations/OSISAF/' # Observed sea-ice extent
dir_output = dir_input2
dir_fig = '/home/dadocq/Documents/Papers/My_Papers/RESIST_Antarctic/LaTeX/'

# Load Antarctic sea-ice extent EC-Earth3
filename = dir_input2 + 'EC-Earth3_Antarctic_timeseries_JAS.npy'
sie_summer_ecearth,notused,notused,notused,notused,notused,notused,notused = np.load(filename,allow_pickle=True)

# Load Antarctic sea-ice extent CESM2
filename = dir_input2 + 'CESM2_Antarctic_timeseries_JAS.npy'
sie_summer_cesm,notused,notused,notused,notused,notused,notused,notused = np.load(filename,allow_pickle=True)

# Load Antarctic sea-ice extent MPI-ESM1-2-LR
filename = dir_input2 + 'MPI-ESM1-2-LR_Antarctic_timeseries_JAS.npy'
sie_summer_mpi,notused,notused,notused,notused,notused,notused,notused = np.load(filename,allow_pickle=True)

# Load Antarctic sea-ice extent CanESM5
filename = dir_input2 + 'CanESM5_Antarctic_timeseries_JAS.npy'
sie_summer_canesm,notused,notused,notused,notused,notused,notused,notused = np.load(filename,allow_pickle=True)

# Load Antarctic sea-ice extent ACCESS-ESM1-5
filename = dir_input2 + 'ACCESS-ESM1-5_Antarctic_timeseries_JAS.npy'
sie_summer_access,notused,notused,notused,notused,notused,notused,notused = np.load(filename,allow_pickle=True)

# Load regional sea-ice extent from ACCESS during historical summer (1970-2014)
filename = dir_input + 'ACCESS-ESM1-5/SIE_AntReg_ACCESS-ESM1-5_historical.npy'
sie_access_bas_hist,sie_access_ws_hist,sie_access_io_hist,sie_access_wpo_hist,sie_access_rs_hist = np.load(filename,allow_pickle=True)
n_members_access = np.size(sie_access_bas_hist,0)
nyears_hist = np.size(sie_access_bas_hist,1)

# Load regional sea-ice extent from ACCESS during future summer (2015-2100)
filename = dir_input + 'ACCESS-ESM1-5/SIE_AntReg_ACCESS-ESM1-5_ssp370.npy'
sie_access_bas_ssp,sie_access_ws_ssp,sie_access_io_ssp,sie_access_wpo_ssp,sie_access_rs_ssp = np.load(filename,allow_pickle=True)
nyears_ssp = np.size(sie_access_bas_ssp,1)

# Load regional sea-ice extent from CanESM during historical summer (1970-2014)
filename = dir_input + 'CanESM5/SIE_AntReg_CanESM5_historical.npy'
sie_canesm_bas_hist,sie_canesm_ws_hist,sie_canesm_io_hist,sie_canesm_wpo_hist,sie_canesm_rs_hist = np.load(filename,allow_pickle=True)
n_members_canesm = np.size(sie_canesm_bas_hist,0)

# Load regional sea-ice extent from CanESM during future summer (2015-2100)
filename = dir_input + 'CanESM5/SIE_AntReg_CanESM5_ssp370.npy'
sie_canesm_bas_ssp,sie_canesm_ws_ssp,sie_canesm_io_ssp,sie_canesm_wpo_ssp,sie_canesm_rs_ssp = np.load(filename,allow_pickle=True)

# Load regional sea-ice extent from MPI during historical summer (1970-2014)
filename = dir_input + 'MPI-ESM1-2-LR/SIE_AntReg_MPI-ESM1-2-LR_historical.npy'
sie_mpi_bas_hist,sie_mpi_ws_hist,sie_mpi_io_hist,sie_mpi_wpo_hist,sie_mpi_rs_hist = np.load(filename,allow_pickle=True)
n_members_mpi = np.size(sie_mpi_bas_hist,0)

# Load regional sea-ice extent from MPI during future summer (2015-2100)
filename = dir_input + 'MPI-ESM1-2-LR/SIE_AntReg_MPI-ESM1-2-LR_ssp370.npy'
sie_mpi_bas_ssp,sie_mpi_ws_ssp,sie_mpi_io_ssp,sie_mpi_wpo_ssp,sie_mpi_rs_ssp = np.load(filename,allow_pickle=True)

# Load regional sea-ice extent from CESM during historical summer (1970-2014)
filename = dir_input + 'CESM2-LE/SIE_AntReg_CESM2-LE_historical.npy'
sie_cesm_bas_hist,sie_cesm_ws_hist,sie_cesm_io_hist,sie_cesm_wpo_hist,sie_cesm_rs_hist = np.load(filename,allow_pickle=True)
n_members_cesm = np.size(sie_cesm_bas_hist,0)

# Load regional sea-ice extent from CESM during future summer (2015-2100)
filename = dir_input + 'CESM2-LE/SIE_AntReg_CESM2-LE_ssp370.npy'
sie_cesm_bas_ssp,sie_cesm_ws_ssp,sie_cesm_io_ssp,sie_cesm_wpo_ssp,sie_cesm_rs_ssp = np.load(filename,allow_pickle=True)

# Load regional sea-ice extent from EC-Earth3 during historical summer (1970-2014)
filename = dir_input + 'SMHI-LENS/input/SIE_AntReg_SMHI-LENS_historical.npy'
sie_ecearth_bas_hist,sie_ecearth_ws_hist,sie_ecearth_io_hist,sie_ecearth_wpo_hist,sie_ecearth_rs_hist = np.load(filename,allow_pickle=True)
n_members_ecearth = np.size(sie_ecearth_bas_hist,0)

# Load regional sea-ice extent from EC-Earth3 during future summer (2015-2100)
filename = dir_input + 'SMHI-LENS/input/SIE_AntReg_SMHI-LENS_ssp370.npy'
sie_ecearth_bas_ssp,sie_ecearth_ws_ssp,sie_ecearth_io_ssp,sie_ecearth_wpo_ssp,sie_ecearth_rs_ssp = np.load(filename,allow_pickle=True)

# Load Antarctic SIE from OSI SAF (OSI-420) 1979-2024 (until Nov 2024)
# https://osi-saf.eumetsat.int/products/osi-420
filename = dir_osisaf + 'sh_indices/osisaf_sh_sie_monthly.nc'
fh = Dataset(filename, mode='r')
sie_obs_init = fh.variables['sie'][:]
fh.close()
sie_obs_init2 = sie_obs_init[12:552] # 1979-2023
sie_obs_jan2024 = sie_obs_init[552] # Jan 2024
sie_obs_feb2024 = sie_obs_init[553] # Feb 2024
sie_obs_init2[sie_obs_init2<0.] = np.nan # replace negative values by NaN
nyears_obs = 45
sie_obs = np.zeros((nyears_obs,nmy))
for mon in np.arange(nmy):
    sie_obs[:,mon] = sie_obs_init2[mon:551:12]
    sie_obs[:,mon] = interpolate_nan(sie_obs[:,mon]) # interpolate if NaN
    
# Load BAS SIE from OSI SAF (OSI-420) 1979-2024 (until Nov 2024)
filename = dir_osisaf + 'sh_indices/osisaf_bell_sie_monthly.nc'
fh = Dataset(filename, mode='r')
sie_obs_init = fh.variables['sie'][:]
fh.close()
sie_obs_init2 = sie_obs_init[12:552] # 1979-2023
sie_obs_bas_jan2024 = sie_obs_init[552] # Jan 2024
sie_obs_bas_feb2024 = sie_obs_init[553] # Feb 2024
sie_obs_init2[sie_obs_init2<0.] = np.nan
sie_obs_bas = np.zeros((nyears_obs,nmy))
for mon in np.arange(nmy):
    sie_obs_bas[:,mon] = sie_obs_init2[mon::12]
    sie_obs_bas[:,mon] = interpolate_nan(sie_obs_bas[:,mon])
    
# Load WS SIE from OSI SAF (OSI-420) 1979-2024 (until Nov 2024)
filename = dir_osisaf + 'sh_indices/osisaf_wedd_sie_monthly.nc'
fh = Dataset(filename, mode='r')
sie_obs_init = fh.variables['sie'][:]
fh.close()
sie_obs_init2 = sie_obs_init[12:552] # 1979-2023
sie_obs_ws_jan2024 = sie_obs_init[552] # Jan 2024
sie_obs_ws_feb2024 = sie_obs_init[553] # Feb 2024
sie_obs_init2[sie_obs_init2<0.] = np.nan
sie_obs_ws = np.zeros((nyears_obs,nmy))
for mon in np.arange(nmy):
    sie_obs_ws[:,mon] = sie_obs_init2[mon::12]
    sie_obs_ws[:,mon] = interpolate_nan(sie_obs_ws[:,mon])

# Load IO SIE from OSI SAF (OSI-420) 1979-2024 (until Nov 2024)
filename = dir_osisaf + 'sh_indices/osisaf_indi_sie_monthly.nc'
fh = Dataset(filename, mode='r')
sie_obs_init = fh.variables['sie'][:]
fh.close()
sie_obs_init2 = sie_obs_init[12:552] # 1979-2023
sie_obs_io_jan2024 = sie_obs_init[552] # Jan 2024
sie_obs_io_feb2024 = sie_obs_init[553] # Feb 2024
sie_obs_init2[sie_obs_init2<0.] = np.nan
sie_obs_io = np.zeros((nyears_obs,nmy))
for mon in np.arange(nmy):
    sie_obs_io[:,mon] = sie_obs_init2[mon::12]
    sie_obs_io[:,mon] = interpolate_nan(sie_obs_io[:,mon])

# Load WPO SIE from OSI SAF (OSI-420) 1979-2024 (until Nov 2024)
filename = dir_osisaf + 'sh_indices/osisaf_wpac_sie_monthly.nc'
fh = Dataset(filename, mode='r')
sie_obs_init = fh.variables['sie'][:]
fh.close()
sie_obs_init2 = sie_obs_init[12:552] # 1979-2023
sie_obs_wpo_jan2024 = sie_obs_init[552] # Jan 2024
sie_obs_wpo_feb2024 = sie_obs_init[553] # Feb 2024
sie_obs_init2[sie_obs_init2<0.] = np.nan
sie_obs_wpo = np.zeros((nyears_obs,nmy))
for mon in np.arange(nmy):
    sie_obs_wpo[:,mon] = sie_obs_init2[mon::12]
    sie_obs_wpo[:,mon] = interpolate_nan(sie_obs_wpo[:,mon])

# Load RS SIE from OSI SAF (OSI-420) 1979-2024 (until Nov 2024)
filename = dir_osisaf + 'sh_indices/osisaf_ross_sie_monthly.nc'
fh = Dataset(filename, mode='r')
sie_obs_init = fh.variables['sie'][:]
fh.close()
sie_obs_init2 = sie_obs_init[12:552] # 1979-2023
sie_obs_rs_jan2024 = sie_obs_init[552] # Jan 2024
sie_obs_rs_feb2024 = sie_obs_init[553] # Feb 2024
sie_obs_init2[sie_obs_init2<0.] = np.nan
for m in np.arange(1,np.size(sie_obs_init2)-1):
    if np.isnan(sie_obs_init2[m]) == True:
        sie_obs_init2[m] = (sie_obs_init2[m-1] + sie_obs_init2[m+1]) / 2.
sie_obs_rs = np.zeros((nyears_obs,nmy))
for mon in np.arange(nmy):
    sie_obs_rs[:,mon] = sie_obs_init2[mon::12]
    sie_obs_rs[:,mon] = interpolate_nan(sie_obs_rs[:,mon])

# Concatenate historical and future summers ACCESS (1970-2100)
nyears = nyears_hist + nyears_ssp
sie_access_bas = np.zeros((n_members_access,nyears,nmy))
sie_access_ws = np.zeros((n_members_access,nyears,nmy))
sie_access_io = np.zeros((n_members_access,nyears,nmy))
sie_access_wpo = np.zeros((n_members_access,nyears,nmy))
sie_access_rs = np.zeros((n_members_access,nyears,nmy))
for m in np.arange(n_members_access):
    for i in np.arange(nmy):
        sie_access_bas[m,:,i] = np.concatenate((sie_access_bas_hist[m,:,i],sie_access_bas_ssp[m,:,i]))
        sie_access_ws[m,:,i] = np.concatenate((sie_access_ws_hist[m,:,i],sie_access_ws_ssp[m,:,i]))
        sie_access_io[m,:,i] = np.concatenate((sie_access_io_hist[m,:,i],sie_access_io_ssp[m,:,i]))
        sie_access_wpo[m,:,i] = np.concatenate((sie_access_wpo_hist[m,:,i],sie_access_wpo_ssp[m,:,i]))
        sie_access_rs[m,:,i] = np.concatenate((sie_access_rs_hist[m,:,i],sie_access_rs_ssp[m,:,i]))

# Concatenate historical and future summers CanESM (1970-2100)
sie_canesm_bas = np.zeros((n_members_canesm,nyears,nmy))
sie_canesm_ws = np.zeros((n_members_canesm,nyears,nmy))
sie_canesm_io = np.zeros((n_members_canesm,nyears,nmy))
sie_canesm_wpo = np.zeros((n_members_canesm,nyears,nmy))
sie_canesm_rs = np.zeros((n_members_canesm,nyears,nmy))
for m in np.arange(n_members_canesm):
    for i in np.arange(nmy):
        sie_canesm_bas[m,:,i] = np.concatenate((sie_canesm_bas_hist[m,:,i],sie_canesm_bas_ssp[m,:,i]))
        sie_canesm_ws[m,:,i] = np.concatenate((sie_canesm_ws_hist[m,:,i],sie_canesm_ws_ssp[m,:,i]))
        sie_canesm_io[m,:,i] = np.concatenate((sie_canesm_io_hist[m,:,i],sie_canesm_io_ssp[m,:,i]))
        sie_canesm_wpo[m,:,i] = np.concatenate((sie_canesm_wpo_hist[m,:,i],sie_canesm_wpo_ssp[m,:,i]))
        sie_canesm_rs[m,:,i] = np.concatenate((sie_canesm_rs_hist[m,:,i],sie_canesm_rs_ssp[m,:,i]))
        
# Concatenate historical and future summers MPI (1970-2100)
sie_mpi_bas = np.zeros((n_members_mpi,nyears,nmy))
sie_mpi_ws = np.zeros((n_members_mpi,nyears,nmy))
sie_mpi_io = np.zeros((n_members_mpi,nyears,nmy))
sie_mpi_wpo = np.zeros((n_members_mpi,nyears,nmy))
sie_mpi_rs = np.zeros((n_members_mpi,nyears,nmy))
for m in np.arange(n_members_mpi):
    for i in np.arange(nmy):
        sie_mpi_bas[m,:,i] = np.concatenate((sie_mpi_bas_hist[m,:,i],sie_mpi_bas_ssp[m,:,i]))
        sie_mpi_ws[m,:,i] = np.concatenate((sie_mpi_ws_hist[m,:,i],sie_mpi_ws_ssp[m,:,i]))
        sie_mpi_io[m,:,i] = np.concatenate((sie_mpi_io_hist[m,:,i],sie_mpi_io_ssp[m,:,i]))
        sie_mpi_wpo[m,:,i] = np.concatenate((sie_mpi_wpo_hist[m,:,i],sie_mpi_wpo_ssp[m,:,i]))
        sie_mpi_rs[m,:,i] = np.concatenate((sie_mpi_rs_hist[m,:,i],sie_mpi_rs_ssp[m,:,i]))
        
# Concatenate historical and future summers CESM (1970-2100)
sie_cesm_bas = np.zeros((n_members_cesm,nyears,nmy))
sie_cesm_ws = np.zeros((n_members_cesm,nyears,nmy))
sie_cesm_io = np.zeros((n_members_cesm,nyears,nmy))
sie_cesm_wpo = np.zeros((n_members_cesm,nyears,nmy))
sie_cesm_rs = np.zeros((n_members_cesm,nyears,nmy))
for m in np.arange(n_members_cesm):
    for i in np.arange(nmy):
        sie_cesm_bas[m,:,i] = np.concatenate((sie_cesm_bas_hist[m,:,i],sie_cesm_bas_ssp[m,:,i]))
        sie_cesm_ws[m,:,i] = np.concatenate((sie_cesm_ws_hist[m,:,i],sie_cesm_ws_ssp[m,:,i]))
        sie_cesm_io[m,:,i] = np.concatenate((sie_cesm_io_hist[m,:,i],sie_cesm_io_ssp[m,:,i]))
        sie_cesm_wpo[m,:,i] = np.concatenate((sie_cesm_wpo_hist[m,:,i],sie_cesm_wpo_ssp[m,:,i]))
        sie_cesm_rs[m,:,i] = np.concatenate((sie_cesm_rs_hist[m,:,i],sie_cesm_rs_ssp[m,:,i]))
        
# Concatenate historical and future summers EC-Earth3 (1970-2100)
sie_ecearth_bas = np.zeros((n_members_ecearth,nyears,nmy))
sie_ecearth_ws = np.zeros((n_members_ecearth,nyears,nmy))
sie_ecearth_io = np.zeros((n_members_ecearth,nyears,nmy))
sie_ecearth_wpo = np.zeros((n_members_ecearth,nyears,nmy))
sie_ecearth_rs = np.zeros((n_members_ecearth,nyears,nmy))
for m in np.arange(n_members_ecearth):
    for i in np.arange(nmy):
        sie_ecearth_bas[m,:,i] = np.concatenate((sie_ecearth_bas_hist[m,:,i],sie_ecearth_bas_ssp[m,:,i]))
        sie_ecearth_ws[m,:,i] = np.concatenate((sie_ecearth_ws_hist[m,:,i],sie_ecearth_ws_ssp[m,:,i]))
        sie_ecearth_io[m,:,i] = np.concatenate((sie_ecearth_io_hist[m,:,i],sie_ecearth_io_ssp[m,:,i]))
        sie_ecearth_wpo[m,:,i] = np.concatenate((sie_ecearth_wpo_hist[m,:,i],sie_ecearth_wpo_ssp[m,:,i]))
        sie_ecearth_rs[m,:,i] = np.concatenate((sie_ecearth_rs_hist[m,:,i],sie_ecearth_rs_ssp[m,:,i]))
    
# Summer Antarctic SIE - ACCESS
sie_summer_access_bas = np.zeros((n_members_access,nyears))
sie_summer_access_ws = np.zeros((n_members_access,nyears))
sie_summer_access_io = np.zeros((n_members_access,nyears))
sie_summer_access_wpo = np.zeros((n_members_access,nyears))
sie_summer_access_rs = np.zeros((n_members_access,nyears))
for m in np.arange(n_members_access):
    sie_summer_access_bas[m,0] = np.nanmean(sie_access_bas[m,0,0:2])
    sie_summer_access_ws[m,0] = np.nanmean(sie_access_ws[m,0,0:2])
    sie_summer_access_io[m,0] = np.nanmean(sie_access_io[m,0,0:2])
    sie_summer_access_wpo[m,0] = np.nanmean(sie_access_wpo[m,0,0:2])
    sie_summer_access_rs[m,0] = np.nanmean(sie_access_rs[m,0,0:2])
    for year in np.arange(1,nyears):
        sie_conc_summer_access_bas = np.concatenate((sie_access_bas[m,year-1,11:12],sie_access_bas[m,year,0:2])) # concatenate DJF
        sie_summer_access_bas[m,year] = np.nanmean(sie_conc_summer_access_bas) # mean DJF
        sie_conc_summer_access_ws = np.concatenate((sie_access_ws[m,year-1,11:12],sie_access_ws[m,year,0:2])) # concatenate DJF
        sie_summer_access_ws[m,year] = np.nanmean(sie_conc_summer_access_ws) # mean DJF
        sie_conc_summer_access_io = np.concatenate((sie_access_io[m,year-1,11:12],sie_access_io[m,year,0:2])) # concatenate DJF
        sie_summer_access_io[m,year] = np.nanmean(sie_conc_summer_access_io) # mean DJF
        sie_conc_summer_access_wpo = np.concatenate((sie_access_wpo[m,year-1,11:12],sie_access_wpo[m,year,0:2])) # concatenate DJF
        sie_summer_access_wpo[m,year] = np.nanmean(sie_conc_summer_access_wpo) # mean DJF
        sie_conc_summer_access_rs = np.concatenate((sie_access_rs[m,year-1,11:12],sie_access_rs[m,year,0:2])) # concatenate DJF
        sie_summer_access_rs[m,year] = np.nanmean(sie_conc_summer_access_rs) # mean DJF

# Summer Antarctic SIE - CanESM
sie_summer_canesm_bas = np.zeros((n_members_canesm,nyears))
sie_summer_canesm_ws = np.zeros((n_members_canesm,nyears))
sie_summer_canesm_io = np.zeros((n_members_canesm,nyears))
sie_summer_canesm_wpo = np.zeros((n_members_canesm,nyears))
sie_summer_canesm_rs = np.zeros((n_members_canesm,nyears))
for m in np.arange(n_members_canesm):
    sie_summer_canesm_bas[m,0] = np.nanmean(sie_canesm_bas[m,0,0:2])
    sie_summer_canesm_ws[m,0] = np.nanmean(sie_canesm_ws[m,0,0:2])
    sie_summer_canesm_io[m,0] = np.nanmean(sie_canesm_io[m,0,0:2])
    sie_summer_canesm_wpo[m,0] = np.nanmean(sie_canesm_wpo[m,0,0:2])
    sie_summer_canesm_rs[m,0] = np.nanmean(sie_canesm_rs[m,0,0:2])
    for year in np.arange(1,nyears):
        sie_conc_summer_canesm_bas = np.concatenate((sie_canesm_bas[m,year-1,11:12],sie_canesm_bas[m,year,0:2])) # concatenate DJF
        sie_summer_canesm_bas[m,year] = np.nanmean(sie_conc_summer_canesm_bas) # mean DJF
        sie_conc_summer_canesm_ws = np.concatenate((sie_canesm_ws[m,year-1,11:12],sie_canesm_ws[m,year,0:2])) # concatenate DJF
        sie_summer_canesm_ws[m,year] = np.nanmean(sie_conc_summer_canesm_ws) # mean DJF
        sie_conc_summer_canesm_io = np.concatenate((sie_canesm_io[m,year-1,11:12],sie_canesm_io[m,year,0:2])) # concatenate DJF
        sie_summer_canesm_io[m,year] = np.nanmean(sie_conc_summer_canesm_io) # mean DJF
        sie_conc_summer_canesm_wpo = np.concatenate((sie_canesm_wpo[m,year-1,11:12],sie_canesm_wpo[m,year,0:2])) # concatenate DJF
        sie_summer_canesm_wpo[m,year] = np.nanmean(sie_conc_summer_canesm_wpo) # mean DJF
        sie_conc_summer_canesm_rs = np.concatenate((sie_canesm_rs[m,year-1,11:12],sie_canesm_rs[m,year,0:2])) # concatenate DJF
        sie_summer_canesm_rs[m,year] = np.nanmean(sie_conc_summer_canesm_rs) # mean DJF
    
# Summer Antarctic SIE - MPI
sie_summer_mpi_bas = np.zeros((n_members_mpi,nyears))
sie_summer_mpi_ws = np.zeros((n_members_mpi,nyears))
sie_summer_mpi_io = np.zeros((n_members_mpi,nyears))
sie_summer_mpi_wpo = np.zeros((n_members_mpi,nyears))
sie_summer_mpi_rs = np.zeros((n_members_mpi,nyears))
for m in np.arange(n_members_mpi):
    sie_summer_mpi_bas[m,0] = np.nanmean(sie_mpi_bas[m,0,0:2])
    sie_summer_mpi_ws[m,0] = np.nanmean(sie_mpi_ws[m,0,0:2])
    sie_summer_mpi_io[m,0] = np.nanmean(sie_mpi_io[m,0,0:2])
    sie_summer_mpi_wpo[m,0] = np.nanmean(sie_mpi_wpo[m,0,0:2])
    sie_summer_mpi_rs[m,0] = np.nanmean(sie_mpi_rs[m,0,0:2])
    for year in np.arange(1,nyears):
        sie_conc_summer_mpi_bas = np.concatenate((sie_mpi_bas[m,year-1,11:12],sie_mpi_bas[m,year,0:2])) # concatenate DJF
        sie_summer_mpi_bas[m,year] = np.nanmean(sie_conc_summer_mpi_bas) # mean DJF
        sie_conc_summer_mpi_ws = np.concatenate((sie_mpi_ws[m,year-1,11:12],sie_mpi_ws[m,year,0:2])) # concatenate DJF
        sie_summer_mpi_ws[m,year] = np.nanmean(sie_conc_summer_mpi_ws) # mean DJF
        sie_conc_summer_mpi_io = np.concatenate((sie_mpi_io[m,year-1,11:12],sie_mpi_io[m,year,0:2])) # concatenate DJF
        sie_summer_mpi_io[m,year] = np.nanmean(sie_conc_summer_mpi_io) # mean DJF
        sie_conc_summer_mpi_wpo = np.concatenate((sie_mpi_wpo[m,year-1,11:12],sie_mpi_wpo[m,year,0:2])) # concatenate DJF
        sie_summer_mpi_wpo[m,year] = np.nanmean(sie_conc_summer_mpi_wpo) # mean DJF
        sie_conc_summer_mpi_rs = np.concatenate((sie_mpi_rs[m,year-1,11:12],sie_mpi_rs[m,year,0:2])) # concatenate DJF
        sie_summer_mpi_rs[m,year] = np.nanmean(sie_conc_summer_mpi_rs) # mean DJF
    
# Summer Antarctic SIE - CESM
sie_summer_cesm_bas = np.zeros((n_members_cesm,nyears))
sie_summer_cesm_ws = np.zeros((n_members_cesm,nyears))
sie_summer_cesm_io = np.zeros((n_members_cesm,nyears))
sie_summer_cesm_wpo = np.zeros((n_members_cesm,nyears))
sie_summer_cesm_rs = np.zeros((n_members_cesm,nyears))
for m in np.arange(n_members_cesm):
    sie_summer_cesm_bas[m,0] = np.nanmean(sie_cesm_bas[m,0,0:2])
    sie_summer_cesm_ws[m,0] = np.nanmean(sie_cesm_ws[m,0,0:2])
    sie_summer_cesm_io[m,0] = np.nanmean(sie_cesm_io[m,0,0:2])
    sie_summer_cesm_wpo[m,0] = np.nanmean(sie_cesm_wpo[m,0,0:2])
    sie_summer_cesm_rs[m,0] = np.nanmean(sie_cesm_rs[m,0,0:2])
    for year in np.arange(1,nyears):
        sie_conc_summer_cesm_bas = np.concatenate((sie_cesm_bas[m,year-1,11:12],sie_cesm_bas[m,year,0:2])) # concatenate DJF
        sie_summer_cesm_bas[m,year] = np.nanmean(sie_conc_summer_cesm_bas) # mean DJF
        sie_conc_summer_cesm_ws = np.concatenate((sie_cesm_ws[m,year-1,11:12],sie_cesm_ws[m,year,0:2])) # concatenate DJF
        sie_summer_cesm_ws[m,year] = np.nanmean(sie_conc_summer_cesm_ws) # mean DJF
        sie_conc_summer_cesm_io = np.concatenate((sie_cesm_io[m,year-1,11:12],sie_cesm_io[m,year,0:2])) # concatenate DJF
        sie_summer_cesm_io[m,year] = np.nanmean(sie_conc_summer_cesm_io) # mean DJF
        sie_conc_summer_cesm_wpo = np.concatenate((sie_cesm_wpo[m,year-1,11:12],sie_cesm_wpo[m,year,0:2])) # concatenate DJF
        sie_summer_cesm_wpo[m,year] = np.nanmean(sie_conc_summer_cesm_wpo) # mean DJF
        sie_conc_summer_cesm_rs = np.concatenate((sie_cesm_rs[m,year-1,11:12],sie_cesm_rs[m,year,0:2])) # concatenate DJF
        sie_summer_cesm_rs[m,year] = np.nanmean(sie_conc_summer_cesm_rs) # mean DJF
    
# Summer Antarctic SIE - EC-Earth3
sie_summer_ecearth_bas = np.zeros((n_members_ecearth,nyears))
sie_summer_ecearth_ws = np.zeros((n_members_ecearth,nyears))
sie_summer_ecearth_io = np.zeros((n_members_ecearth,nyears))
sie_summer_ecearth_wpo = np.zeros((n_members_ecearth,nyears))
sie_summer_ecearth_rs = np.zeros((n_members_ecearth,nyears))
for m in np.arange(n_members_ecearth):
    sie_summer_ecearth_bas[m,0] = np.nanmean(sie_ecearth_bas[m,0,0:2])
    sie_summer_ecearth_ws[m,0] = np.nanmean(sie_ecearth_ws[m,0,0:2])
    sie_summer_ecearth_io[m,0] = np.nanmean(sie_ecearth_io[m,0,0:2])
    sie_summer_ecearth_wpo[m,0] = np.nanmean(sie_ecearth_wpo[m,0,0:2])
    sie_summer_ecearth_rs[m,0] = np.nanmean(sie_ecearth_rs[m,0,0:2])
    for year in np.arange(1,nyears):
        sie_conc_summer_ecearth_bas = np.concatenate((sie_ecearth_bas[m,year-1,11:12],sie_ecearth_bas[m,year,0:2])) # concatenate DJF
        sie_summer_ecearth_bas[m,year] = np.nanmean(sie_conc_summer_ecearth_bas) # mean DJF
        sie_conc_summer_ecearth_ws = np.concatenate((sie_ecearth_ws[m,year-1,11:12],sie_ecearth_ws[m,year,0:2])) # concatenate DJF
        sie_summer_ecearth_ws[m,year] = np.nanmean(sie_conc_summer_ecearth_ws) # mean DJF
        sie_conc_summer_ecearth_io = np.concatenate((sie_ecearth_io[m,year-1,11:12],sie_ecearth_io[m,year,0:2])) # concatenate DJF
        sie_summer_ecearth_io[m,year] = np.nanmean(sie_conc_summer_ecearth_io) # mean DJF
        sie_conc_summer_ecearth_wpo = np.concatenate((sie_ecearth_wpo[m,year-1,11:12],sie_ecearth_wpo[m,year,0:2])) # concatenate DJF
        sie_summer_ecearth_wpo[m,year] = np.nanmean(sie_conc_summer_ecearth_wpo) # mean DJF
        sie_conc_summer_ecearth_rs = np.concatenate((sie_ecearth_rs[m,year-1,11:12],sie_ecearth_rs[m,year,0:2])) # concatenate DJF
        sie_summer_ecearth_rs[m,year] = np.nanmean(sie_conc_summer_ecearth_rs) # mean DJF

# Summer Antarctic SIE - Obs.
sie_summer_obs = np.zeros(nyears_obs+1)
sie_summer_obs_bas = np.zeros(nyears_obs+1)
sie_summer_obs_ws = np.zeros(nyears_obs+1)
sie_summer_obs_io = np.zeros(nyears_obs+1)
sie_summer_obs_wpo = np.zeros(nyears_obs+1)
sie_summer_obs_rs = np.zeros(nyears_obs+1)
sie_summer_obs[0] = np.nanmean(sie_obs[0,0:2]) # initial year (JF 1979)
sie_summer_obs_bas[0] = np.nanmean(sie_obs_bas[0,0:2])
sie_summer_obs_ws[0] = np.nanmean(sie_obs_ws[0,0:2])
sie_summer_obs_io[0] = np.nanmean(sie_obs_io[0,0:2])
sie_summer_obs_wpo[0] = np.nanmean(sie_obs_wpo[0,0:2])
sie_summer_obs_rs[0] = np.nanmean(sie_obs_rs[0,0:2])
for year in np.arange(1,nyears_obs+1):
    if year == 45: # DJF 2023-2024
        sie_conc_summer_obs = np.concatenate((sie_obs[year-1,11:12],np.array([sie_obs_jan2024]),np.array([sie_obs_feb2024])))
        sie_conc_summer_obs_bas = np.concatenate((sie_obs_bas[year-1,11:12],np.array([sie_obs_bas_jan2024]),np.array([sie_obs_bas_feb2024])))
        sie_conc_summer_obs_ws = np.concatenate((sie_obs_ws[year-1,11:12],np.array([sie_obs_ws_jan2024]),np.array([sie_obs_ws_feb2024])))
        sie_conc_summer_obs_io = np.concatenate((sie_obs_io[year-1,11:12],np.array([sie_obs_io_jan2024]),np.array([sie_obs_io_feb2024])))
        sie_conc_summer_obs_wpo = np.concatenate((sie_obs_wpo[year-1,11:12],np.array([sie_obs_wpo_jan2024]),np.array([sie_obs_wpo_feb2024])))
        sie_conc_summer_obs_rs = np.concatenate((sie_obs_rs[year-1,11:12],np.array([sie_obs_rs_jan2024]),np.array([sie_obs_rs_feb2024])))
    else:
        sie_conc_summer_obs = np.concatenate((sie_obs[year-1,11:12],sie_obs[year,0:2]))
        sie_conc_summer_obs_bas = np.concatenate((sie_obs_bas[year-1,11:12],sie_obs_bas[year,0:2]))
        sie_conc_summer_obs_ws = np.concatenate((sie_obs_ws[year-1,11:12],sie_obs_ws[year,0:2]))
        sie_conc_summer_obs_io = np.concatenate((sie_obs_io[year-1,11:12],sie_obs_io[year,0:2]))
        sie_conc_summer_obs_wpo = np.concatenate((sie_obs_wpo[year-1,11:12],sie_obs_wpo[year,0:2]))
        sie_conc_summer_obs_rs = np.concatenate((sie_obs_rs[year-1,11:12],sie_obs_rs[year,0:2]))
    sie_summer_obs[year] = np.nanmean(sie_conc_summer_obs)
    sie_summer_obs_bas[year] = np.nanmean(sie_conc_summer_obs_bas)
    sie_summer_obs_ws[year] = np.nanmean(sie_conc_summer_obs_ws)
    sie_summer_obs_io[year] = np.nanmean(sie_conc_summer_obs_io)
    sie_summer_obs_wpo[year] = np.nanmean(sie_conc_summer_obs_wpo)
    sie_summer_obs_rs[year] = np.nanmean(sie_conc_summer_obs_rs)

# Save time series
if save_var == True:
    filename_ecearth = dir_output + 'SSIE_EC-Earth3_timeseries.npy'
    filename_cesm = dir_output + 'SSIE_CESM2_timeseries.npy'
    filename_mpi = dir_output + 'SSIE_MPI-ESM1-2-LR_timeseries.npy'
    filename_canesm = dir_output + 'SSIE_CanESM5_timeseries.npy'
    filename_access = dir_output + 'SSIE_ACCESS-ESM1-5_timeseries.npy'
    filename_obs = dir_output + 'SSIE_obs_timeseries.npy'
    np.save(filename_ecearth,[sie_summer_ecearth_bas,sie_summer_ecearth_ws,sie_summer_ecearth_io,sie_summer_ecearth_wpo,sie_summer_ecearth_rs])
    np.save(filename_cesm,[sie_summer_cesm_bas,sie_summer_cesm_ws,sie_summer_cesm_io,sie_summer_cesm_wpo,sie_summer_cesm_rs])
    np.save(filename_mpi,[sie_summer_mpi_bas,sie_summer_mpi_ws,sie_summer_mpi_io,sie_summer_mpi_wpo,sie_summer_mpi_rs])
    np.save(filename_canesm,[sie_summer_canesm_bas,sie_summer_canesm_ws,sie_summer_canesm_io,sie_summer_canesm_wpo,sie_summer_canesm_rs])
    np.save(filename_access,[sie_summer_access_bas,sie_summer_access_ws,sie_summer_access_io,sie_summer_access_wpo,sie_summer_access_rs])
    np.save(filename_obs,[sie_summer_obs_bas,sie_summer_obs_ws,sie_summer_obs_io,sie_summer_obs_wpo,sie_summer_obs_rs])  
    
# Plot options
xrange = np.arange(11,141,20)
name_xticks = ['1980','2000','2020','2040','2060','2080','2100']

# Time series of original variables
fig,ax = plt.subplots(3,2,figsize=(24,24))
fig.subplots_adjust(left=0.08,bottom=0.05,right=0.95,top=0.95,hspace=0.2,wspace=0.2)

# Antarctic SIE
ax[0,0].plot(np.arange(np.size(sie_summer_access,1)-1)+2,np.nanmean(sie_summer_access[:,1::],axis=0),'-',color='orange',linewidth=4,label='ACCESS-ESM1.5')
ax[0,0].plot(np.arange(np.size(sie_summer_canesm,1)-1)+2,np.nanmean(sie_summer_canesm[:,1::],axis=0),'-',color='green',linewidth=4,label='CanESM5')
ax[0,0].plot(np.arange(np.size(sie_summer_cesm,1)-1)+2,np.nanmean(sie_summer_cesm[:,1::],axis=0),'-',color='blue',linewidth=4,label='CESM2')
ax[0,0].plot(np.arange(np.size(sie_summer_ecearth,1)-1)+2,np.nanmean(sie_summer_ecearth[:,1::],axis=0),'-',color='red',linewidth=4,label='EC-Earth3')
ax[0,0].plot(np.arange(np.size(sie_summer_mpi,1)-1)+2,np.nanmean(sie_summer_mpi[:,1::],axis=0),'-',color='gray',linewidth=4,label='MPI-ESM1.2-LR')
ax[0,0].plot(np.arange(np.size(sie_summer_obs)-1)+11,sie_summer_obs[1::],'k.-',linewidth=2,label='Observations')
ax[0,0].fill_between(np.arange(np.size(sie_summer_ecearth,1)-1)+2,np.nanmin(sie_summer_ecearth[:,1::],axis=0),np.nanmax(sie_summer_ecearth[:,1::],axis=0),color='red',alpha=0.1)
ax[0,0].fill_between(np.arange(np.size(sie_summer_cesm,1)-1)+2,np.nanmin(sie_summer_cesm[:,1::],axis=0),np.nanmax(sie_summer_cesm[:,1::],axis=0),color='blue',alpha=0.1)
ax[0,0].fill_between(np.arange(np.size(sie_summer_mpi,1)-1)+2,np.nanmin(sie_summer_mpi[:,1::],axis=0),np.nanmax(sie_summer_mpi[:,1::],axis=0),color='gray',alpha=0.1)
ax[0,0].fill_between(np.arange(np.size(sie_summer_canesm,1)-1)+2,np.nanmin(sie_summer_canesm[:,1::],axis=0),np.nanmax(sie_summer_canesm[:,1::],axis=0),color='green',alpha=0.1)
ax[0,0].fill_between(np.arange(np.size(sie_summer_access,1)-1)+2,np.nanmin(sie_summer_access[:,1::],axis=0),np.nanmax(sie_summer_access[:,1::],axis=0),color='orange',alpha=0.1)
ax[0,0].set_ylabel('Total Antarctic SIE (10$^6$ km$^2$)',fontsize=26)
ax[0,0].set_xticks(xrange)
ax[0,0].set_xticklabels(name_xticks)
ax[0,0].tick_params(labelsize=20)
ax[0,0].grid(linestyle='--')
ax[0,0].axis([-1, 133, -0.5, 15])
ax[0,0].legend(loc='upper right',fontsize=22,shadow=True,frameon=False,ncol=2)
ax[0,0].set_title('a',loc='left',fontsize=30,fontweight='bold')

# Bellingshausen-Amundsen SIE
ax[0,1].plot(np.arange(np.size(sie_summer_access_bas,1)-1)+2,np.nanmean(sie_summer_access_bas[:,1::],axis=0),'-',color='orange',linewidth=4,label='ACCESS-ESM1.5')
ax[0,1].plot(np.arange(np.size(sie_summer_canesm_bas,1)-1)+2,np.nanmean(sie_summer_canesm_bas[:,1::],axis=0),'-',color='green',linewidth=4,label='CanESM5')
ax[0,1].plot(np.arange(np.size(sie_summer_cesm_bas,1)-1)+2,np.nanmean(sie_summer_cesm_bas[:,1::],axis=0),'-',color='blue',linewidth=4,label='CESM2')
ax[0,1].plot(np.arange(np.size(sie_summer_ecearth_bas,1)-1)+2,np.nanmean(sie_summer_ecearth_bas[:,1::],axis=0),'-',color='red',linewidth=4,label='EC-Earth3')
ax[0,1].plot(np.arange(np.size(sie_summer_mpi_bas,1)-1)+2,np.nanmean(sie_summer_mpi_bas[:,1::],axis=0),'-',color='gray',linewidth=4,label='MPI-ESM1.2-LR')
ax[0,1].plot(np.arange(np.size(sie_summer_obs_bas)-1)+11,sie_summer_obs_bas[1::],'k.-',linewidth=2,label='Observations')
ax[0,1].fill_between(np.arange(np.size(sie_summer_ecearth_bas,1)-1)+2,np.nanmin(sie_summer_ecearth_bas[:,1::],axis=0),np.nanmax(sie_summer_ecearth_bas[:,1::],axis=0),color='red',alpha=0.1)
ax[0,1].fill_between(np.arange(np.size(sie_summer_cesm_bas,1)-1)+2,np.nanmin(sie_summer_cesm_bas[:,1::],axis=0),np.nanmax(sie_summer_cesm_bas[:,1::],axis=0),color='blue',alpha=0.1)
ax[0,1].fill_between(np.arange(np.size(sie_summer_mpi_bas,1)-1)+2,np.nanmin(sie_summer_mpi_bas[:,1::],axis=0),np.nanmax(sie_summer_mpi_bas[:,1::],axis=0),color='gray',alpha=0.1)
ax[0,1].fill_between(np.arange(np.size(sie_summer_canesm_bas,1)-1)+2,np.nanmin(sie_summer_canesm_bas[:,1::],axis=0),np.nanmax(sie_summer_canesm_bas[:,1::],axis=0),color='green',alpha=0.1)
ax[0,1].fill_between(np.arange(np.size(sie_summer_access_bas,1)-1)+2,np.nanmin(sie_summer_access_bas[:,1::],axis=0),np.nanmax(sie_summer_access_bas[:,1::],axis=0),color='orange',alpha=0.1)
ax[0,1].set_ylabel('Bellingshausen-Amundsen SIE (10$^6$ km$^2$)',fontsize=26)
ax[0,1].set_xticks(xrange)
ax[0,1].set_xticklabels(name_xticks)
ax[0,1].tick_params(labelsize=20)
ax[0,1].legend(loc='upper right',fontsize=22,shadow=True,frameon=False,ncol=2)
ax[0,1].grid(linestyle='--')
ax[0,1].axis([-1, 133, -0.2, 4])
ax[0,1].set_title('b',loc='left',fontsize=30,fontweight='bold')

# Weddell SIE
ax[1,0].plot(np.arange(np.size(sie_summer_access_ws,1)-1)+2,np.nanmean(sie_summer_access_ws[:,1::],axis=0),'-',color='orange',linewidth=4,label='ACCESS-ESM1.5')
ax[1,0].plot(np.arange(np.size(sie_summer_canesm_ws,1)-1)+2,np.nanmean(sie_summer_canesm_ws[:,1::],axis=0),'-',color='green',linewidth=4,label='CanESM5')
ax[1,0].plot(np.arange(np.size(sie_summer_cesm_ws,1)-1)+2,np.nanmean(sie_summer_cesm_ws[:,1::],axis=0),'-',color='blue',linewidth=4,label='CESM2')
ax[1,0].plot(np.arange(np.size(sie_summer_ecearth_ws,1)-1)+2,np.nanmean(sie_summer_ecearth_ws[:,1::],axis=0),'-',color='red',linewidth=4,label='EC-Earth3')
ax[1,0].plot(np.arange(np.size(sie_summer_mpi_ws,1)-1)+2,np.nanmean(sie_summer_mpi_ws[:,1::],axis=0),'-',color='gray',linewidth=4,label='MPI-ESM1.2-LR')
ax[1,0].plot(np.arange(np.size(sie_summer_obs_ws)-1)+11,sie_summer_obs_ws[1::],'k.-',linewidth=2,label='Observations')
ax[1,0].fill_between(np.arange(np.size(sie_summer_ecearth_ws,1)-1)+2,np.nanmin(sie_summer_ecearth_ws[:,1::],axis=0),np.nanmax(sie_summer_ecearth_ws[:,1::],axis=0),color='red',alpha=0.1)
ax[1,0].fill_between(np.arange(np.size(sie_summer_cesm_ws,1)-1)+2,np.nanmin(sie_summer_cesm_ws[:,1::],axis=0),np.nanmax(sie_summer_cesm_ws[:,1::],axis=0),color='blue',alpha=0.1)
ax[1,0].fill_between(np.arange(np.size(sie_summer_mpi_ws,1)-1)+2,np.nanmin(sie_summer_mpi_ws[:,1::],axis=0),np.nanmax(sie_summer_mpi_ws[:,1::],axis=0),color='gray',alpha=0.1)
ax[1,0].fill_between(np.arange(np.size(sie_summer_canesm_ws,1)-1)+2,np.nanmin(sie_summer_canesm_ws[:,1::],axis=0),np.nanmax(sie_summer_canesm_ws[:,1::],axis=0),color='green',alpha=0.1)
ax[1,0].fill_between(np.arange(np.size(sie_summer_access_ws,1)-1)+2,np.nanmin(sie_summer_access_ws[:,1::],axis=0),np.nanmax(sie_summer_access_ws[:,1::],axis=0),color='orange',alpha=0.1)
ax[1,0].set_ylabel('Weddell SIE (10$^6$ km$^2$)',fontsize=26)
ax[1,0].set_xticks(xrange)
ax[1,0].set_xticklabels(name_xticks)
ax[1,0].tick_params(labelsize=20)
ax[1,0].grid(linestyle='--')
ax[1,0].axis([-1, 133, -0.2, 6])
ax[1,0].legend(loc='upper right',fontsize=22,shadow=True,frameon=False,ncol=2)
ax[1,0].set_title('c',loc='left',fontsize=30,fontweight='bold')

# Indian Ocean SIE
ax[1,1].plot(np.arange(np.size(sie_summer_access_io,1)-1)+2,np.nanmean(sie_summer_access_io[:,1::],axis=0),'-',color='orange',linewidth=4,label='ACCESS-ESM1.5')
ax[1,1].plot(np.arange(np.size(sie_summer_canesm_io,1)-1)+2,np.nanmean(sie_summer_canesm_io[:,1::],axis=0),'-',color='green',linewidth=4,label='CanESM5')
ax[1,1].plot(np.arange(np.size(sie_summer_cesm_io,1)-1)+2,np.nanmean(sie_summer_cesm_io[:,1::],axis=0),'-',color='blue',linewidth=4,label='CESM2')
ax[1,1].plot(np.arange(np.size(sie_summer_ecearth_io,1)-1)+2,np.nanmean(sie_summer_ecearth_io[:,1::],axis=0),'-',color='red',linewidth=4,label='EC-Earth3')
ax[1,1].plot(np.arange(np.size(sie_summer_mpi_io,1)-1)+2,np.nanmean(sie_summer_mpi_io[:,1::],axis=0),'-',color='gray',linewidth=4,label='MPI-ESM1.2-LR')
ax[1,1].plot(np.arange(np.size(sie_summer_obs_io)-1)+11,sie_summer_obs_io[1::],'k.-',linewidth=2,label='Observations')
ax[1,1].fill_between(np.arange(np.size(sie_summer_ecearth_io,1)-1)+2,np.nanmin(sie_summer_ecearth_io[:,1::],axis=0),np.nanmax(sie_summer_ecearth_io[:,1::],axis=0),color='red',alpha=0.1)
ax[1,1].fill_between(np.arange(np.size(sie_summer_cesm_io,1)-1)+2,np.nanmin(sie_summer_cesm_io[:,1::],axis=0),np.nanmax(sie_summer_cesm_io[:,1::],axis=0),color='blue',alpha=0.1)
ax[1,1].fill_between(np.arange(np.size(sie_summer_mpi_io,1)-1)+2,np.nanmin(sie_summer_mpi_io[:,1::],axis=0),np.nanmax(sie_summer_mpi_io[:,1::],axis=0),color='gray',alpha=0.1)
ax[1,1].fill_between(np.arange(np.size(sie_summer_canesm_io,1)-1)+2,np.nanmin(sie_summer_canesm_io[:,1::],axis=0),np.nanmax(sie_summer_canesm_io[:,1::],axis=0),color='green',alpha=0.1)
ax[1,1].fill_between(np.arange(np.size(sie_summer_access_io,1)-1)+2,np.nanmin(sie_summer_access_io[:,1::],axis=0),np.nanmax(sie_summer_access_io[:,1::],axis=0),color='orange',alpha=0.1)
ax[1,1].set_ylabel('Indian Ocean SIE (10$^6$ km$^2$)',fontsize=26)
ax[1,1].set_xticks(xrange)
ax[1,1].set_xticklabels(name_xticks)
ax[1,1].tick_params(labelsize=20)
ax[1,1].legend(loc='upper right',fontsize=22,shadow=True,frameon=False,ncol=2)
ax[1,1].grid(linestyle='--')
ax[1,1].axis([-1, 133, -0.2, 4])
ax[1,1].set_title('d',loc='left',fontsize=30,fontweight='bold')

# Western Pacific Ocean SIE
ax[2,0].plot(np.arange(np.size(sie_summer_access_wpo,1)-1)+2,np.nanmean(sie_summer_access_wpo[:,1::],axis=0),'-',color='orange',linewidth=4,label='ACCESS-ESM1.5')
ax[2,0].plot(np.arange(np.size(sie_summer_canesm_wpo,1)-1)+2,np.nanmean(sie_summer_canesm_wpo[:,1::],axis=0),'-',color='green',linewidth=4,label='CanESM5')
ax[2,0].plot(np.arange(np.size(sie_summer_cesm_wpo,1)-1)+2,np.nanmean(sie_summer_cesm_wpo[:,1::],axis=0),'-',color='blue',linewidth=4,label='CESM2')
ax[2,0].plot(np.arange(np.size(sie_summer_ecearth_wpo,1)-1)+2,np.nanmean(sie_summer_ecearth_wpo[:,1::],axis=0),'-',color='red',linewidth=4,label='EC-Earth3')
ax[2,0].plot(np.arange(np.size(sie_summer_mpi_wpo,1)-1)+2,np.nanmean(sie_summer_mpi_wpo[:,1::],axis=0),'-',color='gray',linewidth=4,label='MPI-ESM1.2-LR')
ax[2,0].plot(np.arange(np.size(sie_summer_obs_wpo)-1)+11,sie_summer_obs_wpo[1::],'k.-',linewidth=2,label='Observations')
ax[2,0].fill_between(np.arange(np.size(sie_summer_ecearth_wpo,1)-1)+2,np.nanmin(sie_summer_ecearth_wpo[:,1::],axis=0),np.nanmax(sie_summer_ecearth_wpo[:,1::],axis=0),color='red',alpha=0.1)
ax[2,0].fill_between(np.arange(np.size(sie_summer_cesm_wpo,1)-1)+2,np.nanmin(sie_summer_cesm_wpo[:,1::],axis=0),np.nanmax(sie_summer_cesm_wpo[:,1::],axis=0),color='blue',alpha=0.1)
ax[2,0].fill_between(np.arange(np.size(sie_summer_mpi_wpo,1)-1)+2,np.nanmin(sie_summer_mpi_wpo[:,1::],axis=0),np.nanmax(sie_summer_mpi_wpo[:,1::],axis=0),color='gray',alpha=0.1)
ax[2,0].fill_between(np.arange(np.size(sie_summer_canesm_wpo,1)-1)+2,np.nanmin(sie_summer_canesm_wpo[:,1::],axis=0),np.nanmax(sie_summer_canesm_wpo[:,1::],axis=0),color='green',alpha=0.1)
ax[2,0].fill_between(np.arange(np.size(sie_summer_access_wpo,1)-1)+2,np.nanmin(sie_summer_access_wpo[:,1::],axis=0),np.nanmax(sie_summer_access_wpo[:,1::],axis=0),color='orange',alpha=0.1)
ax[2,0].set_ylabel('Western Pacific Ocean SIE (10$^6$ km$^2$)',fontsize=26)
ax[2,0].set_xticks(xrange)
ax[2,0].set_xticklabels(name_xticks)
ax[2,0].tick_params(labelsize=20)
ax[2,0].legend(loc='upper right',fontsize=22,shadow=True,frameon=False,ncol=2)
ax[2,0].grid(linestyle='--')
ax[2,0].axis([-1, 133, -0.2, 4])
ax[2,0].set_title('e',loc='left',fontsize=30,fontweight='bold')

# Ross SIE
ax[2,1].plot(np.arange(np.size(sie_summer_access_rs,1)-1)+2,np.nanmean(sie_summer_access_rs[:,1::],axis=0),'-',color='orange',linewidth=4,label='ACCESS-ESM1.5')
ax[2,1].plot(np.arange(np.size(sie_summer_canesm_rs,1)-1)+2,np.nanmean(sie_summer_canesm_rs[:,1::],axis=0),'-',color='green',linewidth=4,label='CanESM5')
ax[2,1].plot(np.arange(np.size(sie_summer_cesm_rs,1)-1)+2,np.nanmean(sie_summer_cesm_rs[:,1::],axis=0),'-',color='blue',linewidth=4,label='CESM2')
ax[2,1].plot(np.arange(np.size(sie_summer_ecearth_bas,1)-1)+2,np.nanmean(sie_summer_ecearth_rs[:,1::],axis=0),'-',color='red',linewidth=4,label='EC-Earth3')
ax[2,1].plot(np.arange(np.size(sie_summer_mpi_rs,1)-1)+2,np.nanmean(sie_summer_mpi_rs[:,1::],axis=0),'-',color='gray',linewidth=4,label='MPI-ESM1.2-LR')
ax[2,1].plot(np.arange(np.size(sie_summer_obs_rs)-1)+11,sie_summer_obs_rs[1::],'k.-',linewidth=2,label='Observations')
ax[2,1].fill_between(np.arange(np.size(sie_summer_ecearth_rs,1)-1)+2,np.nanmin(sie_summer_ecearth_rs[:,1::],axis=0),np.nanmax(sie_summer_ecearth_rs[:,1::],axis=0),color='red',alpha=0.1)
ax[2,1].fill_between(np.arange(np.size(sie_summer_cesm_rs,1)-1)+2,np.nanmin(sie_summer_cesm_rs[:,1::],axis=0),np.nanmax(sie_summer_cesm_rs[:,1::],axis=0),color='blue',alpha=0.1)
ax[2,1].fill_between(np.arange(np.size(sie_summer_mpi_rs,1)-1)+2,np.nanmin(sie_summer_mpi_rs[:,1::],axis=0),np.nanmax(sie_summer_mpi_rs[:,1::],axis=0),color='gray',alpha=0.1)
ax[2,1].fill_between(np.arange(np.size(sie_summer_canesm_rs,1)-1)+2,np.nanmin(sie_summer_canesm_rs[:,1::],axis=0),np.nanmax(sie_summer_canesm_rs[:,1::],axis=0),color='green',alpha=0.1)
ax[2,1].fill_between(np.arange(np.size(sie_summer_access_rs,1)-1)+2,np.nanmin(sie_summer_access_rs[:,1::],axis=0),np.nanmax(sie_summer_access_rs[:,1::],axis=0),color='orange',alpha=0.1)
ax[2,1].set_ylabel('Ross SIE (10$^6$ km$^2$)',fontsize=26)
ax[2,1].set_xticks(xrange)
ax[2,1].set_xticklabels(name_xticks)
ax[2,1].tick_params(labelsize=20)
ax[2,1].legend(loc='upper right',fontsize=22,shadow=True,frameon=False,ncol=2)
ax[2,1].grid(linestyle='--')
ax[2,1].axis([-1, 133, -0.2, 4])
ax[2,1].set_title('f',loc='left',fontsize=30,fontweight='bold')

# Save Fig.
if save_fig == True:
    fig.savefig(dir_fig + 'fig_a4.pdf')