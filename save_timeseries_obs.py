#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Save time series of original variables - Observations

Variables:
Summer sea-ice extent (DJF)
Previous winter/spring sea-ice extent (JAS or OND)
Previous Antarctic mean surface air temperature (<60S; JAS or OND)
Previous Antarctic mean SST (<60S; JAS or OND)
Previous Southern Annular Mode (SAM; JAS or OND)
Previous Amundsen Sea Low (ASL; JAS or OND)
Previous Niño3.4 (JAS or OND)
Previous DMI (JAS or OND)

Last updated: 15/01/2025

@author: David Docquier
"""

# Import libraries
import numpy as np
from netCDF4 import Dataset
np.bool = np.bool_

# Options
season = 'OND' # JAS (previous winter), OND (previous spring)
if season == 'JAS':
    index_start = 6
    index_end = 9
elif season == 'OND':
    index_start = 9
    index_end = 12
nmy = 12 # number of months in a year
save_var = False

# Interpolate NaN values
def interpolate_nan(array_like):
    array = array_like.copy()
    nans = np.isnan(array)
    
    def get_x(a):
        return a.nonzero()[0]

    array[nans] = np.interp(get_x(nans), get_x(~nans), array[~nans])

    return array

# Working directories
dir_osisaf = '/home/dadocq/Documents/Observations/OSISAF/' # Observed sea-ice extent
dir_hadcrut5 = '/home/dadocq/Documents/Observations/HadCRUT5/' # Observed surface air temperature
dir_ostia = '/home/dadocq/Documents/Observations/OSTIA/' # Observed SST
dir_marshall = '/home/dadocq/Documents/Observations/SAM_Marshall/' # SAM index (Marshall index)
dir_asl = '/home/dadocq/Documents/Observations/ASL/' # ASL index
dir_indices = '/home/dadocq/Documents/Observations/NOAA_Indices/' # ENSO and DMI
dir_output = '/home/dadocq/Documents/Papers/My_Papers/RESIST_Antarctic/output/seasons/'

# Load monthly SIE from OSI SAF (OSI-420) 1979-2024 (until Nov 2024)
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

# Summer and previous (JAS/OND) Antarctic SIE
sie_obs_previous = np.nanmean(sie_obs[:,index_start:index_end],axis=1)
sie_obs_summer = np.zeros(nyears_obs+1)
sie_obs_summer[0] = np.nanmean(sie_obs[0,0:2]) # initial year (JF 1979)
for year in np.arange(1,nyears_obs+1):
    if year == 45: # DJF 2023-2024
        sie_conc_obs_summer = np.concatenate((sie_obs[year-1,11:12],np.array([sie_obs_jan2024]),np.array([sie_obs_feb2024]))) # concatenate DJF
    else:
        sie_conc_obs_summer = np.concatenate((sie_obs[year-1,11:12],sie_obs[year,0:2])) # concatenate DJF
    sie_obs_summer[year] = np.nanmean(sie_conc_obs_summer) # mean DJF

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
nm_hadcrut5 = np.size(tas_obs_ano,0)

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

# Compute Antarctic mean surface temperature anomalies from HadCRUT5 1970-2023
tas_mean_obs_ano = np.zeros(nm_hadcrut5)
for mon in np.arange(nm_hadcrut5):
    tas_mean_obs_ano[mon] = np.average(tas_obs_ano[mon,:,:] * (lat_hadcrut5 <= -60.) * np.isfinite(sst_hadsst4),weights=grid_area_hadcrut5 * (lat_hadcrut5 <= -60.) * np.isfinite(sst_hadsst4))
   
# Compute Antarctic mean absolute surface temperature from HadCRUT5 (average over 1961-1990)
tas_mean_obs_cycle = np.zeros(nmy)
for mon in np.arange(nmy):
    tas_mean_obs_cycle[mon] = np.average(tas_obs_cycle[mon,:,:] * (lat_hadcrut5 <= -60.) * np.isfinite(sst_hadsst4),weights=grid_area_hadcrut5 * (lat_hadcrut5 <= -60.) * np.isfinite(sst_hadsst4))
 
# Compute seasonal mean (JAS/OND) surface temperature anomalies from HadCRUT5 1970-2023
nyears_hadcrut5 = int(nm_hadcrut5/nmy)
tas_mean_obs_ano_mon = np.zeros((nyears_hadcrut5,nmy))
for mon in np.arange(nmy):
    tas_mean_obs_ano_mon[:,mon] = tas_mean_obs_ano[mon::12]
tas_mean_obs_ano_seasonalmean = np.nanmean(tas_mean_obs_ano_mon[:,index_start:index_end],axis=1)

# Compute seasonal mean (JAS/OND) absolute surface temperature from HadCRUT5 (average over 1961-1990)
tas_mean_obs_cycle_seasonalmean = np.nanmean(tas_mean_obs_cycle[index_start:index_end])

# Compute seasonal mean (JAS/OND) absolute surface temperature from HadCRUT5 1970-2023
tas_obs = tas_mean_obs_ano_seasonalmean + tas_mean_obs_cycle_seasonalmean

# Load SST OSTIA 1982-2023 and convert into degC
filename = dir_ostia + 'OSTIA_SST_SecZ83.lat-80to-60.1982to2023_1m.nc'
fh = Dataset(filename, mode='r')
sst_obs_init = fh.variables['sst'][:,0]
fh.close()
sst_obs_init = sst_obs_init - 273.15
nyears_obs2 = int(2023-1982+1)
sst_obs_init2 = np.zeros((nyears_obs2,nmy))
for mon in np.arange(nmy):
    sst_obs_init2[:,mon] = sst_obs_init[mon::12]
    
# Compute seasonal mean SST (JAS or OND)
sst_obs = np.nanmean(sst_obs_init2[:,index_start:index_end],axis=1)

# Load SAM Marshall Index (1970-2023) and compute seasonal mean (JAS/OND)
# Marshall Index: https://legacy.bas.ac.uk/met/gjma/sam.html
filename = dir_marshall + 'newsam.1957.2023.txt'
if season == 'JAS':
    sam_obs_init = np.loadtxt(filename,skiprows=15,usecols=(7,8,9))
else:
    sam_obs_init = np.loadtxt(filename,skiprows=15,usecols=(10,11,12))
nyears_noaa = np.size(sam_obs_init,0)
sam_obs = np.nanmean(sam_obs_init,axis=1)
nyears_sam = np.size(sam_obs)

# Load ASL Index (1970-2023)
# ASL Index: https://scotthosking.com/asl_index
filename = dir_asl + 'asli_era5_v3-latest.txt'
asl_obs_init = np.loadtxt(filename,delimiter=',',skiprows=162,max_rows=648,usecols=(3))
nyears_asl = int(np.size(asl_obs_init) / nmy)
asl_obs_mon = np.zeros((nyears_asl,nmy))
for mon in np.arange(nmy):
    asl_obs_mon[:,mon] = asl_obs_init[mon::12]

# Compute seasonal mean (JAS/OND) ASL index
asl_obs = np.zeros(nyears_asl)
for year in np.arange(nyears_asl):
    asl_obs[year] = np.nanmean(asl_obs_mon[year,index_start:index_end])

# Load observed Niño3.4 from ERSSTv5 (https://psl.noaa.gov/data/climateindices/list/) 1970-2023
filename = dir_indices + 'nino34.anom.data'
if season == 'JAS':
    nino34_obs_init = np.loadtxt(filename,skiprows=23,max_rows=54,usecols=(7,8,9))
else:
    nino34_obs_init = np.loadtxt(filename,skiprows=23,max_rows=54,usecols=(10,11,12))
nino34_obs = np.nanmean(nino34_obs_init,axis=1)

# Load observed DMI from ERSSTv5 (https://www.cpc.ncep.noaa.gov/products/international/ocean_monitoring/IODMI/DMI_month.html) 1970-2023
filename = dir_indices + 'mnth.ersstv5.clim19912020.dmi_current.txt'
dmi_obs_init = np.loadtxt(filename,skiprows=249,max_rows=648)[:,4]
dmi_obs_init2 = np.reshape(dmi_obs_init,(54,12))
dmi_obs_init3 = dmi_obs_init2[:,index_start:index_end]
dmi_obs = np.nanmean(dmi_obs_init3,axis=1)

# Save time series
filename = dir_output + 'Obs_Antarctic_timeseries_' + season + '.npy'
if save_var == True:
    np.save(filename,np.array([sie_obs_summer,sie_obs_previous,tas_obs,sst_obs,sam_obs,asl_obs,nino34_obs,dmi_obs],dtype=object))