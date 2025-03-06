#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Figure 2: Plot time series of original variables - All models and Observations

Model large ensembles: EC-Earth3 (SMHI-LENS), CESM2-LE, MPI-ESM1-2-LR, CanESM5, ACCESS-ESM1-5
Time series saved via save_timeseries.py

Observations
Time series saved via save_timeseries_obs.py

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
import matplotlib.pyplot as plt

# Options
season = 'OND' # JAS (previous winter), OND (previous spring; default)
if season == 'JAS':
    string_season = 'Winter'
elif season == 'OND':
    string_season = 'Spring'
save_fig = True

# Working directories
dir_input = '/home/dadocq/Documents/Papers/My_Papers/RESIST_Antarctic/output/seasons/'
dir_fig = '/home/dadocq/Documents/Papers/My_Papers/RESIST_Antarctic/LaTeX/'

# Load EC-Earth3 variables
filename = dir_input + 'EC-Earth3_Antarctic_timeseries_' + season + '.npy'
sie_summer_ecearth,sie_previous_ecearth,tas_ecearth,sst_ecearth,sam_ecearth,asl_ecearth,nino34_ecearth,dmi_ecearth = np.load(filename,allow_pickle=True)

# Load CESM2 variables
filename = dir_input + 'CESM2_Antarctic_timeseries_' + season + '.npy'
sie_summer_cesm,sie_previous_cesm,tas_cesm,sst_cesm,sam_cesm,asl_cesm,nino34_cesm,dmi_cesm = np.load(filename,allow_pickle=True)
#
# Load MPI-ESM1-2-LR variables
filename = dir_input + 'MPI-ESM1-2-LR_Antarctic_timeseries_' + season + '.npy'
sie_summer_mpi,sie_previous_mpi,tas_mpi,sst_mpi,sam_mpi,asl_mpi,nino34_mpi,dmi_mpi = np.load(filename,allow_pickle=True)

# Load CanESM5 variables
filename = dir_input + 'CanESM5_Antarctic_timeseries_' + season + '.npy'
sie_summer_canesm,sie_previous_canesm,tas_canesm,sst_canesm,sam_canesm,asl_canesm,nino34_canesm,dmi_canesm = np.load(filename,allow_pickle=True)

# Load ACCESS-ESM1-5 variables
filename = dir_input + 'ACCESS-ESM1-5_Antarctic_timeseries_' + season + '.npy'
sie_summer_access,sie_previous_access,tas_access,sst_access,sam_access,asl_access,nino34_access,dmi_access = np.load(filename,allow_pickle=True)

# Load observations
filename = dir_input + 'Obs_Antarctic_timeseries_' + season + '.npy'
sie_summer_obs,sie_previous_obs,tas_obs,sst_obs,sam_obs,asl_obs,nino34_obs,dmi_obs = np.load(filename,allow_pickle=True)

# Plot options
xrange = np.arange(11,141,20)
name_xticks = ['1980','2000','2020','2040','2060','2080','2100']

# Time series of original variables
fig,ax = plt.subplots(3,3,figsize=(30,24))
fig.subplots_adjust(left=0.08,bottom=0.05,right=0.95,top=0.95,hspace=0.2,wspace=0.2)

# Summer sea-ice extent
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
ax[0,0].set_ylabel('Summer sea-ice extent (10$^6$ km$^2$)',fontsize=26)
ax[0,0].set_xticks(xrange)
ax[0,0].set_xticklabels(name_xticks)
ax[0,0].tick_params(labelsize=20)
ax[0,0].legend(loc='upper right',fontsize=22,shadow=True,frameon=False,ncol=2)
ax[0,0].grid(linestyle='--')
ax[0,0].axis([-1, 133, -0.5, 15])
ax[0,0].set_title('a',loc='left',fontsize=30,fontweight='bold')

# Previous sea-ice extent
ax[0,1].plot(np.arange(np.size(sie_previous_ecearth,1))+1,np.nanmean(sie_previous_ecearth,axis=0),'-',color='red',linewidth=4,label='EC-Earth3')
ax[0,1].plot(np.arange(np.size(sie_previous_cesm,1))+1,np.nanmean(sie_previous_cesm,axis=0),'-',color='blue',linewidth=4,label='CESM2')
ax[0,1].plot(np.arange(np.size(sie_previous_mpi,1))+1,np.nanmean(sie_previous_mpi,axis=0),'-',color='gray',linewidth=4,label='MPI-ESM1.2-LR')
ax[0,1].plot(np.arange(np.size(sie_previous_canesm,1))+1,np.nanmean(sie_previous_canesm,axis=0),'-',color='green',linewidth=4,label='CanESM5')
ax[0,1].plot(np.arange(np.size(sie_previous_access,1))+1,np.nanmean(sie_previous_access,axis=0),'-',color='orange',linewidth=4,label='ACCESS-ESM1.5')
ax[0,1].plot(np.arange(np.size(sie_previous_obs))+10,sie_previous_obs,'k.-',linewidth=2,label='Observations')
ax[0,1].fill_between(np.arange(np.size(sie_previous_ecearth,1))+1,np.nanmin(sie_previous_ecearth,axis=0),np.nanmax(sie_previous_ecearth,axis=0),color='red',alpha=0.1)
ax[0,1].fill_between(np.arange(np.size(sie_previous_cesm,1))+1,np.nanmin(sie_previous_cesm,axis=0),np.nanmax(sie_previous_cesm,axis=0),color='blue',alpha=0.1)
ax[0,1].fill_between(np.arange(np.size(sie_previous_mpi,1))+1,np.nanmin(sie_previous_mpi,axis=0),np.nanmax(sie_previous_mpi,axis=0),color='gray',alpha=0.1)
ax[0,1].fill_between(np.arange(np.size(sie_previous_canesm,1))+1,np.nanmin(sie_previous_canesm,axis=0),np.nanmax(sie_previous_canesm,axis=0),color='green',alpha=0.1)
ax[0,1].fill_between(np.arange(np.size(sie_previous_access,1))+1,np.nanmin(sie_previous_access,axis=0),np.nanmax(sie_previous_access,axis=0),color='orange',alpha=0.1)
ax[0,1].set_ylabel(string_season + ' sea-ice extent (10$^6$ km$^2$)',fontsize=26)
ax[0,1].set_xticks(xrange)
ax[0,1].set_xticklabels(name_xticks)
ax[0,1].tick_params(labelsize=20)
ax[0,1].legend(loc='lower left',fontsize=22,shadow=True,frameon=False,ncol=2)
ax[0,1].grid(linestyle='--')
ax[0,1].axis([-1, 133, -1, 24])
ax[0,1].set_title('b',loc='left',fontsize=30,fontweight='bold')

# Previous surface air temperature
ax[0,2].plot(np.arange(np.size(tas_access,1))+1,np.nanmean(tas_access,axis=0),'-',color='orange',linewidth=4,label='ACCESS-ESM1.5')
ax[0,2].plot(np.arange(np.size(tas_canesm,1))+1,np.nanmean(tas_canesm,axis=0),'-',color='green',linewidth=4,label='CanESM5')
ax[0,2].plot(np.arange(np.size(tas_cesm,1))+1,np.nanmean(tas_cesm,axis=0),'-',color='blue',linewidth=4,label='CESM2')
ax[0,2].plot(np.arange(np.size(tas_ecearth,1))+1,np.nanmean(tas_ecearth,axis=0),'-',color='red',linewidth=4,label='EC-Earth3')
ax[0,2].plot(np.arange(np.size(tas_mpi,1))+1,np.nanmean(tas_mpi,axis=0),'-',color='gray',linewidth=4,label='MPI-ESM1.2-LR')
ax[0,2].plot(np.arange(np.size(tas_obs))+1,tas_obs,'k.-',linewidth=2,label='Observations')
ax[0,2].fill_between(np.arange(np.size(tas_ecearth,1))+1,np.nanmin(tas_ecearth,axis=0),np.nanmax(tas_ecearth,axis=0),color='red',alpha=0.1)
ax[0,2].fill_between(np.arange(np.size(tas_cesm,1))+1,np.nanmin(tas_cesm,axis=0),np.nanmax(tas_cesm,axis=0),color='blue',alpha=0.1)
ax[0,2].fill_between(np.arange(np.size(tas_mpi,1))+1,np.nanmin(tas_mpi,axis=0),np.nanmax(tas_mpi,axis=0),color='gray',alpha=0.1)
ax[0,2].fill_between(np.arange(np.size(tas_canesm,1))+1,np.nanmin(tas_canesm,axis=0),np.nanmax(tas_canesm,axis=0),color='green',alpha=0.1)
ax[0,2].fill_between(np.arange(np.size(tas_access,1))+1,np.nanmin(tas_access,axis=0),np.nanmax(tas_access,axis=0),color='orange',alpha=0.1)
ax[0,2].set_ylabel(string_season + ' surface air temperature ($^\circ$C)',fontsize=26)
ax[0,2].set_xticks(xrange)
ax[0,2].set_xticklabels(name_xticks)
ax[0,2].tick_params(labelsize=20)
ax[0,2].legend(loc='upper left',fontsize=22,shadow=True,frameon=False,ncol=2)
ax[0,2].grid(linestyle='--')
ax[0,2].axis([-1, 133, -10, 4])
ax[0,2].set_title('c',loc='left',fontsize=30,fontweight='bold')

# Previous SST
ax[1,0].plot(np.arange(np.size(sst_access,1))+1,np.nanmean(sst_access,axis=0),'-',color='orange',linewidth=4,label='ACCESS-ESM1.5')
ax[1,0].plot(np.arange(np.size(sst_canesm,1))+1,np.nanmean(sst_canesm,axis=0),'-',color='green',linewidth=4,label='CanESM5')
ax[1,0].plot(np.arange(np.size(sst_cesm,1))+1,np.nanmean(sst_cesm,axis=0),'-',color='blue',linewidth=4,label='CESM2')
ax[1,0].plot(np.arange(np.size(sst_ecearth,1))+1,np.nanmean(sst_ecearth,axis=0),'-',color='red',linewidth=4,label='EC-Earth3')
ax[1,0].plot(np.arange(np.size(sst_mpi,1))+1,np.nanmean(sst_mpi,axis=0),'-',color='gray',linewidth=4,label='MPI-ESM1.2-LR')
ax[1,0].plot(np.arange(np.size(sst_obs))+13,sst_obs,'k.-',linewidth=2,label='Observations')
ax[1,0].fill_between(np.arange(np.size(sst_ecearth,1))+1,np.nanmin(sst_ecearth,axis=0),np.nanmax(sst_ecearth,axis=0),color='red',alpha=0.1)
ax[1,0].fill_between(np.arange(np.size(sst_cesm,1))+1,np.nanmin(sst_cesm,axis=0),np.nanmax(sst_cesm,axis=0),color='blue',alpha=0.1)
ax[1,0].fill_between(np.arange(np.size(sst_mpi,1))+1,np.nanmin(sst_mpi,axis=0),np.nanmax(sst_mpi,axis=0),color='gray',alpha=0.1)
ax[1,0].fill_between(np.arange(np.size(sst_canesm,1))+1,np.nanmin(sst_canesm,axis=0),np.nanmax(sst_canesm,axis=0),color='green',alpha=0.1)
ax[1,0].fill_between(np.arange(np.size(sst_access,1))+1,np.nanmin(sst_access,axis=0),np.nanmax(sst_access,axis=0),color='orange',alpha=0.1)
ax[1,0].set_ylabel(string_season + ' SST ($^\circ$C)',fontsize=26)
ax[1,0].set_xticks(xrange)
ax[1,0].set_xticklabels(name_xticks)
ax[1,0].tick_params(labelsize=20)
ax[1,0].grid(linestyle='--')
ax[1,0].axis([-1, 133, -2, 4])
ax[1,0].legend(loc='upper left',fontsize=22,shadow=True,frameon=False,ncol=2)
ax[1,0].set_title('d',loc='left',fontsize=30,fontweight='bold')

# Previous SAM
ax[1,1].plot(np.arange(np.size(sam_access,1))+1,np.nanmean(sam_access,axis=0),'-',color='orange',linewidth=4,label='ACCESS-ESM1.5')
ax[1,1].plot(np.arange(np.size(sam_canesm,1))+1,np.nanmean(sam_canesm,axis=0),'-',color='green',linewidth=4,label='CanESM5')
ax[1,1].plot(np.arange(np.size(sam_cesm,1))+1,np.nanmean(sam_cesm,axis=0),'-',color='blue',linewidth=4,label='CESM2')
ax[1,1].plot(np.arange(np.size(sam_ecearth,1))+1,np.nanmean(sam_ecearth,axis=0),'-',color='red',linewidth=4,label='EC-Earth3')
ax[1,1].plot(np.arange(np.size(sam_mpi,1))+1,np.nanmean(sam_mpi,axis=0),'-',color='gray',linewidth=4,label='MPI-ESM1.2-LR')
ax[1,1].plot(np.arange(np.size(sam_obs))+1,sam_obs,'k.-',linewidth=2,label='Observations')
ax[1,1].fill_between(np.arange(np.size(sam_ecearth,1))+1,np.nanmean(sam_ecearth,axis=0)-np.nanstd(sam_ecearth,axis=0),np.nanmean(sam_ecearth,axis=0)+np.nanstd(sam_ecearth,axis=0),color='r',alpha=0.1)
ax[1,1].fill_between(np.arange(np.size(sam_ecearth,1))+1,np.nanmin(sam_ecearth,axis=0),np.nanmax(sam_ecearth,axis=0),color='red',alpha=0.1)
ax[1,1].fill_between(np.arange(np.size(sam_cesm,1))+1,np.nanmin(sam_cesm,axis=0),np.nanmax(sam_cesm,axis=0),color='blue',alpha=0.1)
ax[1,1].fill_between(np.arange(np.size(sam_mpi,1))+1,np.nanmin(sam_mpi,axis=0),np.nanmax(sam_mpi,axis=0),color='gray',alpha=0.1)
ax[1,1].fill_between(np.arange(np.size(sam_canesm,1))+1,np.nanmin(sam_canesm,axis=0),np.nanmax(sam_canesm,axis=0),color='green',alpha=0.1)
ax[1,1].fill_between(np.arange(np.size(sam_access,1))+1,np.nanmin(sam_access,axis=0),np.nanmax(sam_access,axis=0),color='orange',alpha=0.1)
ax[1,1].set_ylabel(string_season + ' SAM',fontsize=26)
ax[1,1].set_xticks(xrange)
ax[1,1].set_xticklabels(name_xticks)
ax[1,1].tick_params(labelsize=20)
ax[1,1].grid(linestyle='--')
ax[1,1].axis([-1, 133, -5, 5])
ax[1,1].legend(loc='upper left',fontsize=22,shadow=True,frameon=False,ncol=2)
ax[1,1].set_title('e',loc='left',fontsize=30,fontweight='bold')

# Previous ASL
ax[1,2].plot(np.arange(np.size(asl_access,1)-1)+2,np.nanmean(asl_access[:,1::],axis=0),'-',color='orange',linewidth=4,label='ACCESS-ESM1.5')
ax[1,2].plot(np.arange(np.size(asl_canesm,1)-1)+2,np.nanmean(asl_canesm[:,1::],axis=0),'-',color='green',linewidth=4,label='CanESM5')
ax[1,2].plot(np.arange(np.size(asl_cesm,1)-1)+2,np.nanmean(asl_cesm[:,1::],axis=0),'-',color='blue',linewidth=4,label='CESM2')
ax[1,2].plot(np.arange(np.size(asl_ecearth,1)-1)+2,np.nanmean(asl_ecearth[:,1::],axis=0),'-',color='red',linewidth=4,label='EC-Earth3')
ax[1,2].plot(np.arange(np.size(asl_mpi,1)-1)+2,np.nanmean(asl_mpi[:,1::],axis=0),'-',color='gray',linewidth=4,label='MPI-ESM1.2-LR')
ax[1,2].plot(np.arange(np.size(asl_obs))+2,asl_obs,'k.-',linewidth=2,label='Reanalysis')
ax[1,2].fill_between(np.arange(np.size(asl_ecearth,1)-1)+2,np.nanmin(asl_ecearth[:,1::],axis=0),np.nanmax(asl_ecearth[:,1::],axis=0),color='red',alpha=0.1)
ax[1,2].fill_between(np.arange(np.size(asl_cesm,1)-1)+2,np.nanmin(asl_cesm[:,1::],axis=0),np.nanmax(asl_cesm[:,1::],axis=0),color='blue',alpha=0.1)
ax[1,2].fill_between(np.arange(np.size(asl_mpi,1)-1)+2,np.nanmin(asl_mpi[:,1::],axis=0),np.nanmax(asl_mpi[:,1::],axis=0),color='gray',alpha=0.1)
ax[1,2].fill_between(np.arange(np.size(asl_canesm,1)-1)+2,np.nanmin(asl_canesm[:,1::],axis=0),np.nanmax(asl_canesm[:,1::],axis=0),color='green',alpha=0.1)
ax[1,2].fill_between(np.arange(np.size(asl_access,1)-1)+2,np.nanmin(asl_access[:,1::],axis=0),np.nanmax(asl_access[:,1::],axis=0),color='orange',alpha=0.1)
ax[1,2].set_ylabel(string_season + ' ASL central pressure (hPa)',fontsize=26)
ax[1,2].set_xticks(xrange)
ax[1,2].set_xticklabels(name_xticks)
ax[1,2].tick_params(labelsize=20)
ax[1,2].legend(loc='upper right',fontsize=22,shadow=True,frameon=False,ncol=2)
ax[1,2].grid(linestyle='--')
ax[1,2].axis([-1, 133, 960, 1000])
ax[1,2].set_title('f',loc='left',fontsize=30,fontweight='bold')

# Previous Niño3.4
ax[2,0].plot(np.arange(np.size(nino34_access,1))+1,np.nanmean(nino34_access,axis=0),'-',color='orange',linewidth=4,label='ACCESS-ESM1.5')
ax[2,0].plot(np.arange(np.size(nino34_canesm,1))+1,np.nanmean(nino34_canesm,axis=0),'-',color='green',linewidth=4,label='CanESM5')
ax[2,0].plot(np.arange(np.size(nino34_cesm,1))+1,np.nanmean(nino34_cesm,axis=0),'-',color='blue',linewidth=4,label='CESM2')
ax[2,0].plot(np.arange(np.size(nino34_ecearth,1))+1,np.nanmean(nino34_ecearth,axis=0),'-',color='red',linewidth=4,label='EC-Earth3')
ax[2,0].plot(np.arange(np.size(nino34_mpi,1))+1,np.nanmean(nino34_mpi,axis=0),'-',color='gray',linewidth=4,label='MPI-ESM1.2-LR')
ax[2,0].plot(np.arange(np.size(nino34_obs))+1,nino34_obs,'k.-',linewidth=2,label='Observations')
ax[2,0].fill_between(np.arange(np.size(nino34_ecearth,1))+1,np.nanmin(nino34_ecearth,axis=0),np.nanmax(nino34_ecearth,axis=0),color='red',alpha=0.1)
ax[2,0].fill_between(np.arange(np.size(nino34_cesm,1))+1,np.nanmin(nino34_cesm,axis=0),np.nanmax(nino34_cesm,axis=0),color='blue',alpha=0.1)
ax[2,0].fill_between(np.arange(np.size(nino34_mpi,1))+1,np.nanmin(nino34_mpi,axis=0),np.nanmax(nino34_mpi,axis=0),color='gray',alpha=0.1)
ax[2,0].fill_between(np.arange(np.size(nino34_canesm,1))+1,np.nanmin(nino34_canesm,axis=0),np.nanmax(nino34_canesm,axis=0),color='green',alpha=0.1)
ax[2,0].fill_between(np.arange(np.size(nino34_access,1))+1,np.nanmin(nino34_access,axis=0),np.nanmax(nino34_access,axis=0),color='orange',alpha=0.1)
ax[2,0].set_ylabel(string_season + ' Niño3.4',fontsize=26)
ax[2,0].set_xticks(xrange)
ax[2,0].set_xticklabels(name_xticks)
ax[2,0].tick_params(labelsize=20)
ax[2,0].grid(linestyle='--')
ax[2,0].axis([-1, 133, -5, 8])
ax[2,0].legend(loc='upper left',fontsize=22,shadow=True,frameon=False,ncol=2)
ax[2,0].set_title('g',loc='left',fontsize=30,fontweight='bold')

# Previous DMI (IOD)
ax[2,1].plot(np.arange(np.size(dmi_access,1))+1,np.nanmean(dmi_access,axis=0),'-',color='orange',linewidth=4,label='ACCESS-ESM1.5')
ax[2,1].plot(np.arange(np.size(dmi_canesm,1))+1,np.nanmean(dmi_canesm,axis=0),'-',color='green',linewidth=4,label='CanESM5')
ax[2,1].plot(np.arange(np.size(dmi_cesm,1))+1,np.nanmean(dmi_cesm,axis=0),'-',color='blue',linewidth=4,label='CESM2')
ax[2,1].plot(np.arange(np.size(dmi_ecearth,1))+1,np.nanmean(dmi_ecearth,axis=0),'-',color='red',linewidth=4,label='EC-Earth3')
ax[2,1].plot(np.arange(np.size(dmi_mpi,1))+1,np.nanmean(dmi_mpi,axis=0),'-',color='gray',linewidth=4,label='MPI-ESM1.2-LR')
ax[2,1].plot(np.arange(np.size(dmi_obs))+1,dmi_obs,'k.-',linewidth=2,label='Observations')
ax[2,1].fill_between(np.arange(np.size(dmi_ecearth,1))+1,np.nanmin(dmi_ecearth,axis=0),np.nanmax(dmi_ecearth,axis=0),color='red',alpha=0.1)
ax[2,1].fill_between(np.arange(np.size(dmi_cesm,1))+1,np.nanmin(dmi_cesm,axis=0),np.nanmax(dmi_cesm,axis=0),color='blue',alpha=0.1)
ax[2,1].fill_between(np.arange(np.size(dmi_mpi,1))+1,np.nanmin(dmi_mpi,axis=0),np.nanmax(dmi_mpi,axis=0),color='gray',alpha=0.1)
ax[2,1].fill_between(np.arange(np.size(dmi_canesm,1))+1,np.nanmin(dmi_canesm,axis=0),np.nanmax(dmi_canesm,axis=0),color='green',alpha=0.1)
ax[2,1].fill_between(np.arange(np.size(dmi_access,1))+1,np.nanmin(dmi_access,axis=0),np.nanmax(dmi_access,axis=0),color='orange',alpha=0.1)
ax[2,1].set_ylabel(string_season + ' DMI (IOD)',fontsize=26)
ax[2,1].set_xticks(xrange)
ax[2,1].set_xticklabels(name_xticks)
ax[2,1].tick_params(labelsize=20)
ax[2,1].grid(linestyle='--')
ax[2,1].axis([-1, 133, -3, 4])
ax[2,1].legend(loc='upper left',fontsize=22,shadow=True,frameon=False,ncol=2)
ax[2,1].set_title('h',loc='left',fontsize=30,fontweight='bold')

plt.delaxes(ax[2,2])

# Save Fig.
if save_fig == True:
    fig.savefig(dir_fig + 'fig2.pdf')