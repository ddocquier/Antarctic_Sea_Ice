#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
GOAL
    Figure 1: Map of Antarctic sectors according to Zwally et al. (1983)
PROGRAMMER
    D. Docquier
LAST UPDATE
    15/01/2025
'''

# Standard libraries
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER,LATITUDE_FORMATTER
import matplotlib.ticker as mticker
import matplotlib.path as mpath

# Options
save_fig = True

# Working directories
dir_fig = '/home/dadocq/Documents/Papers/My_Papers/RESIST_Antarctic/LaTeX/'

# Cartopy projection
proj = ccrs.SouthPolarStereo()

# Create circle
r_extent = 4651194.319 * 1.005
circle_path = mpath.Path.unit_circle()
circle_path = mpath.Path(circle_path.vertices.copy() * r_extent, circle_path.codes.copy())

# Sector boundaries
lat_SP1,lon_SP1 = -82.,-60.
lat_60W,lon_60W = -50.,-60.
lat_SP2,lon_SP2 = -70.,20.
lat_20E,lon_20E = -50.,20.
lat_SP3,lon_SP3 = -67.5,90.
lat_90E,lon_90E = -50.,90.
lat_SP4,lon_SP4 = -70.,160.
lat_160E,lon_160E = -50.,160.
lat_SP5,lon_SP5 = -74.5,-130.
lat_130W,lon_130W = -50.,-130.

#######
# Map #
####### 
 
# Create map with lines and labels   
fig = plt.figure(figsize=(9,9))
ax = fig.add_subplot(projection=proj)
ax.set_extent([-180,180,-90,-50],crs=ccrs.PlateCarree())
ax.coastlines()
gl = ax.gridlines(draw_labels=True,color='lightgray',linestyle='--',linewidth=1)
gl.xlocator = mticker.FixedLocator(np.arange(-180,181,45))
gl.xformatter = LONGITUDE_FORMATTER
gl.ylocator = mticker.FixedLocator(np.arange(-90,-50,10))
gl.yformatter = LATITUDE_FORMATTER
gl.xlabel_style = {'size': 14}
gl.ylabel_style = {'size': 14}
ax.add_feature(cfeature.LAND)
ax.set_boundary(circle_path)

# Plot sector boundaries
geodetic = ccrs.Geodetic()
lon_SP_t1,lat_SP_t1 = ccrs.PlateCarree().transform_point(lon_SP1,lat_SP1,geodetic)
lon_SP_t2,lat_SP_t2 = ccrs.PlateCarree().transform_point(lon_SP2,lat_SP2,geodetic)
lon_SP_t3,lat_SP_t3 = ccrs.PlateCarree().transform_point(lon_SP3,lat_SP3,geodetic)
lon_SP_t4,lat_SP_t4 = ccrs.PlateCarree().transform_point(lon_SP4,lat_SP4,geodetic)
lon_SP_t5,lat_SP_t5 = ccrs.PlateCarree().transform_point(lon_SP5,lat_SP5,geodetic)
lon_60W_t,lat_60W_t = ccrs.PlateCarree().transform_point(lon_60W,lat_60W,geodetic)
lon_20E_t,lat_20E_t = ccrs.PlateCarree().transform_point(lon_20E,lat_20E,geodetic)
lon_90E_t,lat_90E_t = ccrs.PlateCarree().transform_point(lon_90E,lat_90E,geodetic)
lon_160E_t,lat_160E_t = ccrs.PlateCarree().transform_point(lon_160E,lat_160E,geodetic)
lon_130W_t,lat_130W_t = ccrs.PlateCarree().transform_point(lon_130W,lat_130W,geodetic)
plt.plot([lon_SP_t1,lon_60W_t],[lat_SP_t1,lat_60W_t],color='black',linewidth=1,linestyle='-.',transform=ccrs.PlateCarree())
plt.plot([lon_SP_t2,lon_20E_t],[lat_SP_t2,lat_20E_t],color='black',linewidth=1,linestyle='-.',transform=ccrs.PlateCarree())
plt.plot([lon_SP_t3,lon_90E_t],[lat_SP_t3,lat_90E_t],color='black',linewidth=1,linestyle='-.',transform=ccrs.PlateCarree())
plt.plot([lon_SP_t4,lon_160E_t],[lat_SP_t4,lat_160E_t],color='black',linewidth=1,linestyle='-.',transform=ccrs.PlateCarree())
plt.plot([lon_SP_t5,lon_130W_t],[lat_SP_t5,lat_130W_t],color='black',linewidth=1,linestyle='-.',transform=ccrs.PlateCarree())

# Add text for sectors
ax.text(-100,-62,'BAS',weight='bold',fontsize=18,transform=ccrs.PlateCarree())
ax.text(-20,-62,'WS',weight='bold',fontsize=18,transform=ccrs.PlateCarree())
ax.text(60,-62,'IO',weight='bold',fontsize=18,transform=ccrs.PlateCarree())
ax.text(125,-62,'WPO',weight='bold',fontsize=18,transform=ccrs.PlateCarree())
ax.text(-160,-62,'RS',weight='bold',fontsize=18,transform=ccrs.PlateCarree())

# Save figure
filename = dir_fig + 'fig1.pdf'
fig.savefig(filename)