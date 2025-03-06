#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Figure 4: Plot causal graphs for Antarctic OND

Last updated: 03/03/2025

@author: David Docquier
"""

# Import libraries
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# Options
save_fig = True
nvar = 8 # number of variables

# Working directories
dir_input = '/home/dadocq/Documents/Papers/My_Papers/RESIST_Antarctic/output/seasons/'
dir_fig = '/home/dadocq/Documents/Papers/My_Papers/RESIST_Antarctic/LaTeX/'

# Create graph
G = nx.DiGraph()
G.add_edges_from(
    [('SSIE','PSIE'),('PSIE','T$_{2m}$'),('T$_{2m}$','SST'),('SST','SAM'),
     ('SAM','ASL'),('ASL','N3.4'),('N3.4','DMI'),('DMI','SSIE')])
pos = {'SSIE': np.array([0,1]),
       'PSIE': np.array([-0.7,0.6]),
       'T$_{2m}$': np.array([-1,0]),
       'SST': np.array([0.7,0.6]),
       'SAM': np.array([-0.7,-0.6]),
       'ASL': np.array([0,-1]),
       'N3.4': np.array([1,0]),
       'DMI': np.array([0.7,-0.6])}

# ACCESS
drivers_SSIE_access = [('PSIE','SSIE'),('SAM','SSIE')]

drivers_PSIE_access = [('SST','PSIE'),('SAM','PSIE'),('ASL','PSIE')]
drivers_SAM_access = [('PSIE','SAM'),('SST','SAM')]

drivers_SST_access = [('PSIE','SST'),('SAM','SST')]
drivers_ASL_access = [('SST','ASL')]

cen_psie_access = 6
cen_t2m_access = 0
cen_sam_access = 5
cen_asl_access = 2
cen_dmi_access = 0
cen_n34_access = 0
cen_sst_access = 5

# CanESM
drivers_SSIE_canesm = [('PSIE','SSIE'),('T$_{2m}$','SSIE'),('SST','SSIE'),('SAM','SSIE')]

drivers_PSIE_canesm = [('T$_{2m}$','PSIE'),('SST','PSIE')]
drivers_T2m_canesm = [('PSIE','T$_{2m}$'),('SST','T$_{2m}$'),('SAM','T$_{2m}$')]
drivers_SST_canesm = [('PSIE','SST'),('T$_{2m}$','SST'),('SAM','SST'),('N3.4','SST'),('DMI','SST')]
drivers_SAM_canesm = [('ASL','SAM')]

drivers_N34_canesm = [('DMI','N3.4')]
drivers_DMI_canesm = [('N3.4','DMI')]

cen_psie_canesm = 5
cen_t2m_canesm = 6
cen_sam_canesm = 4
cen_asl_canesm = 1
cen_dmi_canesm = 3
cen_n34_canesm = 3
cen_sst_canesm = 8

# CESM
drivers_SSIE_cesm = [('PSIE','SSIE'),('SAM','SSIE')]

drivers_PSIE_cesm = [('SAM','PSIE'),('N3.4','PSIE')]

drivers_SST_cesm = [('PSIE','SST'),('SAM','SST'),('ASL','SST'),('N3.4','SST')]
drivers_N34_cesm = [('SST','N3.4'),('ASL','N3.4'),('DMI','N3.4')]

drivers_ASL_cesm = [('N3.4','ASL'),('DMI','ASL')]
drivers_DMI_cesm = [('SST','DMI'),('ASL','DMI'),('N3.4','DMI')]

cen_psie_cesm = 4
cen_t2m_cesm = 0
cen_sam_cesm = 3
cen_asl_cesm = 5
cen_dmi_cesm = 5
cen_n34_cesm = 7
cen_sst_cesm = 6

# EC-Earth
drivers_SSIE_ecearth = [('PSIE','SSIE'),('SST','SSIE')]

drivers_PSIE_ecearth = [('T$_{2m}$','PSIE'),('SST','PSIE')]
drivers_SST_ecearth = [('PSIE','SST'),('SAM','SST')]

drivers_T2m_ecearth = [('PSIE','T$_{2m}$'),('SST','T$_{2m}$'),('SAM','T$_{2m}$')]
drivers_SAM_ecearth = [('PSIE','SAM'),('SST','SAM'),('ASL','SAM')]

drivers_ASL_ecearth = [('SST','ASL')]

cen_psie_ecearth = 6
cen_t2m_ecearth = 4
cen_sam_ecearth = 5
cen_asl_ecearth = 2
cen_dmi_ecearth = 0
cen_n34_ecearth = 0
cen_sst_ecearth = 7

# MPI
drivers_SSIE_mpi = [('PSIE','SSIE'),('T$_{2m}$','SSIE'),('SST','SSIE')]

drivers_PSIE_mpi = [('T$_{2m}$','PSIE'),('SST','PSIE'),('SAM','PSIE')]
drivers_T2m_mpi = [('PSIE','T$_{2m}$'),('SST','T$_{2m}$'),('SAM','T$_{2m}$')]
drivers_SST_mpi = [('PSIE','SST'),('SAM','SST'),('ASL','SST')]

drivers_SAM_mpi = [('PSIE','SAM'),('SST','SAM')]
drivers_ASL_mpi = [('N3.4','ASL')]

drivers_N34_mpi = [('PSIE','N3.4'),('SST','N3.4'),('SAM','N3.4'),('DMI','N3.4')]

drivers_DMI_mpi = [('N3.4','DMI')]

cen_psie_mpi = 8
cen_t2m_mpi = 5
cen_sam_mpi = 6
cen_asl_mpi = 2
cen_dmi_mpi = 2
cen_n34_mpi = 6
cen_sst_mpi = 8

color_map = []
for node in G:
    if node == 'SSIE':
        color_map.append('red')
    elif node == 'PSIE':
        color_map.append('blue')
    elif node == 'T$_{2m}$':
        color_map.append('gray')
    elif node == 'SST':
        color_map.append('green')
    elif node == 'SAM':
        color_map.append('black')
    elif node == 'ASL':
        color_map.append('orange')
    elif node == 'N3.4':
        color_map.append('cyan')
    elif node == 'DMI':
        color_map.append('magenta')
    
# Causal graphs
fig,ax = plt.subplots(2,3,figsize=(35,24))
fig.subplots_adjust(left=0.05,bottom=0.05,right=0.95,top=0.9,wspace=0.2,hspace=0.2)

# ACCESS-ESM1.5
nx.draw_networkx_nodes(G,pos,node_color=color_map,node_size=4000,node_shape='o',alpha=0.4,ax=ax[0,0])
nx.draw_networkx_labels(G,pos,font_size=25,ax=ax[0,0])
nx.draw_networkx_edges(G,pos,edgelist=drivers_SSIE_access,edge_color='r',arrows=True,arrowsize=50,connectionstyle='arc3,rad=0.2',width=9,min_source_margin=30,min_target_margin=30,ax=ax[0,0])
nx.draw_networkx_edges(G,pos,edgelist=drivers_PSIE_access,edge_color='b',arrows=True,arrowsize=50,connectionstyle='arc3,rad=0.2',width=5,min_source_margin=30,min_target_margin=30,ax=ax[0,0])
nx.draw_networkx_edges(G,pos,edgelist=drivers_SAM_access,edge_color='k',arrows=True,arrowsize=50,connectionstyle='arc3,rad=0.2',width=5,min_source_margin=30,min_target_margin=30,ax=ax[0,0])
nx.draw_networkx_edges(G,pos,edgelist=drivers_SST_access,edge_color='g',arrows=True,style='--',arrowsize=50,connectionstyle='arc3,rad=0.2',width=5,min_source_margin=30,min_target_margin=30,ax=ax[0,0])
nx.draw_networkx_edges(G,pos,edgelist=drivers_ASL_access,edge_color='orange',arrows=True,style='--',arrowsize=50,connectionstyle='arc3,rad=0.2',width=5,min_source_margin=30,min_target_margin=30,ax=ax[0,0])
ax[0,0].set_title('(a) \n',loc='left',fontsize=35,fontweight='bold')
ax[0,0].set_title('ACCESS-ESM1.5 (PSIE [6]) \n',loc='center',fontsize=35)

# CanESM
nx.draw_networkx_nodes(G,pos,node_color=color_map,node_size=4000,node_shape='o',alpha=0.4,ax=ax[0,1])
nx.draw_networkx_labels(G,pos,font_size=25,ax=ax[0,1])
nx.draw_networkx_edges(G,pos,edgelist=drivers_SSIE_canesm,edge_color='r',arrows=True,arrowsize=50,connectionstyle='arc3,rad=0.2',width=9,min_source_margin=30,min_target_margin=30,ax=ax[0,1])
nx.draw_networkx_edges(G,pos,edgelist=drivers_PSIE_canesm,edge_color='b',arrows=True,arrowsize=50,connectionstyle='arc3,rad=0.2',width=5,min_source_margin=30,min_target_margin=30,ax=ax[0,1])
nx.draw_networkx_edges(G,pos,edgelist=drivers_T2m_canesm,edge_color='gray',arrows=True,arrowsize=50,connectionstyle='arc3,rad=0.2',width=5,min_source_margin=30,min_target_margin=30,ax=ax[0,1])
nx.draw_networkx_edges(G,pos,edgelist=drivers_SAM_canesm,edge_color='k',arrows=True,arrowsize=50,connectionstyle='arc3,rad=0.2',width=5,min_source_margin=30,min_target_margin=30,ax=ax[0,1])
nx.draw_networkx_edges(G,pos,edgelist=drivers_SST_canesm,edge_color='g',arrows=True,arrowsize=50,connectionstyle='arc3,rad=0.2',width=5,min_source_margin=30,min_target_margin=30,ax=ax[0,1])
nx.draw_networkx_edges(G,pos,edgelist=drivers_N34_canesm,edge_color='c',arrows=True,style='--',arrowsize=50,connectionstyle='arc3,rad=0.2',width=5,min_source_margin=30,min_target_margin=30,ax=ax[0,1])
nx.draw_networkx_edges(G,pos,edgelist=drivers_DMI_canesm,edge_color='m',arrows=True,style='--',arrowsize=50,connectionstyle='arc3,rad=0.2',width=5,min_source_margin=30,min_target_margin=30,ax=ax[0,1])
ax[0,1].set_title('(b) \n',loc='left',fontsize=35,fontweight='bold')
ax[0,1].set_title('CanESM5 (SST [8]) \n',loc='center',fontsize=35)

# CESM
nx.draw_networkx_nodes(G,pos,node_color=color_map,node_size=4000,node_shape='o',alpha=0.4,ax=ax[0,2])
nx.draw_networkx_labels(G,pos,font_size=25,ax=ax[0,2])
nx.draw_networkx_edges(G,pos,edgelist=drivers_SSIE_cesm,edge_color='r',arrows=True,arrowsize=50,connectionstyle='arc3,rad=0.2',width=9,min_source_margin=30,min_target_margin=30,ax=ax[0,2])
nx.draw_networkx_edges(G,pos,edgelist=drivers_PSIE_cesm,edge_color='b',arrows=True,arrowsize=50,connectionstyle='arc3,rad=0.2',width=5,min_source_margin=30,min_target_margin=30,ax=ax[0,2])
nx.draw_networkx_edges(G,pos,edgelist=drivers_SST_cesm,edge_color='g',arrows=True,style='--',arrowsize=50,connectionstyle='arc3,rad=0.2',width=5,min_source_margin=30,min_target_margin=30,ax=ax[0,2])
nx.draw_networkx_edges(G,pos,edgelist=drivers_N34_cesm,edge_color='c',arrows=True,style='--',arrowsize=50,connectionstyle='arc3,rad=0.2',width=5,min_source_margin=30,min_target_margin=30,ax=ax[0,2])
nx.draw_networkx_edges(G,pos,edgelist=drivers_ASL_cesm,edge_color='orange',arrows=True,arrowsize=50,connectionstyle='arc3,rad=0.2',width=0.5,min_source_margin=30,min_target_margin=30,ax=ax[0,2])
nx.draw_networkx_edges(G,pos,edgelist=drivers_DMI_cesm,edge_color='m',arrows=True,arrowsize=50,connectionstyle='arc3,rad=0.2',width=0.5,min_source_margin=30,min_target_margin=30,ax=ax[0,2])
ax[0,2].set_title('(c) \n',loc='left',fontsize=35,fontweight='bold')
ax[0,2].set_title('CESM2 (N3.4 [7]) \n',loc='center',fontsize=35)

# EC-Earth
nx.draw_networkx_nodes(G,pos,node_color=color_map,node_size=4000,node_shape='o',alpha=0.4,ax=ax[1,0])
nx.draw_networkx_labels(G,pos,font_size=25,ax=ax[1,0])
nx.draw_networkx_edges(G,pos,edgelist=drivers_SSIE_ecearth,edge_color='r',arrows=True,arrowsize=50,connectionstyle='arc3,rad=0.2',width=9,min_source_margin=30,min_target_margin=30,ax=ax[1,0])
nx.draw_networkx_edges(G,pos,edgelist=drivers_PSIE_ecearth,edge_color='b',arrows=True,arrowsize=50,connectionstyle='arc3,rad=0.2',width=5,min_source_margin=30,min_target_margin=30,ax=ax[1,0])
nx.draw_networkx_edges(G,pos,edgelist=drivers_SST_ecearth,edge_color='g',arrows=True,arrowsize=50,connectionstyle='arc3,rad=0.2',width=5,min_source_margin=30,min_target_margin=30,ax=ax[1,0])
nx.draw_networkx_edges(G,pos,edgelist=drivers_T2m_ecearth,edge_color='gray',arrows=True,style='--',arrowsize=50,connectionstyle='arc3,rad=0.2',width=5,min_source_margin=30,min_target_margin=30,ax=ax[1,0])
nx.draw_networkx_edges(G,pos,edgelist=drivers_SAM_ecearth,edge_color='k',arrows=True,style='--',arrowsize=50,connectionstyle='arc3,rad=0.2',width=5,min_source_margin=30,min_target_margin=30,ax=ax[1,0])
nx.draw_networkx_edges(G,pos,edgelist=drivers_ASL_ecearth,edge_color='orange',arrows=True,arrowsize=50,connectionstyle='arc3,rad=0.2',width=0.5,min_source_margin=30,min_target_margin=30,ax=ax[1,0])
ax[1,0].set_title('(d) \n',loc='left',fontsize=35,fontweight='bold')
ax[1,0].set_title('EC-Earth3 (SST [7]) \n',loc='center',fontsize=35)

# MPI
nx.draw_networkx_nodes(G,pos,node_color=color_map,node_size=4000,node_shape='o',alpha=0.4,ax=ax[1,1])
nx.draw_networkx_labels(G,pos,font_size=25,ax=ax[1,1])
nx.draw_networkx_edges(G,pos,edgelist=drivers_SSIE_mpi,edge_color='r',arrows=True,arrowsize=50,connectionstyle='arc3,rad=0.2',width=9,min_source_margin=30,min_target_margin=30,ax=ax[1,1])
nx.draw_networkx_edges(G,pos,edgelist=drivers_PSIE_mpi,edge_color='b',arrows=True,arrowsize=50,connectionstyle='arc3,rad=0.2',width=5,min_source_margin=30,min_target_margin=30,ax=ax[1,1])
nx.draw_networkx_edges(G,pos,edgelist=drivers_T2m_mpi,edge_color='gray',arrows=True,arrowsize=50,connectionstyle='arc3,rad=0.2',width=5,min_source_margin=30,min_target_margin=30,ax=ax[1,1])
nx.draw_networkx_edges(G,pos,edgelist=drivers_SST_mpi,edge_color='green',arrows=True,arrowsize=50,connectionstyle='arc3,rad=0.2',width=5,min_source_margin=30,min_target_margin=30,ax=ax[1,1])
nx.draw_networkx_edges(G,pos,edgelist=drivers_SAM_mpi,edge_color='k',arrows=True,style='--',arrowsize=50,connectionstyle='arc3,rad=0.2',width=5,min_source_margin=30,min_target_margin=30,ax=ax[1,1])
nx.draw_networkx_edges(G,pos,edgelist=drivers_ASL_mpi,edge_color='orange',arrows=True,style='--',arrowsize=50,connectionstyle='arc3,rad=0.2',width=5,min_source_margin=30,min_target_margin=30,ax=ax[1,1])
nx.draw_networkx_edges(G,pos,edgelist=drivers_N34_mpi,edge_color='c',arrows=True,arrowsize=50,connectionstyle='arc3,rad=0.2',width=0.5,min_source_margin=30,min_target_margin=30,ax=ax[1,1])
nx.draw_networkx_edges(G,pos,edgelist=drivers_DMI_mpi,edge_color='m',arrows=True,arrowsize=50,connectionstyle='arc3,rad=0.2',width=0.5,min_source_margin=30,min_target_margin=30,ax=ax[1,1])
ax[1,1].set_title('(e) \n',loc='left',fontsize=35,fontweight='bold')
ax[1,1].set_title('MPI-ESM1.2-LR (PSIE, SST [8]) \n',loc='center',fontsize=35)

plt.delaxes(ax=ax[1,2])

# Save figure
if save_fig == True:
    fig.savefig(dir_fig + 'fig4.pdf')