#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Figure 8: Plot causal graphs from CanESM5 (cold model) for all sectors in OND

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

# Antarctic
drivers_SSIE_ant = [('PSIE','SSIE'),('T$_{2m}$','SSIE'),('SST','SSIE'),('SAM','SSIE')]

drivers_PSIE_ant = [('T$_{2m}$','PSIE'),('SST','PSIE')]
drivers_T2m_ant = [('PSIE','T$_{2m}$'),('SST','T$_{2m}$'),('SAM','T$_{2m}$')]
drivers_SST_ant = [('PSIE','SST'),('T$_{2m}$','SST'),('SAM','SST'),('N3.4','SST'),('DMI','SST')]
drivers_SAM_ant = [('ASL','SAM')]

drivers_N34_ant = [('DMI','N3.4')]
drivers_DMI_ant = [('N3.4','DMI')]

cen_psie_ant = 5
cen_t2m_ant = 6
cen_sam_ant = 4
cen_asl_ant = 1
cen_dmi_ant = 3
cen_n34_ant = 3
cen_sst_ant = 8

# BAS
drivers_SSIE_bas = [('PSIE','SSIE'),('T$_{2m}$','SSIE'),('SST','SSIE')]

drivers_PSIE_bas = [('T$_{2m}$','PSIE'),('SST','PSIE')]
drivers_T2m_bas = [('PSIE','T$_{2m}$'),('SST','T$_{2m}$')]
drivers_SST_bas = [('PSIE','SST'),('T$_{2m}$','SST'),('SAM','SST')]

drivers_SAM_bas = [('ASL','SAM')]

cen_psie_bas = 5
cen_t2m_bas = 5
cen_sam_bas = 2
cen_asl_bas = 1
cen_dmi_bas = 0
cen_n34_bas = 0
cen_sst_bas = 6

# WS
drivers_SSIE_ws = [('PSIE','SSIE'),('T$_{2m}$','SSIE'),('SST','SSIE'),('SAM','SSIE')]

drivers_PSIE_ws = [('SST','PSIE'),('SAM','PSIE')]
drivers_SST_ws = [('PSIE','SST'),('T$_{2m}$','SST')]
drivers_SAM_ws = [('ASL','SAM')]

cen_psie_ws = 4
cen_t2m_ws = 2
cen_sam_ws = 3
cen_asl_ws = 1
cen_dmi_ws = 0
cen_n34_ws = 0
cen_sst_ws = 4

# IO
drivers_SSIE_io = [('SST','SSIE')]

drivers_SST_io = [('SAM','SST')]

drivers_SAM_io = [('ASL','SAM')]

cen_psie_io = 0
cen_t2m_io = 0
cen_sam_io = 2
cen_asl_io = 1
cen_dmi_io = 0
cen_n34_io = 0
cen_sst_io = 2

# WPO
drivers_SSIE_wpo = [('N3.4','SSIE'),('DMI','SSIE')]

drivers_N34_wpo = [('DMI','N3.4')]
drivers_DMI_wpo = [('N3.4','DMI')]

cen_psie_wpo = 0
cen_t2m_wpo = 0
cen_sam_wpo = 0
cen_asl_wpo = 0
cen_dmi_wpo = 3
cen_n34_wpo = 3
cen_sst_wpo = 0

# RS
drivers_SSIE_rs = [('PSIE','SSIE'),('SST','SSIE'),('SAM','SSIE')]

drivers_PSIE_rs = [('SST','PSIE'),('SAM','PSIE'),('N3.4','PSIE'),('DMI','PSIE')]
drivers_SST_rs = [('PSIE','SST'),('SAM','SST'),('DMI','SST')]
drivers_SAM_rs = [('ASL','SAM')]

drivers_N34_rs = [('PSIE','N3.4'),('T$_{2m}$','N3.4'),('SST','N3.4'),('DMI','N3.4')]
drivers_DMI_rs = [('N3.4','DMI')]

drivers_T2m_rs = [('PSIE','T$_{2m}$'),('SST','T$_{2m}$'),('SAM','T$_{2m}$')]

cen_psie_rs = 8
cen_t2m_rs = 4
cen_sam_rs = 5
cen_asl_rs = 1
cen_dmi_rs = 4
cen_n34_rs = 6
cen_sst_rs = 7

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

# Antarctic
nx.draw_networkx_nodes(G,pos,node_color=color_map,node_size=4000,node_shape='o',alpha=0.4,ax=ax[0,0])
nx.draw_networkx_labels(G,pos,font_size=25,ax=ax[0,0])
nx.draw_networkx_edges(G,pos,edgelist=drivers_SSIE_ant,edge_color='r',arrows=True,arrowsize=50,connectionstyle='arc3,rad=0.2',width=9,min_source_margin=30,min_target_margin=30,ax=ax[0,0])
nx.draw_networkx_edges(G,pos,edgelist=drivers_PSIE_ant,edge_color='b',arrows=True,arrowsize=50,connectionstyle='arc3,rad=0.2',width=5,min_source_margin=30,min_target_margin=30,ax=ax[0,0])
nx.draw_networkx_edges(G,pos,edgelist=drivers_T2m_ant,edge_color='gray',arrows=True,arrowsize=50,connectionstyle='arc3,rad=0.2',width=5,min_source_margin=30,min_target_margin=30,ax=ax[0,0])
nx.draw_networkx_edges(G,pos,edgelist=drivers_SAM_ant,edge_color='k',arrows=True,arrowsize=50,connectionstyle='arc3,rad=0.2',width=5,min_source_margin=30,min_target_margin=30,ax=ax[0,0])
nx.draw_networkx_edges(G,pos,edgelist=drivers_SST_ant,edge_color='g',arrows=True,arrowsize=50,connectionstyle='arc3,rad=0.2',width=5,min_source_margin=30,min_target_margin=30,ax=ax[0,0])
nx.draw_networkx_edges(G,pos,edgelist=drivers_N34_ant,edge_color='c',arrows=True,style='--',arrowsize=50,connectionstyle='arc3,rad=0.2',width=5,min_source_margin=30,min_target_margin=30,ax=ax[0,0])
nx.draw_networkx_edges(G,pos,edgelist=drivers_DMI_ant,edge_color='m',arrows=True,style='--',arrowsize=50,connectionstyle='arc3,rad=0.2',width=5,min_source_margin=30,min_target_margin=30,ax=ax[0,0])
ax[0,0].set_title('(a) \n',loc='left',fontsize=35,fontweight='bold')
ax[0,0].set_title('Pan-Antarctic (SST [8]) \n',loc='center',fontsize=35)

# BAS
nx.draw_networkx_nodes(G,pos,node_color=color_map,node_size=4000,node_shape='o',alpha=0.4,ax=ax[0,1])
nx.draw_networkx_labels(G,pos,font_size=25,ax=ax[0,1])
nx.draw_networkx_edges(G,pos,edgelist=drivers_SSIE_bas,edge_color='r',arrows=True,arrowsize=50,connectionstyle='arc3,rad=0.2',width=9,min_source_margin=30,min_target_margin=30,ax=ax[0,1])
nx.draw_networkx_edges(G,pos,edgelist=drivers_PSIE_bas,edge_color='b',arrows=True,arrowsize=50,connectionstyle='arc3,rad=0.2',width=5,min_source_margin=30,min_target_margin=30,ax=ax[0,1])
nx.draw_networkx_edges(G,pos,edgelist=drivers_T2m_bas,edge_color='gray',arrows=True,arrowsize=50,connectionstyle='arc3,rad=0.2',width=5,min_source_margin=30,min_target_margin=30,ax=ax[0,1])
nx.draw_networkx_edges(G,pos,edgelist=drivers_SST_bas,edge_color='g',arrows=True,arrowsize=50,connectionstyle='arc3,rad=0.2',width=5,min_source_margin=30,min_target_margin=30,ax=ax[0,1])
nx.draw_networkx_edges(G,pos,edgelist=drivers_SAM_bas,edge_color='k',arrows=True,style='--',arrowsize=50,connectionstyle='arc3,rad=0.2',width=5,min_source_margin=30,min_target_margin=30,ax=ax[0,1])
ax[0,1].set_title('(b) \n',loc='left',fontsize=35,fontweight='bold')
ax[0,1].set_title('Bell.-Amundsen Seas (SST [6]) \n',loc='center',fontsize=35)

# WS
nx.draw_networkx_nodes(G,pos,node_color=color_map,node_size=4000,node_shape='o',alpha=0.4,ax=ax[0,2])
nx.draw_networkx_labels(G,pos,font_size=25,ax=ax[0,2])
nx.draw_networkx_edges(G,pos,edgelist=drivers_SSIE_ws,edge_color='r',arrows=True,arrowsize=50,connectionstyle='arc3,rad=0.2',width=9,min_source_margin=30,min_target_margin=30,ax=ax[0,2])
nx.draw_networkx_edges(G,pos,edgelist=drivers_PSIE_ws,edge_color='b',arrows=True,arrowsize=50,connectionstyle='arc3,rad=0.2',width=5,min_source_margin=30,min_target_margin=30,ax=ax[0,2])
nx.draw_networkx_edges(G,pos,edgelist=drivers_SST_ws,edge_color='g',arrows=True,arrowsize=50,connectionstyle='arc3,rad=0.2',width=5,min_source_margin=30,min_target_margin=30,ax=ax[0,2])
nx.draw_networkx_edges(G,pos,edgelist=drivers_SAM_ws,edge_color='k',arrows=True,arrowsize=50,connectionstyle='arc3,rad=0.2',width=5,min_source_margin=30,min_target_margin=30,ax=ax[0,2])
ax[0,2].set_title('(c) \n',loc='left',fontsize=35,fontweight='bold')
ax[0,2].set_title('Weddell Sea (PSIE, SST [4]) \n',loc='center',fontsize=35)

# IO
nx.draw_networkx_nodes(G,pos,node_color=color_map,node_size=4000,node_shape='o',alpha=0.4,ax=ax[1,0])
nx.draw_networkx_labels(G,pos,font_size=25,ax=ax[1,0])
nx.draw_networkx_edges(G,pos,edgelist=drivers_SSIE_io,edge_color='r',arrows=True,arrowsize=50,connectionstyle='arc3,rad=0.2',width=9,min_source_margin=30,min_target_margin=30,ax=ax[1,0])
nx.draw_networkx_edges(G,pos,edgelist=drivers_SST_io,edge_color='g',arrows=True,arrowsize=50,connectionstyle='arc3,rad=0.2',width=5,min_source_margin=30,min_target_margin=30,ax=ax[1,0])
nx.draw_networkx_edges(G,pos,edgelist=drivers_SAM_io,edge_color='k',arrows=True,style='--',arrowsize=50,connectionstyle='arc3,rad=0.2',width=5,min_source_margin=30,min_target_margin=30,ax=ax[1,0])
ax[1,0].set_title('(d) \n',loc='left',fontsize=35,fontweight='bold')
ax[1,0].set_title('Indian Ocean (SST [2]) \n',loc='center',fontsize=35)

# WPO
nx.draw_networkx_nodes(G,pos,node_color=color_map,node_size=4000,node_shape='o',alpha=0.4,ax=ax[1,1])
nx.draw_networkx_labels(G,pos,font_size=25,ax=ax[1,1])
nx.draw_networkx_edges(G,pos,edgelist=drivers_SSIE_wpo,edge_color='r',arrows=True,arrowsize=50,connectionstyle='arc3,rad=0.2',width=9,min_source_margin=30,min_target_margin=30,ax=ax[1,1])
nx.draw_networkx_edges(G,pos,edgelist=drivers_N34_wpo,edge_color='c',arrows=True,arrowsize=50,connectionstyle='arc3,rad=0.2',width=5,min_source_margin=30,min_target_margin=30,ax=ax[1,1])
nx.draw_networkx_edges(G,pos,edgelist=drivers_DMI_wpo,edge_color='m',arrows=True,arrowsize=50,connectionstyle='arc3,rad=0.2',width=5,min_source_margin=30,min_target_margin=30,ax=ax[1,1])
ax[1,1].set_title('(e) \n',loc='left',fontsize=35,fontweight='bold')
ax[1,1].set_title('West. Pac. Oc. (DMI, N3.4 [3]) \n',loc='center',fontsize=35)

# RS
nx.draw_networkx_nodes(G,pos,node_color=color_map,node_size=4000,node_shape='o',alpha=0.4,ax=ax[1,2])
nx.draw_networkx_labels(G,pos,font_size=25,ax=ax[1,2])
nx.draw_networkx_edges(G,pos,edgelist=drivers_SSIE_rs,edge_color='r',arrows=True,arrowsize=50,connectionstyle='arc3,rad=0.2',width=9,min_source_margin=30,min_target_margin=30,ax=ax[1,2])
nx.draw_networkx_edges(G,pos,edgelist=drivers_PSIE_rs,edge_color='b',arrows=True,arrowsize=50,connectionstyle='arc3,rad=0.2',width=5,min_source_margin=30,min_target_margin=30,ax=ax[1,2])
nx.draw_networkx_edges(G,pos,edgelist=drivers_SST_rs,edge_color='g',arrows=True,arrowsize=50,connectionstyle='arc3,rad=0.2',width=5,min_source_margin=30,min_target_margin=30,ax=ax[1,2])
nx.draw_networkx_edges(G,pos,edgelist=drivers_SAM_rs,edge_color='k',arrows=True,arrowsize=50,connectionstyle='arc3,rad=0.2',width=5,min_source_margin=30,min_target_margin=30,ax=ax[1,2])
nx.draw_networkx_edges(G,pos,edgelist=drivers_N34_rs,edge_color='c',arrows=True,style='--',arrowsize=50,connectionstyle='arc3,rad=0.2',width=5,min_source_margin=30,min_target_margin=30,ax=ax[1,2])
nx.draw_networkx_edges(G,pos,edgelist=drivers_DMI_rs,edge_color='m',arrows=True,style='--',arrowsize=50,connectionstyle='arc3,rad=0.2',width=5,min_source_margin=30,min_target_margin=30,ax=ax[1,2])
nx.draw_networkx_edges(G,pos,edgelist=drivers_T2m_rs,edge_color='gray',arrows=True,arrowsize=50,connectionstyle='arc3,rad=0.2',width=0.5,min_source_margin=30,min_target_margin=30,ax=ax[1,2])
ax[1,2].set_title('(f) \n',loc='left',fontsize=35,fontweight='bold')
ax[1,2].set_title('Ross Sea (PSIE [7]) \n',loc='center',fontsize=35)

# Save figure
if save_fig == True:
    fig.savefig(dir_fig + 'fig8.pdf')