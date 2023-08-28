import os
os.environ['USE_PYGEOS'] = '0'
import sys
import time
import json
import requests
import warnings
import numpy as np
import numpy.random as rand
import pandas as pd
import geopandas as gpd
import contextily as cx
import scipy.stats as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mtick
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm
from shapely.ops import cascaded_union
from itertools import combinations
from shapely.geometry import Point,Polygon,MultiPolygon
from scipy.stats import t
from scipy.stats._continuous_distns import _distn_names
from cycler import cycler

color_scheme_7_0=["#58b5e1", "#316387", "#40ceae", "#285d28", "#ade64f", "#63a122", "#2ce462"]

#Defining some 4 pronged color schemes (source: http://vrl.cs.brown.edu/color)
color_scheme_4_0=["#8de4d3", "#0e503e", "#43e26d", "#2da0a1"]
color_scheme_4_1=["#069668", "#49edc9", "#2d595a", "#8dd2d8"]

#Defining some 3 pronged color schemes (source: http://vrl.cs.brown.edu/color)
color_scheme_3_0=["#72e5ef", "#1c5b5a", "#2da0a1"]
color_scheme_3_1=["#256676", "#72b6bc", "#1eefc9"]
color_scheme_3_2=['#40655e', '#a2e0dd', '#31d0a5']

#Defining some 2 pronged color schemes (source: http://vrl.cs.brown.edu/color)
color_scheme_2_0=["#21f0b6", "#2a6866"]
color_scheme_2_1=["#72e5ef", "#3a427d"]
color_scheme_2_2=["#6f309f", "#dfccfa"]

#Distributions to try (scipy.stats continuous distributions)
dist_names=['alpha','beta','gamma','logistic','norm','lognorm']
dist_labels=['Alpha','Beta','Gamma','Logistic','Normal','Log Normal']

#Named color schemes from https://www.canva.com/learn/100-color-combinations/

colors={
	'day_night':["#e6df44","#f0810f","#063852","#011a27"],
	'beach_house':["#d5c9b1","#e05858","#bfdccf","#5f968e"],
	'autumn':["#db9501","#c05805","#6e6702","#2e2300"],
	'ocean':["#003b46","#07575b","#66a5ad","#c4dfe6"],
	'forest':["#7d4427","#a2c523","#486b00","#2e4600"],
	'aqua':["#004d47","#128277","#52958b","#b9c4c9"],
	'field':["#5a5f37","#fffae1","#524a3a","#919636"],
	'misty':["#04202c","#304040","#5b7065","#c9d1c8"],
	'greens':["#265c00","#68a225","#b3de81","#fdffff"],
	'citroen':["#b38540","#563e20","#7e7b15","#ebdf00"],
	'blues':["#1e1f26","#283655","#4d648d","#d0e1f9"],
	'dusk':["#363237","#2d4262","#73605b","#d09683"],
	'ice':["#1995ad","#a1d6e2","#bcbabe","#f1f1f2"],
}

hatches_default=['/','\\','|','-','+','x','o','O','.','*']

continental_us=([1,4,5,6,8,9,10,11,12,13,16,17,18,19,20,21,22,23,24,25,26,
	27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,44,45,46,47,48,49,50,
	51,53,54,55,56])

alaska=2
hawaii=15

def BackgroundMap(x,y,margin=.05,zoom='auto'):

	maxx=x.max()
	minx=x.min()
	maxy=y.max()
	miny=y.min()

	west=minx-(maxx-minx)*margin
	east=maxx+(maxx-minx)*margin
	south=miny-(maxy-miny)*margin
	north=maxy+(maxy-miny)*margin

	print(west,south,east,north)

	map_img,map_ext=cx.bounds2img(west,south,east,north,ll=True,
		source=cx.providers.OpenStreetMap.Mapnik,zoom=zoom)
	map_img,map_ext=cx.warp_tiles(map_img,map_ext)

	return map_img,map_ext

def PlotMeanCI(x,y,figsize=(8,8),colors=color_scheme_2_1,ax=None,
	poly_kwargs={},line_kwargs={},axes_kwargs={}):

	cmap=LinearSegmentedColormap.from_list('custom', colors, N=256)

	return_fig=False
	if ax==None:
		fig,ax=plt.subplots(figsize=figsize)
		return_fig=True

	mean=y.mean(axis=0)
	stdev=y.std(axis=0)

	ax.plot(x,mean,color=cmap(.99),**line_kwargs)
	ax.fill_between(x,mean-stdev,mean+stdev,color=cmap(0),**poly_kwargs)

	ax.set(**axes_kwargs)

	if return_fig:
		return fig

def PlotLine(x,y,figsize=(8,8),colors=color_scheme_2_1,ax=None,
	poly_kwargs={},line_kwargs={},axes_kwargs={}):

	return_fig=False
	if ax==None:
		fig,ax=plt.subplots(figsize=figsize)
		return_fig=True

	ax.plot(x,y,**line_kwargs)

	ax.set(**axes_kwargs)

	if return_fig:
		return fig

def PlotContour(x_grid,y_grid,c_values,figsize=(8,8),ax=None,colors=color_scheme_2_1,
	contour_kwargs={},axes_kwargs={},colorbar_kwargs={}):
	
	if type(colors)==str:
		cmap=colors
	else:
		cmap=LinearSegmentedColormap.from_list('custom', colors, N=256)

	return_fig=False
	if ax is None:
		fig,ax=plt.subplots(figsize=figsize)
		return_fig=True
	
	cs=ax.contourf(x_grid,y_grid,c_values,cmap=cmap,**contour_kwargs)

	sm=plt.cm.ScalarMappable(cmap=cmap)
	plt.colorbar(sm,ax=ax,**colorbar_kwargs)

	ax.set(**axes_kwargs)
	

	if return_fig:
		return fig


def PlotGraph(graph,figsize=(8,8),colors=color_scheme_2_1,ax=None,
	scatter_kwargs={},line_kwargs={},axes_kwargs={},text_kwargs={},show_lines=True):

	cmap=LinearSegmentedColormap.from_list('custom', colors, N=256)

	return_fig=False
	if ax==None:
		fig,ax=plt.subplots(figsize=figsize)
		return_fig=True

	if 'label' in scatter_kwargs:
		generate_label=False
	else:
		generate_label=True

	for idx,component in enumerate(graph.components):
		color=cmap(idx/len(graph.components))
		scatter_kwargs['color']=color
		if generate_label:
			scatter_kwargs['label']='Cluster {:.0f}'.format(idx+1)
		# print('idx',idx)
		PlotComponent(component,ax=ax,scatter_kwargs=scatter_kwargs,line_kwargs=line_kwargs,
			text_kwargs=text_kwargs,show_lines=show_lines)

	if generate_label:
		del scatter_kwargs['label']

	ax.set(**axes_kwargs)

	if return_fig:
		return fig

def PlotGiantComponent(graph,figsize=(8,8),colors=color_scheme_2_1,ax=None,
	scatter_kwargs={},line_kwargs={},axes_kwargs={},text_kwargs={},show_lines=True):

	return_fig=False
	if ax==None:
		fig,ax=plt.subplots(figsize=figsize)
		return_fig=True

	component_sizes=np.zeros(len(graph.components))
	for idx,component in enumerate(graph.components):
		component_sizes[idx]=len(component.vertices)

	PlotComponent(graph.components[np.argmax(component_sizes)],
		ax=ax,scatter_kwargs=scatter_kwargs,line_kwargs=line_kwargs,
		text_kwargs=text_kwargs,show_lines=show_lines)

	ax.set(**axes_kwargs)

	if return_fig:
		return fig


def PlotComponent(component,figsize=(8,8),colors=color_scheme_2_1,ax=None,
	scatter_kwargs={},line_kwargs={},axes_kwargs={},text_kwargs={},colorbar_kwargs={},
	show_lines=True):

	cmap=LinearSegmentedColormap.from_list('custom', colors, N=256)

	return_fig=False
	if ax==None:
		fig,ax=plt.subplots(figsize=figsize)
		return_fig=True

	n_v=len(component.vertices)
	x=np.zeros(n_v)
	y=np.zeros(n_v)

	edges=[]

	for idx in range(n_v):
		x[idx]=component.vertices[idx]['x']
		y[idx]=component.vertices[idx]['y']
		edges.extend(component.vertices[idx]['edges'])
		# ax.text(x[idx],y[idx],component.vertices[idx]['id'],color='k',**text_kwargs)

	sm=ax.scatter(x,y,**scatter_kwargs)

	if show_lines:
		n_e=len(edges)
		if n_e>0:
			line_x=np.zeros((n_e,2))
			line_y=np.zeros((n_e,2))
			keep=np.array([False]*n_e)

			for idx,edge in enumerate(edges):
				if edge['present']:
					# print(edge['vertices'])
					line_x[idx]=edge['x']
					line_y[idx]=edge['y']
					keep[idx]=True

			# print(line_x)
			# print(n_e,keep)
			line_x=line_x[keep]
			line_y=line_y[keep]

			ax.plot(line_x.T,line_y.T,**line_kwargs)

	# sm=plt.cm.ScalarMappable(cmap=cmap)
	if colorbar_kwargs:
		plt.colorbar(sm,ax=ax,**colorbar_kwargs)

	ax.set(**axes_kwargs)

	if return_fig:
		return fig

def SeriesPlot(x,y,labels=[],figsize=(8,8),colors=color_scheme_2_1,ax=None,
	line_kwargs={},axes_kwargs={}):

	cmap=LinearSegmentedColormap.from_list('custom', colors, N=256)

	return_fig=False
	if ax==None:
		fig,ax=plt.subplots(figsize=figsize)
		return_fig=True

	if labels:
		for idx in range(len(y)):
			color=cmap(np.interp(idx,[0,len(y)],[0,.99]))
			ax.plot(x,y[idx],color=color,label=labels[idx],**line_kwargs)
			ax.legend()
	else:
		for idx in range(len(y)):
			color=cmap(np.interp(idx,[0,len(y)],[0,.99]))
			ax.plot(x,y[idx],color=color,**line_kwargs)

	ax.set(**axes_kwargs)
	ax.grid(ls='--')

	if return_fig:
		return fig

def SignificantParametersPlot(model,alpha=.05,figsize=(8,8),xlim=None,colors=color_scheme_2_1,lw=3,
	facecolor='lightgray',ax=None,title='',fontsize='x-large'):
	
	cmap=LinearSegmentedColormap.from_list('custom', colors, N=256)

	params=model._results.params[1:]
	error=model._results.bse[1:]
	pvalues=model._results.pvalues[1:]
	names=np.array(list(dict(model.params).keys()))[1:]
	params=params[pvalues<alpha]
	error=error[pvalues<alpha]
	names=names[pvalues<alpha]
	pvalues1=pvalues[pvalues<alpha]
	name_lengths=[len(name) for name in names]
	name_length_order=np.flip(np.argsort(name_lengths))

	return_fig=False
	if ax==None:
		fig,ax=plt.subplots(figsize=figsize)
		return_fig=True

	ax.barh(list(range(len(params))),params[name_length_order],xerr=error,
		ec=cmap(.99),ls='-',lw=lw,fc=cmap(0),height=.75,
		error_kw=dict(ecolor=cmap(.99),lw=lw,capsize=5,capthick=2))

	if title:
		ax.set_title(title,fontsize=fontsize)

	ax.set_facecolor(facecolor)
	ax.set_xlabel('Coefficient Value [-]',fontsize=fontsize)
	ax.set_ylabel('Coefficient',fontsize=fontsize)
	ax.set_yticks(list(range(len(names))))
	ax.set_yticklabels(names[name_length_order],fontsize=fontsize)
	if xlim != None:
		ax.set_xlim(xlim)
	ax.grid(linestyle='--')

	if return_fig:
		return fig

def HexPlot(data,figsize=(8,8),margin=.05,alpha=1,colors=color_scheme_2_1,ax=None,
	fontsize='medium',column=None,color_label='',fmt='{x:.0f}',title=''):
	
	formatter=mtick.StrMethodFormatter(fmt)

	cmap=LinearSegmentedColormap.from_list('custom', colors, N=256)

	minx=data.bounds['minx'].min()
	maxx=data.bounds['maxx'].max()
	miny=data.bounds['miny'].min()
	maxy=data.bounds['maxy'].max()

	return_fig=False
	if ax==None:
		fig,ax=plt.subplots(figsize=figsize)
		return_fig=True

	cs=data.plot(ax=ax,fc=colors[0],ec='k',alpha=alpha,cmap=cmap,column=column)

	if title:
		ax.set_title(title)

	ax.set_xlim([minx-(maxx-minx)*margin,maxx+(maxx-minx)*margin])
	ax.set_ylim([miny-(maxy-miny)*margin,maxy+(maxy-miny)*margin])
	ax.set_xticks([])
	ax.set_yticks([])
	# ax.spines['top'].set_visible(False)
	# ax.spines['right'].set_visible(False)
	# ax.spines['bottom'].set_visible(False)
	# ax.spines['left'].set_visible(False)

	if color_label:
		vmin=data[column].min()
		vmax=data[column].max()
		divider=make_axes_locatable(ax)
		sm=plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
		# if return_fig:
		cb=plt.colorbar(sm,ax=ax,format=formatter)
		cb.set_label(label=color_label,size=fontsize)
	# ax.set_aspect('equal','box')

	if return_fig:
		return fig

def StackedBarPlot(positions,values,figsize=(8,8),
	bar_labels=[],labels='',xlabel='',ylabel='',second_y=[],
	colors=color_scheme_2_1,lw=3,
	fontsize='medium',facecolor='whitesmoke',ax=None,width=.8,bottom=None,fmt='{x:.0f}',
	hatches=hatches_default):

	formatter=mtick.StrMethodFormatter(fmt)
	
	cmap=LinearSegmentedColormap.from_list('custom', colors, N=256)

	return_fig=False
	if ax==None:
		fig,ax=plt.subplots(figsize=figsize)
		return_fig=True

	if bottom is None:
		bottom=np.zeros(values.shape[1])

	handles=[]

	for idx,val in enumerate(values):
		axb=ax.bar(positions,val,color=cmap(idx/(len(values)+.001)),bottom=bottom,ec='k',
			width=width,hatch=hatches[idx])
		bottom+=val
		handles.append(axb)

	if xlabel:
		ax.set_xlabel(xlabel,fontsize=fontsize)
	if bar_labels:
		ax.set_xticks(positions,bar_labels)
	if ylabel:
		ax.set_ylabel(ylabel,fontsize=fontsize)
		ax.yaxis.set_major_formatter(formatter) 
	if second_y:
		ax1=ax.twinx()
		ax1.set_yticks(second_y[0],second_y[1],fontsize=fontsize)
		ax1.set_ylabel(second_y[2])

	if labels:
		box = ax.get_position()
		ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
		leg=ax.legend(handles[::-1],labels[::-1],ncol=1,
			fontsize=fontsize,loc='center left', bbox_to_anchor=(1, .5))
		
	ax.set_facecolor(facecolor)
	# ax.grid(ls='--')
	ax.tick_params(axis='both',which='major',labelsize=fontsize)


	if return_fig:
		return fig

def Histogram(values,figsize=(8,8),labels=[],colors=color_scheme_2_1,width=.8,lw=2,
	fontsize=12,ax=None,**axes_kwargs):
	# print(axes_kwargs)
	
	cmap=LinearSegmentedColormap.from_list('custom', colors, N=256)
	plt.rc('font', size=fontsize)

	return_fig=False
	if ax==None:
		fig,ax=plt.subplots(figsize=figsize)
		return_fig=True

	ax.bar(np.arange(0,len(values),1),values,color=cmap(0),ec='k',
		width=width,lw=lw)

	if labels:
		box = ax.get_position()
		ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
		ax.legend(labels,fontsize=fontsize,loc='center left', bbox_to_anchor=(1, 0.5))

	ax.set(**axes_kwargs)
	ax.grid(ls='--')

	if return_fig:
		return fig
	