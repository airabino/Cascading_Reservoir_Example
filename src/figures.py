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
import scipy.stats as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap,to_hex
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm
from shapely.ops import cascaded_union
from itertools import combinations
from shapely.geometry import Point,Polygon,MultiPolygon
from scipy.stats import t
from scipy.stats._continuous_distns import _distn_names

#Defining some 5 pronged color schemes

color_scheme_5_0=["#e7b7a5","#da9b83","#b1cdda","#71909e","#325666"]

#Defining some 4 pronged color schemes (source: http://vrl.cs.brown.edu/color)
color_scheme_4_0=["#8de4d3", "#0e503e", "#43e26d", "#2da0a1"]
color_scheme_4_1=["#069668", "#49edc9", "#2d595a", "#8dd2d8"]
color_scheme_4_2=["#f2606b", "#ffdf79", "#c6e2b1", "#509bcf"] #INCOSE IS2023

#Defining some 3 pronged color schemes (source: http://vrl.cs.brown.edu/color)
color_scheme_3_0=["#72e5ef", "#1c5b5a", "#2da0a1"]
color_scheme_3_1=["#256676", "#72b6bc", "#1eefc9"]
color_scheme_3_2=['#40655e', '#a2e0dd', '#31d0a5']
color_scheme_3_3=["#f2606b", "#c6e2b1", "#509bcf"] #INCOSE IS2023 minus yellow

#Defining some 2 pronged color schemes (source: http://vrl.cs.brown.edu/color)
color_scheme_2_0=["#21f0b6", "#2a6866"]
color_scheme_2_1=["#72e5ef", "#3a427d"]
color_scheme_2_2=["#1e4d2b", "#c8c372"] #CSU green/gold

#Distributions to try (scipy.stats continuous distributions)
dist_names=['alpha','beta','gamma','logistic','norm','lognorm']
dist_labels=['Alpha','Beta','Gamma','Logistic','Normal','Log Normal']

def SelectionPlot(selected,background,figsize=(8,8),margin=.05,alpha=1,colors=color_scheme_2_1,ax=None,
	fontsize='medium'):
	
	cmap=LinearSegmentedColormap.from_list('custom', colors, N=256)

	minx=selected.bounds['minx'].min()
	maxx=selected.bounds['maxx'].max()
	miny=selected.bounds['miny'].min()
	maxy=selected.bounds['maxy'].max()

	return_fig=False
	if ax==None:
		fig,ax=plt.subplots(figsize=figsize)
		return_fig=True

	# ax.set_prop_cycle(color=colors)
	# print(to_hex(cmap(.33)))

	background.plot(ax=ax,fc=to_hex(cmap(0)),ec='k',alpha=alpha)
	selected.plot(ax=ax,fc=to_hex(cmap(.99)),ec='k',alpha=alpha)
	ax.set_xlim([minx-(maxx-minx)*margin,maxx+(maxx-minx)*margin])
	ax.set_ylim([miny-(maxy-miny)*margin,maxy+(maxy-miny)*margin])
	ax.set_xlabel('Longitude [deg]',fontsize=fontsize)
	ax.set_ylabel('Latitude [deg]',fontsize=fontsize)
	# ax.set_aspect('equal','box')

	if return_fig:
		return fig

def TractsHexComparisonPlot(tracts,h3_hex,background,figsize=(12,6),margin=.05,alpha=1,
	colors=color_scheme_2_1,fontsize='medium'):
	
	fig,ax=plt.subplots(1,2,figsize=figsize)
	SelectionPlot(tracts,background,ax=ax[0],margin=margin,alpha=alpha,fontsize=fontsize,
		colors=colors)
	SelectionPlot(h3_hex,background,ax=ax[1],margin=margin,alpha=alpha,fontsize=fontsize,
		colors=colors)

	return fig

def TractsHexAreaHistogram(tracts,h3_hex,figsize=(8,8),cutoff=2e4,bins=100,colors=color_scheme_2_1):

	fig,ax=plt.subplots(figsize=figsize)

	ax.set_facecolor('lightgray')

	out=ax.hist(tracts.to_crs(2163).area/1e3,bins=bins,rwidth=.9,color=colors[1],ec='k',label='Census Tracts')
	ax.plot([h3_hex.to_crs(2163).area.mean()/1e3]*2,[out[0].min(),out[0].max()],lw=7,color='k')
	ax.plot([h3_hex.to_crs(2163).area.mean()/1e3]*2,[out[0].min(),out[0].max()],lw=5,color=colors[0],
		label='Hex Cells')
	ax.grid(ls='--')
	ax.set_xlim([0,cutoff])
	ax.legend()
	ax.set_xlabel('Geometry Area [km^2]')
	ax.set_ylabel('Bin Size [-]')

	return fig

def DataColumnPlot(selected,background,figsize=(8,8),margin=.05,alpha=1,colors=color_scheme_2_1,ax=None,
	column=None,color_axis_label=None,fontsize='medium',scale=[]):
	
	cmap=LinearSegmentedColormap.from_list('custom', colors, N=256)

	if scale:
		vmin=scale[0]
		vmax=scale[1]
	else:
		vmin=selected[column].min()
		vmax=selected[column].max()

	minx=selected.bounds['minx'].min()
	maxx=selected.bounds['maxx'].max()
	miny=selected.bounds['miny'].min()
	maxy=selected.bounds['maxy'].max()

	return_fig=False
	if ax==None:
		fig,ax=plt.subplots(figsize=figsize)
		return_fig=True

	ax.set_prop_cycle(color=colors)

	background.plot(ax=ax,fc='lightgray',ec='k',alpha=alpha)
	im=selected.plot(ax=ax,column=column,ec='k',alpha=alpha,cmap=cmap,vmin=vmin,vmax=vmax)
	ax.set_xlim([minx-(maxx-minx)*margin,maxx+(maxx-minx)*margin])
	ax.set_ylim([miny-(maxy-miny)*margin,maxy+(maxy-miny)*margin])
	ax.set_xlabel('Longitude [deg]',fontsize=fontsize)
	ax.set_ylabel('Latitude [deg]',fontsize=fontsize)

	divider=make_axes_locatable(ax)
	# cax=divider.append_axes('bottom', size=(.1), pad=.5)
	sm=plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
	# sm._A=[]
	cbr=plt.colorbar(sm, ax=ax, orientation='vertical')
	cbr.ax.set_ylabel(color_axis_label,
		labelpad=10,fontsize=fontsize)
	# ax.set_aspect('equal','box')

	if return_fig:
		return fig

def DataColumnComparisonPlot(tracts,h3_hex,background,horizontal=True,figsize=(12,6),
	margin=.05,alpha=1,colors=color_scheme_2_1,column=None,color_axis_label=None):
	
	if horizontal:
		fig,ax=plt.subplots(1,2,figsize=figsize)
	else:
		fig,ax=plt.subplots(2,1,figsize=figsize)
	DataColumnPlot(tracts,background,ax=ax[0],margin=margin,alpha=alpha,
		colors=colors,column=column,color_axis_label=color_axis_label)
	DataColumnPlot(h3_hex,background,ax=ax[1],margin=margin,alpha=alpha,
		colors=colors,column=column,color_axis_label=color_axis_label)

	return fig

def HistogramComparisonPlot(tracts,h3_hex,horizontal=True,figsize=(12,6),data_label=None,
	colors=color_scheme_2_1,column=None,cutoff=None,bins=100,
	dist_names=dist_names,dist_labels=dist_labels):
	
	if data_label == None:
		data_label=column
	
	if horizontal:
		fig,ax=plt.subplots(1,2,figsize=figsize)
	else:
		fig,ax=plt.subplots(2,1,figsize=figsize)
	HistogramDist(tracts[column],ax=ax[0],cutoff=cutoff,bins=bins,colors=colors,data_label=data_label,
		dist_names=dist_names,dist_labels=dist_labels)
	HistogramDist(h3_hex[column],ax=ax[1],cutoff=cutoff,bins=bins,colors=colors,data_label=data_label,
		dist_names=dist_names,dist_labels=dist_labels)

	return fig

def HistogramDist(data,figsize=(8,8),column=None,cutoff=None,bins=100,colors=color_scheme_2_1,ax=None,
	data_label=None,dist_names=dist_names,dist_labels=dist_labels,fontsize='medium'):

	if column == None:
		data=data.to_numpy()
	else:
		data=data[column].to_numpy()
	data=data[~np.isnan(data)]

	return_fig=False
	if ax==None:
		fig,ax=plt.subplots(figsize=figsize)
		return_fig=True

	dist_name,dist,params,_=FitBestDist(data,bins=bins,dist_names=dist_names,dist_labels=dist_labels)

	densities,bin_edges=np.histogram(data,bins)
	integral=(densities*np.diff(bin_edges)).sum()
	bin_edges=(bin_edges+np.roll(bin_edges,-1))[:-1]/2.0
	y=dist.pdf(bin_edges,loc=params[-2],scale=params[-1],*params[:-2])*integral
	rmse=np.sqrt(((y-densities)**2).sum()/len(y))
	x=np.linspace(bin_edges.min(),bin_edges.max(),1000)
	y=dist.pdf(x,loc=params[-2],scale=params[-1],*params[:-2])*integral

	ax.set_facecolor('lightgray')
	
	out=ax.hist(data,bins=bins,rwidth=.9,color=colors[1],ec='k',label='Data')
	ax.plot(x,y,lw=7,color='k')
	ax.plot(x,y,lw=5,color=colors[0],label='Best-Fit Distribution:\n{}, RMSE={:.4f}'.format(dist_name,rmse))
	ax.grid(ls='--')
	if cutoff != None:
		ax.set_xlim([0,cutoff])
	ax.legend(fontsize=fontsize)
	ax.set_xlabel(data_label,fontsize=fontsize)
	ax.set_ylabel('Bin Size [-]',fontsize=fontsize)

	if return_fig:
		return fig

def DistributionComparisonPlot(data_list,figsize=(8,8),column=None,cutoff=None,bins=100,
	colors=color_scheme_2_1,ax=None,data_label=None,dist_names=dist_names,
	dist_labels=dist_labels,facecolor='lightgray',lw=5,xlim=None,xlabel='SIC [min/km]'):

	cmap=LinearSegmentedColormap.from_list('custom', colors, N=256)

	return_fig=False
	if ax==None:
		fig,ax=plt.subplots(figsize=figsize)
		return_fig=True

	for idx,data in enumerate(data_list):

		if column == None:
			data=data.to_numpy()
		else:
			data=data[column].to_numpy()
		data=data[~np.isnan(data)]

		dist_name,dist,params,_=FitBestDist(data,bins=bins,dist_names=['lognorm'],dist_labels=['Log Normal'])
		# print(dist_name)

		densities,bin_edges=np.histogram(data,bins)
		integral=(densities*np.diff(bin_edges)).sum()
		bin_edges=(bin_edges+np.roll(bin_edges,-1))[:-1]/2.0
		y=dist.pdf(bin_edges,loc=params[-2],scale=params[-1],*params[:-2])*integral
		rmse=np.sqrt(((y-densities)**2).sum()/len(y))
		if xlim == None:
			x=np.linspace(bin_edges.min(),bin_edges.max(),1000)
		else:
			x=np.linspace(xlim[0],xlim[1],1000)
		# y=dist.pdf(x,loc=params[-2],scale=params[-1],*params[:-2])/densities.sum()
		y=dist.pdf(x,loc=params[-2],scale=params[-1],*params[:-2])

		
		ax.plot(x,y,lw=lw+2,color='k')
		ax.plot(x,y,lw=lw,color=cmap(.99*(idx/(len(data_list)-1))),label=data_label[idx])
	
	ax.grid(ls='--')
	if cutoff != None:
		ax.set_xlim([0,cutoff])
	ax.legend()
	ax.set_xlabel(xlabel)
	ax.set_ylabel('Probability Density Function [-]')
	ax.set_facecolor(facecolor)


	if return_fig:
		return fig

def Correlation(x,y):
	n=len(x)
	return (n*(x*y).sum()-x.sum()*y.sum())/np.sqrt((n*(x**2).sum()-x.sum()**2)*(n*(y**2).sum()-y.sum()**2))

def Determination(x,y):
	return Correlation(x,y)**2

def FitBestDist(data,bins=200,dist_names=dist_names,dist_labels=dist_labels):

	data=data[~np.isnan(data)]

	densities,bin_edges=np.histogram(data,bins,density=True)
	bin_edges=(bin_edges+np.roll(bin_edges,-1))[:-1]/2.0

	rmse=np.empty(len(dist_names))
	params_list=[None]*len(dist_names)

	for idx,dist_name in enumerate(dist_names):

		dist=getattr(st,dist_name)

		try:

			params=dist.fit(data)
			params_list[idx]=params

			arg=params[:-2]
			loc=params[-2]
			scale=params[-1]
			y=dist.pdf(bin_edges,loc=loc,scale=scale,*arg)
			# print(dist_name)
			# print(y)

			rmse[idx]=np.sqrt(((y-densities)**2).sum()/len(y))
			# print(rmse[idx])

		except Exception as e:

			rmse[idx]=sys.maxsize

	rmse[np.isnan(rmse)]=sys.maxsize
	best_dist_index=np.argmin(rmse)

	return (dist_labels[best_dist_index],getattr(st,dist_names[best_dist_index]),
		params_list[best_dist_index],np.min(rmse))

def EVTracePlot(bev,optimal_control,soc_trace,max_dwells_disp=100,figsize=(8,8),
	colors=color_scheme_3_1,facecolor='lightgray'):
	
	cmap=LinearSegmentedColormap.from_list('custom', colors, N=256)

	selection=[0,len(soc_trace)-1]
	if selection[1]>max_dwells_disp:
		selection[1]=max_dwells_disp

	indices=np.arange(selection[0],selection[1],1)
	indices1=np.arange(selection[0],selection[1]+1,1)
	isHomeCharge=(optimal_control[0][indices]>0)&(bev.isHome[indices])
	isWorkCharge=(optimal_control[0][indices]>0)&(bev.isWork[indices])
	isDestCharge=(optimal_control[0][indices]>0)&(bev.isOther[indices])
	isEnRouteCharge=optimal_control[1][indices]>0

	fig,ax=plt.subplots(3,1,figsize=figsize)

	for axis in ax:
		# axis.set_prop_cycle(color=colors)
		axis.set_facecolor(facecolor)

	ax[0].plot(soc_trace[indices1],linewidth=4,color='k')
	ax[0].plot(soc_trace[indices1],linewidth=3,color=cmap(0))
	ax[0].plot([0,len(bev.Parks[indices])],[0,0],linestyle='--',color='k')
	ax[0].plot([0,len(bev.Parks[indices])],[1,1],linestyle='--',color='k')
	ax[0].grid()
	ax[0].set_ylabel('SOC [dim]')

	ax[1].bar(indices,optimal_control[0][indices]*isDestCharge/3600,ec='k',color=cmap(0))
	ax[1].bar(indices,optimal_control[0][indices]*isHomeCharge/3600,ec='k',color=cmap(.33))
	ax[1].bar(indices,optimal_control[0][indices]*isWorkCharge/3600,ec='k',color=cmap(.66))
	ax[1].bar(indices,optimal_control[1][indices]*isEnRouteCharge/3600,ec='k',color=cmap(.99))
	ax[1].legend(['Destination','Home','Work','En Route'])
	ax[1].grid()
	ax[1].set_ylabel('Energizing Time [h]')

	ax[2].bar(np.arange(0,len(bev.Trip_Distances[indices]),1),bev.Trip_Distances[indices]/1000,
		ec='k',color=cmap(0))
	ax[2].grid()
	ax[2].set_xlabel('Trip/Park Event')
	ax[2].set_ylabel('Trip Distance [km]')

	return fig

def SignificantParametersPlot(model,alpha=.05,figsize=(8,8),xlim=None,colors=color_scheme_2_1,lw=3,
	facecolor='lightgray'):

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

	fig,ax=plt.subplots(figsize=figsize)

	plt.barh(list(range(len(params))),params[name_length_order],xerr=error,
		ec=colors[1],ls='-',lw=lw,fc=colors[0],height=.75,
		error_kw=dict(ecolor=colors[1],lw=lw,capsize=5,capthick=2))

	ax.set_facecolor(facecolor)
	ax.set_xlabel('Coefficient Value [-]',fontsize='x-large')
	ax.set_ylabel('Coefficient',fontsize='x-large')
	ax.set_yticks(list(range(len(names))))
	ax.set_yticklabels(names[name_length_order])
	if xlim != None:
		ax.set_xlim(xlim)
	ax.grid(linestyle='--')

	return fig

def SignificantParametersComparisonPlot(model1,model2,model3,alpha=.05,figsize=(8,8),xlim=None,
	colors=color_scheme_2_1,lw=3,facecolor='lightgray'):
	
	cmap=LinearSegmentedColormap.from_list('custom', colors, N=256)

	params=model1._results.params[1:]
	error=model1._results.bse[1:]
	pvalues=model1._results.pvalues[1:]
	names=np.array(list(dict(model1.params).keys()))[1:]
	names=np.array([name.replace('DCFCR','ERCR') for name in names])
	names=np.array([name.replace('DCFCP','ERCP') for name in names])
	params=params[pvalues<alpha]
	error=error[pvalues<alpha]
	names=names[pvalues<alpha]
	pvalues1=pvalues[pvalues<alpha]
	params1=model2._results.params[1:][pvalues<alpha]
	params2=model3._results.params[1:][pvalues<alpha]
	name_lengths=[len(name) for name in names]
	name_length_order=np.argsort(name_lengths)

	fig,ax=plt.subplots(figsize=figsize)

	plt.bar(np.arange(0,len(names),1)-.25,params[name_length_order],width=.2,ls='-',lw=lw,
		fc=cmap(0),ec=(0,0,0,1))
	plt.bar(np.arange(0,len(names),1),params1[name_length_order],width=.2,ls='-',lw=lw,
		fc=cmap(.5),ec=(0,0,0,1))
	plt.bar(np.arange(0,len(names),1)+.25,params2[name_length_order],width=.2,ls='-',lw=lw,
		fc=cmap(.99),ec=(0,0,0,1))

	ax.set_xlabel('Coefficient',fontsize='x-large')
	ax.set_ylabel('Beta [dim]',fontsize='x-large')
	ax.set_xticks(list(range(len(names))))
	ax.set_xticklabels(names[name_length_order],rotation='vertical')
	ax.grid(linestyle='--')
	ax.legend(['National','Colorado','Denver MSA'])

	return fig

def DataScatterPlot(selected,background,resources_lons,resources_lats,margin=.05,figsize=(8,8),
	colors=color_scheme_2_1,data_label='',marker_size=30,ax=None,fontsize='large'):
	
	return_fig=False
	if ax==None:
		fig,ax=plt.subplots(figsize=figsize)
		return_fig=True
	
	minx=selected.bounds['minx'].min()
	maxx=selected.bounds['maxx'].max()
	miny=selected.bounds['miny'].min()
	maxy=selected.bounds['maxy'].max()

	background.plot(ax=ax,facecolor='lightgray',edgecolor='k')
	selected.plot(ax=ax,facecolor=colors[1],edgecolor='k')
	ax.scatter(resources_lons,resources_lats,s=marker_size,c=colors[0],label=data_label,
		edgecolor='k')
	
	ax.set_xlim([minx-(maxx-minx)*margin,maxx+(maxx-minx)*margin])
	ax.set_ylim([miny-(maxy-miny)*margin,maxy+(maxy-miny)*margin])
	ax.set_xlabel('Longitude [deg]',fontsize=fontsize)
	ax.set_ylabel('Latitude [deg]',fontsize=fontsize)
	if data_label:
		ax.legend(fontsize=fontsize,edgecolor='k',facecolor='k',labelcolor='w')

	if return_fig:
		return fig

def ComparisonDataScatterPlot(gdf1,gdf2,background,
	data_1_lons,data_1_lats,data_2_lons,data_2_lats,
	margin=.05,figsize=(12,6),colors=color_scheme_2_1,horizontal=True,
	data_1_label=None,data_2_label=None,
	marker_size=30,fontsize='medium'):
	
	if horizontal:
		fig,ax=plt.subplots(1,2,figsize=figsize)
	else:
		fig,ax=plt.subplots(2,1,figsize=figsize)
	DataScatterPlot(gdf1,background,data_1_lons,data_1_lats,
		ax=ax[0],colors=colors,data_label=data_1_label,
		marker_size=marker_size,margin=margin,fontsize=fontsize)
	DataScatterPlot(gdf2,background,data_2_lons,data_2_lats,
		ax=ax[1],colors=colors,data_label=data_2_label,
		marker_size=marker_size,margin=margin,fontsize=fontsize)

	return fig

def DemographicCorrelationPlot(sic,demographics,figsize=(8,8),xlim=None,colors=color_scheme_2_1,lw=3,
	facecolor='lightgray',marker_size=30,data_label=None,fontsize='large'):
	
	nans=np.isnan(sic)|np.isnan(demographics)
	sic=sic[~nans]
	demographics=demographics[~nans]

	fig,ax=plt.subplots(figsize=figsize)

	ax.scatter(demographics,sic,s=marker_size,c=colors[1],label='Data',ec='k')
	x=np.unique(demographics)
	y=np.poly1d(np.polyfit(demographics, sic, 1))(np.unique(demographics))
	ax.plot(x,y,lw=lw+2,color='k')
	ax.plot(x,y,lw=lw,
		label='Best Fit Line (R^2={:.4f})'.format(Correlation(demographics,sic)**2),color=colors[0])
	ax.set_xlabel(data_label,fontsize=fontsize)
	ax.set_ylabel('SIC [min/km]',fontsize=fontsize)
	ax.grid(ls='--')
	ax.legend(fontsize=fontsize)
	ax.set_facecolor(facecolor)

	return fig