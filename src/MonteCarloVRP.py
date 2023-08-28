import os
import sys
import time
import smopy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib.patches as ptch
import scipy.stats as st
from scipy.stats._continuous_distns import _distn_names
from tqdm import tqdm
from EVRP.MapBoxRouter import *

def GenerateRandomDiGraphAndLoads(n,distance_scale=1,loads_scale=1,directional_difference_scale=0):
	points=np.random.rand(n,2)*distance_scale
	index_from,index_to=np.meshgrid(range(n),range(n))
	di_graph=np.sqrt((points[index_from,0]-points[index_to,0])**2+
		(points[index_from,1]-points[index_to,1])**2)
	di_graph+=np.random.rand(n,n)*directional_difference_scale
	loads=np.random.rand(n)*loads_scale
	return points,di_graph,loads

def Greedy(di_graph,start):
	destinations=np.arange(0,di_graph.shape[0],1).astype(int)
	visited=[]
	current_location=start
	for i in range(di_graph.shape[0]):
		visited.append(current_location)
		available=np.setdiff1d(destinations,visited)
		if len(available)==0:
			break
		current_location=available[np.argmin(di_graph[current_location][available])]
	visited.append(0)
	return np.array(visited),TourDistance(di_graph,visited)

def TourDistance(di_graph,tour):
	location_from=tour[:-1]
	location_to=tour[1:]
	return di_graph[location_from,location_to].sum()

def TwoOpt(di_graph,tour):
	if len(tour)>4:
		# print('a')
		tour_start=tour[0]
		tour_end=tour[-1]
		tour=np.array(tour[1:-1])
		# print('d',tour,type(tour))
		for i in range(100):
			distance=TourDistance(di_graph,np.hstack((tour_start,tour,tour_end)))
			for idx1 in range(1,len(tour)-1):
				for idx2 in range(idx1+1,len(tour)):
					new_tour=np.hstack((tour_start,tour[:idx1],np.flip(tour[idx1:idx2+1]),
						tour[idx2+1:],tour_end)).astype(int)
					new_distance=TourDistance(di_graph,new_tour)
					if new_distance<distance:
						# print(idx1,idx2)
						tour=new_tour[1:-1]
						distance=new_distance
		# print('b',tour,type(tour))
		tour=np.hstack((tour_start,tour,tour_end))
		return tour,TourDistance(di_graph,tour)
	else:
		return tour,None

def Savings_Directional(di_graph,loads,capacity):
	n=di_graph.shape[0]
	cost_from,cost_to=np.meshgrid(di_graph[:,0],di_graph[0])
	savings=cost_from+cost_to-di_graph
	savings[range(n),range(n)]=0
	routes=[[i] for i in range(1,n)]
	route_loads=list(loads)
	index_from,index_to=np.meshgrid(list(range(n)),list(range(n)))
	indices=np.vstack((index_from.flatten(),index_to.flatten())).astype(int).T
	for i in range(100):
		if savings.sum()==0:
			break
		link_savings=savings.flatten()
		best_link_savings=indices[np.argmax(link_savings)]
		from_edge_index=[i for i in range(len(routes)) if best_link_savings[0] in {routes[i][0],routes[i][-1]}]
		to_edge_index=[i for i in range(len(routes)) if best_link_savings[1] in {routes[i][0],routes[i][-1]}]
		if bool(from_edge_index) & bool(to_edge_index):
			if from_edge_index[0] != to_edge_index[0]:
				if route_loads[from_edge_index[0]]+route_loads[to_edge_index[0]]<=capacity:
					routes[from_edge_index[0]].extend(routes[to_edge_index[0]])
					route_loads[from_edge_index[0]]+=route_loads[to_edge_index[0]]
					routes.remove(routes[to_edge_index[0]])
					route_loads.remove(route_loads[to_edge_index[0]])
		savings[best_link_savings[0],best_link_savings[1]]=0
		savings[best_link_savings[1],best_link_savings[0]]=0
	for route in routes:
		route.extend([0])
		route.insert(0,0)
	return routes

class DistributedRandomVariable():
	def __init__(self,dist='norm',loc=1,scale=.1,args=[]):
		self.distribution=getattr(st,dist)
		self.loc=loc
		self.scale=scale
		self.args=args

	def ReturnSample(self,dims):
		return self.distribution(loc=self.loc,scale=self.scale,*self.args).rvs(size=dims)

class StochasticEnergyConsumptionModel():
	def __init__(self,bins,distributions):
		self.bins=bins
		self.distributions=distributions

	def ReturnSample(self,speed,distance):
		pass

class RoadMap():
	def __init__(self,points=None,di_graph=None,durations_graph=None):
		self.points=points
		self.di_graph=di_graph
		self.durations_graph=durations_graph

	def MakeDiGraphFromPoints_MBR(self,points,filter_distance=500,home_index=0):
		if points is None:
			points=self.points
		n=points.shape[0]
		if filter_distance is not None:
			distances=haversine(points[home_index,0],points[home_index,1],
				points[:,0],points[:,1])
			distances[0]=filter_distance
			points=points[distances>=filter_distance]
		print(points.shape)
		n=points.shape[0]
		index_from,index_to=np.meshgrid(range(n),range(n))
		distances=np.zeros((n,n))
		durations=np.zeros((n,n))
		failures=0
		for i in tqdm(range(n)):
			for j in range(n):
				if i!=j:
					try:
						# print(i,j)
						route_summary=RouteSummary(points[i],points[j])
						# print(route_summary)
						distances[i,j]=route_summary['distances']
						durations[i,j]=route_summary['durations']
					except:
						failures+=1
		self.points=points
		self.di_graph=distances
		self.durations_graph=durations
		print(failures)

	def MakeDiGraphFromPoints(self,points=None,directional_difference_scale=0):
		if points is None:
			points=self.points
		n=points.shape[0]
		index_from,index_to=np.meshgrid(range(n),range(n))
		self.di_graph=np.sqrt((points[index_from,0]-points[index_to,0])**2+
			(points[index_from,1]-points[index_to,1])**2)
		self.di_graph+=np.random.rand(n,n)*directional_difference_scale

	def MakeRandom(self,n,distance_scale=1,directional_difference_scale=0):
		self.points=np.random.rand(n,2)*distance_scale
		self.MakeDiGraphFromPoints(self.points,directional_difference_scale)

def MergeItineraries(itineraries,keys):
	itinerary={key:itineraries[0].__dict__[key] for key in keys}
	for key in keys:
		itinerary[key]=np.concatenate([itin.__dict__[key] for itin in itineraries])
	itinerary['day_index']=np.floor((itinerary['datetimes']-
		itinerary['datetimes'][0].astype('datetime64[D]'))/np.timedelta64(1,'D')).astype(int)
	return itinerary

def PullDepotDayINRIX(depot_data,day_index):
	missions=MergeItineraries(depot_data['itineraries'],
		['datetimes','dest_lons','dest_lats','dwell_times'])
	return np.vstack((missions['dest_lons'][missions['day_index']==day_index],
			missions['dest_lats'][missions['day_index']==day_index])).T

def FilterTooClose(locations,cutoff):
	n=points.shape[0]
	index_from,index_to=np.meshgrid(range(n),range(n))
	distances=np.sqrt((points[index_from,0]-points[index_to,0])**2+
		(points[index_from,1]-points[index_to,1])**2)


class StochasticVRPSolver():
	def __init__(self,road_map,
			load_model=DistributedRandomVariable(loc=.1,scale=.001),
			load_capacity_model=DistributedRandomVariable(loc=1,scale=.1),
			volume_model=DistributedRandomVariable(loc=1,scale=.1),
			volume_capacity_model=DistributedRandomVariable(loc=1,scale=.1),
			energy_consumption_model=DistributedRandomVariable(loc=1,scale=.1),
			energy_capacity_model=DistributedRandomVariable(loc=1,scale=.1),
			debug_max_iterations=1000):
		self.road_map=road_map
		self.load_model=load_model
		self.load_capacity_model=load_capacity_model
		self.debug_max_iterations=debug_max_iterations

	def LoadsCutoff(self,loads_from,loads_to,capacities,cutoff_probability):
		loads=self.AddLoads(loads_from,loads_to)
		difference=capacities-loads
		return (difference<0).mean()<=cutoff_probability,loads

	def AddLoads(self,loads_from,loads_to):
		loads_from_mg,loads_to_mg=np.meshgrid(loads_from,loads_to)
		possibilities=(loads_from_mg+loads_to_mg).flatten()
		return np.random.choice(possibilities,len(loads_from))


	def SavingsRouter(self,n_samples,cutoff_probability):
		capacities=self.load_capacity_model.ReturnSample(n_samples)
		n=self.road_map.di_graph.shape[0]
		cost_from,cost_to=np.meshgrid(self.road_map.di_graph[:,0],self.road_map.di_graph[0])
		savings=cost_from+cost_to-self.road_map.di_graph
		savings[savings<0]=0
		savings[range(n),range(n)]=0
		routes=[[i] for i in range(1,n)]
		route_loads=self.load_model.ReturnSample((n-1,n_samples))
		index_from,index_to=np.meshgrid(list(range(n)),list(range(n)))
		indices=np.vstack((index_from.flatten(),index_to.flatten())).astype(int).T
		hit_max_iterations=True
		for i in range(self.debug_max_iterations):
			if i%10000==0:
				print(savings.sum())
			if savings.sum()==0:
				hit_max_iterations=False
				print('All route savings incorporated')
				break
			link_savings=savings.flatten()
			best_link_savings=indices[np.argmax(link_savings)]
			from_edge_index=[i for i in range(len(routes)) if best_link_savings[0] in {routes[i][0],routes[i][-1]}]
			to_edge_index=[i for i in range(len(routes)) if best_link_savings[1] in {routes[i][0],routes[i][-1]}]
			if bool(from_edge_index) & bool(to_edge_index):
				if from_edge_index[0] != to_edge_index[0]:
					managable_load,loads=self.LoadsCutoff(route_loads[from_edge_index[0]].copy(),
						route_loads[to_edge_index[0]].copy(),capacities,cutoff_probability)
					if managable_load:
						routes[from_edge_index[0]].extend(routes[to_edge_index[0]])
						route_loads[from_edge_index[0]]=loads
						routes.remove(routes[to_edge_index[0]])
						route_loads=np.delete(route_loads,to_edge_index[0],0)
			savings[best_link_savings[0],best_link_savings[1]]=0
			savings[best_link_savings[1],best_link_savings[0]]=0
		for route in routes:
			route.extend([0])
			route.insert(0,0)
		if hit_max_iterations:
			print('Max iterations hit before all route savings incorporated')
		return routes,route_loads,capacities,savings

	def TwoOpt(self,di_graph,tour):
		if len(tour)>4:
			for i in range(100):
				distance=TourDistance(di_graph,np.hstack((0,tour,0)))
				for idx1 in range(1,len(tour)-1):
					for idx2 in range(idx1+1,len(tour)):
						new_tour=np.hstack((0,tour[:idx1],np.flip(tour[idx1:idx2+1]),
							tour[idx2+1:],0)).astype(int)
						new_distance=TourDistance(di_graph,new_tour)
						if new_distance<distance:
							tour=new_tour[1:-1]
							distance=new_distance
			return tour,TourDistance(di_graph,tour)
		else:
			return tour,None

	def SavingsRouterTwoOpt(self,n_samples,cutoff_probability):
		capacities=self.load_capacity_model.ReturnSample(n_samples)
		n=self.road_map.di_graph.shape[0]
		cost_from,cost_to=np.meshgrid(self.road_map.di_graph[:,0],self.road_map.di_graph[0])
		savings=cost_from+cost_to-self.road_map.di_graph
		savings[savings<0]=0
		savings[range(n),range(n)]=0
		routes=[[i] for i in range(1,n)]
		route_loads=self.load_model.ReturnSample((n-1,n_samples))
		index_from,index_to=np.meshgrid(list(range(n)),list(range(n)))
		indices=np.vstack((index_from.flatten(),index_to.flatten())).astype(int).T
		hit_max_iterations=True
		for i in tqdm(range(self.debug_max_iterations)):
			if savings.sum()==0:
				hit_max_iterations=False
				print('All route savings incorporated')
				break
			link_savings=savings.flatten()
			best_link_savings=indices[np.argmax(link_savings)]
			from_edge_index=[i for i in range(len(routes)) if best_link_savings[0] in {routes[i][0],routes[i][-1]}]
			to_edge_index=[i for i in range(len(routes)) if best_link_savings[1] in {routes[i][0],routes[i][-1]}]
			if bool(from_edge_index) & bool(to_edge_index):
				if from_edge_index[0] != to_edge_index[0]:
					managable_load,loads=self.LoadsCutoff(route_loads[from_edge_index[0]].copy(),
						route_loads[to_edge_index[0]].copy(),capacities,cutoff_probability)
					if managable_load:
						routes[from_edge_index[0]].extend(routes[to_edge_index[0]])
						route_loads[from_edge_index[0]]=loads
						route_temp,_=self.TwoOpt(self.road_map.di_graph,
							routes[from_edge_index[0]])
						routes[from_edge_index[0]]=list(route_temp)
						routes.remove(routes[to_edge_index[0]])
						route_loads=np.delete(route_loads,to_edge_index[0],0)
			savings[best_link_savings[0],best_link_savings[1]]=0
			savings[best_link_savings[1],best_link_savings[0]]=0
		for route in routes:
			route.extend([0])
			route.insert(0,0)
		if hit_max_iterations:
			print('Max iterations hit before all route savings incorporated')
		return routes,route_loads,capacities,savings

def haversine(lon1, lat1, lon2, lat2):
	r = 6372800. #[m]

	dLat = np.radians(lat2 - lat1)
	dLon = np.radians(lon2 - lon1)
	lat1 = np.radians(lat1)
	lat2 = np.radians(lat2)

	a = np.sin(dLat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dLon/2)**2
	c = 2*np.arcsin(np.sqrt(a))
	return c * r

def ToursPlot(svrp,tours):
	fig=plt.figure(figsize=(8,8))
	for idx,tour in enumerate(tours):
		line_tour=svrp.road_map.points[tour]
		plt.plot(line_tour[:,0],line_tour[:,1],label='Tour {}'.format(idx))
	plt.scatter(svrp.road_map.points[:,0],svrp.road_map.points[:,1],
		color='k',label='Destinations',zorder=idx+1)
	plt.scatter(svrp.road_map.points[0,0],svrp.road_map.points[0,1],
		s=100,color='r',label='Depot',zorder=idx+2)
	plt.xlabel('Longitude [deg]')
	plt.ylabel('Latitude [deg]')
	plt.legend()
	plt.grid(linestyle='--')
	plt.savefig('out2.png', transparent=True)
	return fig


def SmopyPlot(svrp,tours):
	linepath=np.array([[0,0]])
	line_tours=[]
	for tour in tours:
		line_tour=svrp.road_map.points[tour]
		line_tours.append(line_tour)
		linepath=np.vstack((linepath,line_tour))
	linepath=linepath[1:]
	bounds=(np.min(linepath[:,1]),np.min(linepath[:,0]),
		np.max(linepath[:,1]),np.max(linepath[:,0]))
	print(bounds)

	fig,ax=plt.subplots(figsize=(8,8))
	m = smopy.Map(bounds, z=15, margin=.1)
	
	ax = m.show_mpl(ax=ax, cmap='gray')
	# print(m.xmin,m.ymin,m.box_tile, m.z, m.tileserver, m.tilesize, m.maxtiles,m.img.__dict__)
	# m.save_png('out.png')
	# jet = plt.get_cmap('cool')
	# # cNorm  = colors.Normalize(vmin=min([beta.min(),delta.min()]),
	# # 	vmax=max([beta.max(),delta.max()]))
	# # scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet) 
	# colors_list=jet(np.arange(0,len(line_tours))/len(line_tours))
	# for idx,tour in enumerate(line_tours):
	# 	x, y = m.to_pixels(tour[:, 1], tour[:, 0])
	# 	ax.plot(x, y, color=colors_list[idx], lw=2,label='Vehicle {}'.format(idx+1))
	# x, y = m.to_pixels(svrp.road_map.points[:,1], svrp.road_map.points[:,0])
	# ax.scatter(x,y,
	# 	s=50,color='b',label='Destinations',zorder=idx+1)
	# ax.scatter(x[0],y[0],
	# 	s=100,color='r',label='Depot',zorder=idx+2)
	# lon_ticks=np.arange(bounds[1],bounds[3],.5)
	# lat_ticks=np.arange(tour[:,1].min(),tour[:,1].max(),.1)

	# pix_x_ticks,pix_y_ticks=m.to_pixels(lat_ticks,lon_ticks)
	# print(pix_x_ticks,lon_ticks)
	# ax.set_xticks(pix_x_ticks,lon_ticks)
	# ax.set_xlabel('Longitude [deg]')
	# ax.set_ylabel('Latitude [deg]')
	# ax.legend()
	# ax.grid(linestyle='--')