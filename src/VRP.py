import sys
import time
import numpy as np
import numpy.random as rand
import pandas as pd
import EVRP.CompressedPickle as cpkl
import matplotlib.pyplot as plt
from EVRP import LEMVehicleModel as lvm
from EVRP import MapBoxRouter as mbr
from tqdm import tqdm

"""
Simulates symmetric replacement - vehicle itineraries are repeated whole by the new vehicle.
"""
def SymmetricSim(itineraries,sim_vehicles,day):
	out=[None]*(len(itineraries)*len(sim_vehicles))
	idx=0
	for itinerary_idx,itinerary in enumerate(itineraries):
		for vehicle_idx,veh in enumerate(sim_vehicles):
			out[idx]=veh.SimulateItinerary(itinerary,day)
			out[idx]['itinerary_idx']=itinerary_idx
			out[idx]['vehicle_idx']=vehicle_idx
			idx+=1
	return out

def LEMVehicle(data_file,n_bins=5,battery_capacity=150*1000*3600):
	lmd=pd.read_csv(data_file)
	speeds=lmd['GPS_Speed']
	powers=lmd['battery_power_w']
	is_not_charging=lmd['evse_state']=='Not connected'
	speeds_nc=speeds[is_not_charging]
	powers_nc=powers[is_not_charging]
	bins=np.linspace(100/n_bins,100,n_bins)*.44704
	powers,_=FilterOutliersPower(speeds_nc,powers_nc,bins)
	return EnergyConsumptionModel([bins,powers],battery_capacity)

def FilterOutliersPower(speed,power,bins):
	bin_indices=np.digitize(speed,bins)
	power_bins=[None]*len(bins)
	mean_powers=np.empty(len(bins))
	std_powers=np.empty(len(bins))
	for i in range(len(bins)):
		power_bins[i]=power[bin_indices==i]
	for i in range(len(bins)):
		for j in range(100):
			bin_size=power_bins[i].shape[0]
			mean_powers[i]=power_bins[i].mean()
			std_powers[i]=power_bins[i].std()
			power_bins[i]=power_bins[i][((power_bins[i]>=mean_powers[i]-3*std_powers[i])
				&(power_bins[i]<=mean_powers[i]+3*std_powers[i]))]
			if (bin_size-power_bins[i].shape[0])/bin_size<=.01:
				break
	return mean_powers,std_powers

class EnergyConsumptionModel():
	def __init__(self,bin_powers,energy_capacity):
		self.bins=bin_powers[0]
		self.powers=bin_powers[1]
		self.energy_capacity=energy_capacity

	def DeltaSOC_vec(self,route):
		return 1-np.cumsum(self.EnergyConsumed_vec(route))/self.energy_capacity	

	def EnergyConsumed_vec(self,route):
		powers=np.interp(route['speeds'],self.bins,self.powers)
		energy_consumed=route['durations']*powers
		return energy_consumed

	def DeltaSOC(self,route):
		return self.EnergyConsumed(route)/self.energy_capacity

	def EnergyConsumed(self,route):
		powers=np.interp(route['speeds'],self.bins,self.powers)
		energy_consumed=route['durations']*powers
		return energy_consumed.sum()	

class CC_CV_Charger():
	def __init__(self,power,completion=.99):
		self.power=power
		self.completion=completion

	def Charge(self,soc,time,battery_capacity):
		#Calcualting the SOC gained from a charging event of duration td_charge
		#The CC-CV curve is an 80/20 relationship where the charging is linear for the first 80%
		#and tails off for the last 20% approaching 100% SOC at t=infiniti
		delta_soc=self.completion-soc
		if delta_soc>.2:
			delta_soc_20=.2
			delta_soc_80=delta_soc-.2
		else:
			delta_soc_20=delta_soc
			delta_soc_80=0
		charge_time_20=(-np.log(1-delta_soc_20/.2000001)/(self.power/battery_capacity/.2))
		charge_time_80=(delta_soc_80*battery_capacity/self.power)
		return self.completion,time+charge_time_80+charge_time_20

class ICV_ReFuel():
	def __init__(self,power,completion=.99):
		self.power=power
		self.completion=completion

	def Charge(self,soc,time,tank_capacity):
		delta_soc=self.completion-soc
		charge_time=(delta_soc*tank_capacity/self.power)
		return self.completion,time+charge_time

def MergeItineraries(itineraries,keys):
	itinerary={key:itineraries[0].__dict__[key] for key in keys}
	for key in keys:
		itinerary[key]=np.concatenate([itin.__dict__[key] for itin in itineraries])
	itinerary['day_index']=np.floor((itinerary['datetimes']-
		itinerary['datetimes'][0].astype('datetime64[D]'))/np.timedelta64(1,'D')).astype(int)
	return itinerary

def MakeSimDepotINRIX(depot_data,vehicles,max_number_vehicles=None,
		max_load_vehicles=None):
	missions_data=MergeItineraries(depot_data['itineraries'],
		['datetimes','dest_lons','dest_lats','dwell_times'])
	# print(missions_data['dest_lats'].shape)
	# print(haversine(depot_data['lons'].mean(),depot_data['lats'].mean(),
	# 	missions_data['dest_lons'],missions_data['dest_lats'])<100)
	noise_location_indices=haversine(depot_data['lons'].mean(),depot_data['lats'].mean(),
		missions_data['dest_lons'],missions_data['dest_lats'])<=100
	# print(noise_location_indices.shape,noise_location_indices.sum())
	for key in missions_data.keys():
		missions_data[key]=missions_data[key][~noise_location_indices]
	sim_depot=SimDepot([depot_data['lons'].mean(),depot_data['lats'].mean()],
		missions_data,vehicles,max_number_vehicles,max_load_vehicles)
	return sim_depot

"""
SimDepot is a class which evaluates the VRP for a depot on a daily basis. the depot contains
a collection of SimVehicle objects defined in the input vehicles. The method is for the depot
vehicles to solve the TSP individually until all missions are fulfilled or until all vehicles
are used. Missions are defined by a location and a dwell time but it is assumed that missions
can be fulfilled at any time of day with equal validity.
"""
class SimDepot():
	def __init__(self,location,missions,vehicles,max_number_vehicles=None,max_load_vehicles=None,
		debug_max_iterations=100):
		self.location=location
		print(self.location[1],self.location[0])
		self.missions=missions
		self.sim_vehicles=vehicles
		for vehicle in self.sim_vehicles:
			vehicle.home_location=self.location
		self.debug_max_iterations=debug_max_iterations
		if max_number_vehicles==None:
			self.max_iterations=[self.debug_max_iterations]*len(self.sim_vehicles)
		else:
			self.max_iterations=max_number_vehicles
		if max_load_vehicles==None:
			self.max_loads=np.ones(len(self.sim_vehicles))/len(self.sim_vehicles)
		else:
			self.max_loads=max_load_vehicles

	def Simulate_Day(self,day_index,maximum_time):
		locations=np.vstack((self.missions['dest_lons'][self.missions['day_index']==day_index],
			self.missions['dest_lats'][self.missions['day_index']==day_index])).T
		move_to_next=locations.shape[0]-np.cumsum(self.max_loads)*locations.shape[0]
		# print(move_to_next,self.max_loads,np.cumsum(self.max_loads))
		dwells=self.missions['dwell_times'][self.missions['day_index']==day_index]
		# dwells[dwells>=3600]=3600
		#Looping through vehicle types in the order provided
		itineraries=[]
		for idx,vehicle in enumerate(self.sim_vehicles):
			vehicle_itineraries=[]
			n_locations=locations.shape[0]
			for i in range(self.max_iterations[idx]):
				print(i,n_locations,idx)
				itinerary,locations,dwells=vehicle.Simulate(locations,dwells,maximum_time)
				if locations.shape[0]==n_locations:
					#Remaining locations cannot be reached by this depot's EVs
					vehicle_itineraries.append(itinerary)
					break
				elif locations.shape[0]<=move_to_next[idx]:
					#This vehicle type filled its quota
					vehicle_itineraries.append(itinerary)
					break
				else:
					vehicle_itineraries.append(itinerary)
					n_locations=locations.shape[0]
				# break
			itineraries.append(vehicle_itineraries)
			# break
		return itineraries,locations,dwells

"""
SimVehicle is a class which simulates the TSP for a vehicle. The vehicle will attempt to
serve as many locations as possible during its allowable one-daytime limit while also managing
its SOC via a one-step look-ahead algorithm that guarantees that it will always have enough
charge to make it back to home basewhere it can charge (or re-fuel). The SimVehicle class
handles the TSP with the class method Simulate which takes an iterable of lat-lons and a maximum
time at minimum and outputs a dict with information about the daily itinerary.
"""
class SimVehicle():
	def __init__(self,energy_consumption_model,charging_model,home_location=None,
			maximum_soc=.99,minimum_soc=.2,routing_function=mbr.Route,
			debug_max_iterations=100):
		self.energy_consumption_model=energy_consumption_model
		self.charging_model=charging_model
		self.home_location=home_location
		self.maximum_soc=maximum_soc
		self.minimum_soc=minimum_soc
		self.routing_function=routing_function
		self.debug_max_iterations=debug_max_iterations

	def DistancesToLocations(self,current_location,locations):
		current_location_array=np.tile(current_location,(locations.shape[0],1))
		return haversine(current_location_array[:,1],current_location_array[:,0],
			locations[:,1],locations[:,0])

	def SimulateItinerary(self,itinerary,day):
		route={'speeds':itinerary.mean_speeds,'durations':itinerary.durations}
		day_index=np.floor((itinerary.datetimes-
			np.array(itinerary.datetimes[0]).astype('datetime64[D]'))/np.timedelta64(1,'D')).astype(int)
		today=day_index==day
		# print(np.cumsum(itinerary.distances[today]),itinerary.distances[today])
		route={'speeds':itinerary.mean_speeds[today],'durations':itinerary.durations[today]}
		return {'soc':self.energy_consumption_model.DeltaSOC_vec(route),
			'time':(itinerary.datetimes[today]-itinerary.datetimes[today][0])/np.timedelta64(1,'s'),
			'distance':itinerary.distances[today],
			'mean_speed':itinerary.mean_speeds[today],
			'dwell_time':itinerary.dwell_times[today],
			'travel_time':itinerary.durations[today],
			'lon':itinerary.dest_lons[today],
			'lat':itinerary.dest_lats[today]}


	def SimulateRoute(self,router_output,soc,time,dwell):
		soc-=self.energy_consumption_model.DeltaSOC(router_output)
		time+=router_output['durations'].sum()+dwell
		# print('a',router_output['distances'])
		return soc,time,router_output['distances'].sum(),router_output['speeds'].mean()

	def Simulate(self,locations,dwells,maximum_time,starting_location=None):
		#Decides on next location
		#Uses router to determine whether or not to go back home (time/energizing)
		#advances
		# print(self.routing_function)
		soc_trace=np.zeros(self.debug_max_iterations)
		distance_trace=np.zeros(self.debug_max_iterations)
		mean_speed_trace=np.zeros(self.debug_max_iterations)
		time_trace=np.zeros(self.debug_max_iterations)
		dwell_trace=np.zeros(self.debug_max_iterations)
		lon_trace=np.zeros(self.debug_max_iterations)
		lat_trace=np.zeros(self.debug_max_iterations)
		case_trace=np.zeros(self.debug_max_iterations)
		if starting_location is not None:
			current_location=starting_location
		else:
			current_location=self.home_location
		is_home_loc=(locations==self.home_location).sum(axis=1)==2
		locations=locations[~is_home_loc]
		dwells=dwells[~is_home_loc]
		# locations=locations[:2]
		# dwells=dwells[:2]
		soc=self.maximum_soc
		time=0
		soc_trace[0]=soc
		time_trace[0]=time
		dwell_trace[0]=0
		lon_trace[0]=current_location[0]
		lat_trace[0]=current_location[1]
		for i in range(self.debug_max_iterations):
			if locations.shape[0]==0:
				next_dwell=0
				#Gets routes for current location to next location and next location to home
				router_output_loc=self.routing_function(current_location,self.home_location)
				router_output=router_output_loc
				next_location=self.home_location
				#Merges the above routes
				#Gets final SOC and time for traveling to the next location then to home
				soc_nl_h,time_nl_h,_,_=self.SimulateRoute(router_output_loc,soc,time,next_dwell)
			else:
				#Picking next location based on minimum haversine distance to next location
				location_distances=self.DistancesToLocations(current_location,locations)
				# print(location_distances)
				next_location_index=np.argmin(location_distances)
				# print(next_location_index)
				next_location=locations[next_location_index]
				# print(next_location_index,next_location)
				next_dwell=dwells[next_location_index]
				#Gets routes for current location to next location and next location to home
				# print(current_location,next_location)
				router_output_loc=self.routing_function(current_location,next_location)
				router_output_home=self.routing_function(next_location,self.home_location)
				#Merges the above routes
				router_output=mbr.MergeRoutes([router_output_loc.copy(),router_output_home.copy()])
				#Gets final SOC and time for traveling to the next location then to home
				soc_nl_h,time_nl_h,_,_=self.SimulateRoute(router_output,soc,time,next_dwell)
			#If at home base
			if np.all(current_location==self.home_location):
				#No time left to complete any more missions
				if locations.shape[0]==0:
					#This always ends the day
					distance,mean_speed=(0,0)
					dwell=maximum_time-time
					case=7
					break
				elif (time_nl_h>maximum_time):
					#This always ends the day
					distance,mean_speed=(0,0)
					dwell=maximum_time-time
					case=1
					break
				#Vehicles must leave home base with full SOC
				elif (soc<self.maximum_soc):
					time_0=time
					soc,time=self.charging_model.Charge(soc,time,
						self.energy_consumption_model.energy_capacity)
					distance,mean_speed=(0,0)
					dwell=time-time_0
					case=2
				#Vehicle is ready to go
				else:
					case=3
					#Goes to next location
					soc,time,distance,mean_speed=self.SimulateRoute(router_output_loc,soc,time,next_dwell)
					current_location=next_location
					dwell=next_dwell
					try:
						locations=np.delete(locations,next_location_index,0)
						dwells=np.delete(dwells,next_location_index,0)
					except:
						pass
			#Vehicle is at a location other than home base
			else:
				#Cannot make it to next location and back on current charge
				if (soc_nl_h<self.minimum_soc):
					case=4
					#Goes back home
					router_output=self.routing_function(current_location,self.home_location)
					soc,time,distance,mean_speed=self.SimulateRoute(router_output,soc,time,0)
					current_location=self.home_location
					dwell=0
				elif (time_nl_h>maximum_time):
					case=5
					#Goes back home
					# print('c5',current_location,self.home_location)
					router_output=self.routing_function(current_location,self.home_location)
					soc,time,distance,mean_speed=self.SimulateRoute(router_output,soc,time,0)
					current_location=self.home_location
					dwell=0
				else:
					case=6
					#Goes to next location
					soc,time,distance,mean_speed=self.SimulateRoute(router_output_loc,soc,time,next_dwell)
					current_location=next_location
					dwell=next_dwell
					try:
						locations=np.delete(locations,next_location_index,0)
						dwells=np.delete(dwells,next_location_index,0)
					except:
						pass
			# print(i,case,locations,locations.shape,current_location,self.home_location,np.all(current_location==self.home_location))
					# print(locations.shape)
			# print(dwell/3600,case)
			soc_trace[i+1]=soc
			distance_trace[i+1]=distance
			mean_speed_trace[i+1]=mean_speed
			time_trace[i+1]=time
			dwell_trace[i+1]=dwell
			case_trace[i+1]=case
			lon_trace[i+1]=current_location[0]
			lat_trace[i+1]=current_location[1]
		# print(i+1)
		soc_trace=soc_trace[:i+1]
		distance_trace=distance_trace[:i+1]
		mean_speed_trace=mean_speed_trace[:i+1]
		time_trace=time_trace[:i+1]
		dwell_trace=dwell_trace[:i+1]
		lon_trace=lon_trace[:i+1]
		lat_trace=lat_trace[:i+1]
		case_trace=case_trace[:i+1]
		return {'soc':soc_trace,'time':time_trace,'distance':distance_trace,'mean_speed':mean_speed_trace,
			'travel_time':np.append(0,np.diff(time_trace))-dwell_trace,
			'dwell_time':dwell_trace,'lon':lon_trace,'lat':lat_trace,'case':case_trace},locations,dwells

def PlotVehicleItineraryData(itinerary):
	fig=plt.figure(figsize=(8,10))
	plt.subplot(321)
	plt.plot(np.cumsum(itinerary['distance']/1000))
	plt.title('Cumulative Distance Traveled')
	plt.ylabel('Distance [km]')
	plt.xticks(ticks=np.arange(0,itinerary['distance'].shape[0],1),labels=[])
	plt.grid(linestyle='--')
	plt.subplot(322)
	plt.plot(itinerary['time']/3600)
	plt.title('Cumulative Operating Time')
	plt.ylabel('Time [hrs]')
	plt.xticks(ticks=np.arange(0,itinerary['distance'].shape[0],1),labels=[])
	plt.grid(linestyle='--')
	plt.subplot(323)
	plt.plot(itinerary['mean_speed']*3.6)
	plt.title('Mission Mean Speed')
	plt.ylabel('Mean Speed [kmh]')
	plt.xticks(ticks=np.arange(0,itinerary['distance'].shape[0],1),labels=[])
	plt.grid(linestyle='--')
	plt.subplot(324)
	plt.plot(itinerary['travel_time']/60)
	plt.title('Mission Travel Time')
	plt.ylabel('Time [min]')
	plt.xticks(ticks=np.arange(0,itinerary['distance'].shape[0],1),labels=[])
	plt.grid(linestyle='--')
	plt.subplot(325)
	plt.plot(itinerary['soc'])
	plt.title('Mission-End SOC')
	plt.ylabel('SOC [dim]')
	plt.xticks(ticks=np.arange(0,itinerary['distance'].shape[0],1))
	plt.grid(linestyle='--')
	plt.subplot(326)
	plt.plot(itinerary['dwell_time']/60)
	plt.title('Mission Dwell Time')
	plt.ylabel('Time [min]')
	plt.xticks(ticks=np.arange(0,itinerary['distance'].shape[0],1))
	plt.grid(linestyle='--')


def haversine(lon1, lat1, lon2, lat2):
	r = 6372800. #[m]

	dLat = np.radians(lat2 - lat1)
	dLon = np.radians(lon2 - lon1)
	lat1 = np.radians(lat1)
	lat2 = np.radians(lat2)

	a = np.sin(dLat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dLon/2)**2
	c = 2*np.arcsin(np.sqrt(a))
	return c * r








