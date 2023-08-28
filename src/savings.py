import os
import sys
import time
import numpy as np

from numba import njit
from numba import types
from numba.typed import Dict,List

from .utilities import ProgressBar

def AdjacencyFromPathWeights(path_weights,indices=np.array([])):

	if not np.any(indices):
		indices=np.arange(0,len(weights[0]),1,dtype=int)

	n_v=len(indices)

	adjacency=np.zeros((n_v,n_v))

	for idx0 in range(n_v):
		for idx1 in range(n_v):

			adjacency[idx0,idx1]=path_weights[idx0][idx1]

	return adjacency


def Savings(adjacency,depot_vertex,
	delivery_vertices,delivery_loads,
	vehicle_capacity,vehicle_range,
	max_iterations=10000):

	#number of vertices
	n=len(adjacency)

	#Populating the savings matrix
	#This matrix is the difference between the combined weights to origin
	#for the row and column and the weight of the direct link between the row and column
	cost_from,cost_to=np.meshgrid(adjacency[:,0],adjacency[0])
	savings=cost_from+cost_to-adjacency
	savings[savings<0]=0
	savings[np.diag_indices(n)]=0

	#Initial routes are 0-leg routes
	routes=[[idx] for idx in range(1,n)]
	route_loads=list(delivery_loads)
	route_distances=[RouteWeight(adjacency,[0,idx,0]) for idx in range(1,n)]

	#Getting all possible combinations for origins and destinations
	index_from,index_to=np.meshgrid(list(range(n)),list(range(n)))
	indices=np.vstack((index_from.flatten(),index_to.flatten())).astype(int).T

	#Looping until ma iterations or all savings incorporated
	for idx0 in range(max_iterations):
		# print(routes)

		#If all savings incorporated then break the loop
		if savings.sum()==0:
			success=True
			break

		#Initializing nested loop variables
		link_savings=savings.flatten()
		best_link_savings=indices[np.argmax(link_savings)]
		from_edge_index,to_edge_index=[],[]

		#Finding the remaining unused link whih offers the highest savings
		for idx1 in range(len(routes)):

			if best_link_savings[0] in {routes[idx1][0],routes[idx1][-1]}:
				from_edge_index=[idx1]

			if best_link_savings[1] in {routes[idx1][0],routes[idx1][-1]}:
				to_edge_index=[idx1]

		#If there is a valid link found
		if bool(from_edge_index)&bool(to_edge_index):

			#Self loops are not considered
			if from_edge_index[0]!=to_edge_index[0]:
				# print(from_edge_index[0],to_edge_index[0])

				#Calculating the most efficient rote incorporating the new link
				#and its associated weights
				tentative_route=[0,*routes[from_edge_index[0]],*routes[to_edge_index[0]],0]
				tentative_route=TwoOpt(adjacency,tentative_route)

				tentative_route_load=(
					route_loads[from_edge_index[0]]+route_loads[to_edge_index[0]])
				tentative_route_distance=RouteWeight(adjacency,tentative_route)

				feasible_route=((tentative_route_load<=vehicle_capacity)&
					(tentative_route_distance<=vehicle_range))

				#If the weights are less than or equal to the capacities the route is added
				if feasible_route:

					#Adding the tentative route
					routes[from_edge_index[0]].extend(routes[to_edge_index[0]])
					route_loads[from_edge_index[0]]+=route_loads[to_edge_index[0]]
					route_distances[from_edge_index[0]]=tentative_route_distance

					#Removing the absorbed route
					routes.remove(routes[to_edge_index[0]])
					route_loads.remove(route_loads[to_edge_index[0]])
					route_distances.remove(route_distances[to_edge_index[0]])

		savings[best_link_savings[0],best_link_savings[1]]=0
		savings[best_link_savings[1],best_link_savings[0]]=0

	new_routes=[]
	delivery_vertices=np.hstack((0,delivery_vertices))
	for route in routes:
		new_routes.append(list(
			np.hstack((depot_vertex,delivery_vertices[route],depot_vertex))))

	if idx0==max_iterations-1:
		success=False

	return new_routes,route_loads,route_distances,success

def TwoOpt(adjacency,route,max_iterations=100):

	#Since the start and end vertices can't be changed 4 is the minimum route length
	#where savings may be found
	if len(route)>4:

		#Freezing the start and finish vertices
		route_start=route[0]
		route_finish=route[-1]
		route=np.array(route[1:-1])

		#Looping until max iterations or convergence
		for i in range(max_iterations):

			#Calculating the current route distance
			distance=RouteWeight(adjacency,np.hstack((route_start,route,route_finish)))

			#Looping on perturbations
			for idx1 in range(1,len(route)-1):
				for idx2 in range(idx1+1,len(route)):

					#Creating a tentative route
					tentative_route=np.hstack(
						(route_start,route[:idx1],np.flip(route[idx1:idx2+1]),
						route[idx2+1:],route_finish)).astype(int)

					#Calculating tentative current route distance
					tentative_distance=RouteWeight(adjacency,tentative_route)

					#If the tentative route is an improvement it replaces the current route
					if tentative_distance<distance:
						route=tentative_route[1:-1]
						distance=tentative_distance

		route=np.hstack((route_start,route,route_finish))

	return route

def RouteWeight(adjacency,route):

	from_indices=route[:-1]
	to_indices=route[1:]

	route_weight=0

	for idx in range(len(from_indices)):

		route_weight+=adjacency[from_indices[idx],to_indices[idx]]

	return route_weight
