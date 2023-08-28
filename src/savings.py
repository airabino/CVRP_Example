import os
import sys
import time
import numpy as np

from .utilities import ProgressBar

def SavingsRouterTwoOpt(adjacency,depot_vertex,
	delivery_vertices,delivery_loads,
	vehicle_capacities,vehicle_ranges,
	debug_max_iterations=10000):

	#number of vertices
	n=len(adjacency)

	#Populating the savings matrix
	cost_from,cost_to=np.meshgrid(adjacency[:,0],adjacency[0])
	savings=cost_from+cost_to-adjacency
	savings[savings<0]=0
	savings[np.diag_indices(n)]=0

	print(savings)

	#Initial routes are 0-leg routes
	routes=[[i] for i in range(1,n)]

	index_from,index_to=np.meshgrid(list(range(n)),list(range(n)))
	indices=np.vstack((index_from.flatten(),index_to.flatten())).astype(int).T

	hit_max_iterations=True
	for idx0 in range(debug_max_iterations):

		if savings.sum()==0:
			hit_max_iterations=False
			print('All route savings incorporated')
			break

		link_savings=savings.flatten()
		best_link_savings=indices[np.argmax(link_savings)]
		print(link_savings)
		print(best_link_savings)

		for idx1 in range(len(routes)):

			if best_link_savings[0] in {routes[idx1][0],routes[idx1][-1]}:
				from_edge_index=[idx1]

			if best_link_savings[1] in {routes[idx1][0],routes[idx1][-1]}:
				to_edge_index=[idx1]

		print(from_edge_index,to_edge_index)
		# break

		if bool(from_edge_index)&bool(to_edge_index):

			if from_edge_index[0]!=to_edge_index[0]:

				new_load=delivery_loads[from_edge_index[0]]+delivery_loads[to_edge_index[0]]

				if new_load<=capacity:

					routes[from_edge_index[0]].extend(routes[to_edge_index[0]])

					delivery_loads[from_edge_index[0]]+=delivery_loads[to_edge_index[0]]

					routes.remove(routes[to_edge_index[0]])

					delivery_loads.remove(delivery_loads[to_edge_index[0]])

		savings[best_link_savings[0],best_link_savings[1]]=0
		savings[best_link_savings[1],best_link_savings[0]]=0

	for route in routes:
		route.extend([0])
		route.insert(0,0)

	if hit_max_iterations:
		print('Max iterations hit before all route savings incorporated')

	return routes,delivery_loads,capacities,savings