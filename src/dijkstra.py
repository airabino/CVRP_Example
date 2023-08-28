import os
import sys
import time
import numpy as np

from numba import njit
from numba import types
from numba.typed import Dict,List

def All_Pairs_Shortest_Paths(adjacency,chargers=np.array([]),
	vehicle_range=float(sys.maxsize),compiled=True):

	if compiled:
		router=Dijkstra_Range_NJIT
	else:
		router=Dijkstra_Range

	shortest_paths={}
	shortest_path_weights={}
	remaining_range={}

	for idx in range(len(adjacency)):

		(shortest_paths_idx,shortest_path_weights_idx,remaining_range_idx
			)=router(idx,adjacency,
			vehicle_range=vehicle_range,extension=chargers)

		shortest_paths[idx]=shortest_paths_idx
		shortest_path_weights[idx]=shortest_path_weights_idx
		remaining_range[idx]=remaining_range_idx

	return shortest_paths,shortest_path_weights,remaining_range

def Dijkstra_Range(start_vertex,adjacency,vehicle_range=float(sys.maxsize),
	extension=np.array([])):

	if not any(extension):
		extension=np.array([False]*len(adjacency))

	#Indices to vertices
	vertices=np.array(list(range(len(adjacency))))
	# print(vertices)

	#To start with, all vertices are unvisited
	unvisited_vertices=list(range(len(adjacency)))

	#Initializing dicts to stroe shortest path information
	shortest_path={}
	shortest_path_weight={}
	shortest_path_range={}


	# #Initially all paths are of infinite weight
	for vertex in unvisited_vertices:
		shortest_path[vertex]=[]
		shortest_path_weight[vertex]=np.inf
		shortest_path_range[vertex]=vehicle_range

	# The start node has a path weight of zero
	shortest_path[start_vertex]=[start_vertex]
	shortest_path_weight[start_vertex]=0.
	shortest_path_range[start_vertex]=vehicle_range

	#Looping on vertices until all possible are visited.
	#Each time the closest vertex is selected
	while unvisited_vertices:

		#Finding the closest unvisited vertex to the start vertex
		current_vertex=unvisited_vertices[0]

		for vertex in unvisited_vertices:
			if shortest_path_weight[vertex]<shortest_path_weight[current_vertex]:
				current_vertex=vertex

		#Updating path information for neighbors

		#An edge exists if the weight of the edge is greater than zero
		neighbors=vertices[adjacency[current_vertex]>0]

		for neighbor in neighbors:

			tentative_path_weight=(shortest_path_weight[current_vertex]+
				adjacency[current_vertex,neighbor])

			if extension[current_vertex]:
				tentative_path_range=vehicle_range-adjacency[current_vertex,neighbor]
			else:
				tentative_path_range=(shortest_path_range[current_vertex]-
					adjacency[current_vertex,neighbor])

			if tentative_path_range>0:

				if tentative_path_weight<shortest_path_weight[neighbor]:

					shortest_path[neighbor]=[*shortest_path[current_vertex],neighbor]
					shortest_path_weight[neighbor]=tentative_path_weight
					shortest_path_range[neighbor]=tentative_path_range

		unvisited_vertices.remove(current_vertex)

	return shortest_path,shortest_path_weight,shortest_path_range

@njit(cache=True)
def Dijkstra_Range_NJIT(start_vertex,adjacency,vehicle_range=float(sys.maxsize),
	extension=np.array([])):

	if not np.any(extension):
		extension=np.array([False]*len(adjacency))

	#Indices to vertices
	vertices=np.array(list(range(len(adjacency))))
	# print(vertices)

	#To start with, all vertices are unvisited
	unvisited_vertices=list(range(len(adjacency)))

	#Initializing dicts to stroe shortest path information

	# list_type = types.ListType(types.int64)
	shortest_path=Dict.empty(key_type=types.int64,
		value_type=types.ListType(types.int64[:]))

	shortest_path_weight=Dict.empty(key_type=types.int64,value_type=types.float64)

	shortest_path_range=Dict.empty(key_type=types.int64,value_type=types.float64)


	# #Initially all paths are of infinite weight
	for vertex in unvisited_vertices:
		shortest_path[vertex]=List.empty_list(types.int64)
		shortest_path_weight[vertex]=np.inf
		shortest_path_range[vertex]=vehicle_range

	# The start node has a path weight of zero
	shortest_path[start_vertex].append(start_vertex)
	shortest_path_weight[start_vertex]=0.
	shortest_path_range[start_vertex]=vehicle_range

	#Looping on vertices until all possible are visited.
	#Each time the closest vertex is selected
	while unvisited_vertices:

		#Finding the closest unvisited vertex to the start vertex
		current_vertex=unvisited_vertices[0]

		for vertex in unvisited_vertices:
			if shortest_path_weight[vertex]<shortest_path_weight[current_vertex]:
				current_vertex=vertex

		#Updating path information for neighbors

		#An edge exists if the weight of the edge is greater than zero
		neighbors=vertices[adjacency[current_vertex]>0]

		for neighbor in neighbors:

			tentative_path_weight=(shortest_path_weight[current_vertex]+
				adjacency[current_vertex,neighbor])

			if extension[current_vertex]:
				tentative_path_range=vehicle_range-adjacency[current_vertex,neighbor]
			else:
				tentative_path_range=(shortest_path_range[current_vertex]-
					adjacency[current_vertex,neighbor])

			if tentative_path_range>0:

				if tentative_path_weight<shortest_path_weight[neighbor]:

					new_shortst_path=List.empty_list(types.int64)
					new_shortst_path.extend(shortest_path[current_vertex])
					new_shortst_path.append(neighbor)
					shortest_path[neighbor]=new_shortst_path
					shortest_path_weight[neighbor]=tentative_path_weight
					shortest_path_range[neighbor]=tentative_path_range

		unvisited_vertices.remove(current_vertex)

	return shortest_path,shortest_path_weight,shortest_path_range

def Dijkstra(start_vertex,adjacency):

	#Indices to vertices
	vertices=np.array(list(range(len(adjacency))))
	# print(vertices)

	#To start with, all vertices are unvisited
	unvisited_vertices=list(range(len(adjacency)))

	#Initializing dicts to stroe shortest path information
	shortest_path={}
	shortest_path_weight={}


	# #Initially all paths are of infinite weight
	for vertex in unvisited_vertices:
		shortest_path[vertex]=[]
		shortest_path_weight[vertex]=np.inf

	# The start node has a path weight of zero
	shortest_path[start_vertex]=[start_vertex]
	shortest_path_weight[start_vertex]=0.

	#Looping on vertices until all possible are visited.
	#Each time the closest vertex is selected
	while unvisited_vertices:

		#Finding the closest unvisited vertex to the start vertex
		current_vertex=unvisited_vertices[0]

		for vertex in unvisited_vertices:
			if shortest_path_weight[vertex]<shortest_path_weight[current_vertex]:
				current_vertex=vertex

		#Updating path information for neighbors

		#An edge exists if the weight of the edge is greater than zero
		neighbors=vertices[adjacency[current_vertex]>0]

		for neighbor in neighbors:

			tentative_path_weight=(shortest_path_weight[current_vertex]+
				adjacency[current_vertex,neighbor])

			if tentative_path_weight<shortest_path_weight[neighbor]:

				shortest_path[neighbor]=[*shortest_path[current_vertex],neighbor]
				shortest_path_weight[neighbor]=tentative_path_weight

		unvisited_vertices.remove(current_vertex)

	return shortest_path,shortest_path_weight

@njit(cache=True)
def Dijkstra_NJIT(start_vertex,adjacency):

	#Indices to vertices
	vertices=np.array(list(range(len(adjacency))))
	# print(vertices)

	#To start with, all vertices are unvisited
	unvisited_vertices=list(range(len(adjacency)))

	#Initializing dicts to stroe shortest path information

	# list_type = types.ListType(types.int64)
	shortest_path=Dict.empty(key_type=types.int64,
		value_type=types.ListType(types.int64[:]))

	shortest_path_weight=Dict.empty(key_type=types.int64,value_type=types.float64)


	# #Initially all paths are of infinite weight
	for vertex in unvisited_vertices:
		shortest_path[vertex]=List.empty_list(types.int64)
		shortest_path_weight[vertex]=np.inf

	# The start node has a path weight of zero
	shortest_path[start_vertex].append(start_vertex)
	shortest_path_weight[start_vertex]=0.

	#Looping on vertices until all possible are visited.
	#Each time the closest vertex is selected
	while unvisited_vertices:

		#Finding the closest unvisited vertex to the start vertex
		current_vertex=unvisited_vertices[0]

		for vertex in unvisited_vertices:
			if shortest_path_weight[vertex]<shortest_path_weight[current_vertex]:
				current_vertex=vertex

		#Updating path information for neighbors

		#An edge exists if the weight of the edge is greater than zero
		neighbors=vertices[adjacency[current_vertex]>0]

		for neighbor in neighbors:

			tentative_path_weight=(shortest_path_weight[current_vertex]+
				adjacency[current_vertex,neighbor])

			if tentative_path_weight<shortest_path_weight[neighbor]:

				new_shortst_path=List.empty_list(types.int64)
				new_shortst_path.extend(shortest_path[current_vertex])
				new_shortst_path.append(neighbor)
				shortest_path[neighbor]=new_shortst_path
				shortest_path_weight[neighbor]=tentative_path_weight

		unvisited_vertices.remove(current_vertex)

	return shortest_path,shortest_path_weight