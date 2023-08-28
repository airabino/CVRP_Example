import io
import re
import time
import zipfile
import requests
import csv
import json
import smopy
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class AssumedSpeedModel():
	def __init__(self,speeds_file):
		self.speeds_file=speeds_file
		self.Populate()

	def __call__(self,category):
		return self.Speed(category)

	def Populate(self):
		self.lookup={}
		with open(self.speeds_file, 'r') as f:
			reader = csv.reader(f)
			for row in reader:
				if row[2]=='mph':
					self.lookup[row[0]]=float(row[1])*0.44704
				if row[2]=='kmh':
					self.lookup[row[0]]=float(row[1])/3.6
				if row[2]=='mps':
					self.lookup[row[0]]=float(row[1])

	def Speed(self,category):
		try:
			return self.lookup[category]
		except:
			return 11.176


"""
Produces a undirected subgraph with all speeds added via AssumedSpeedModel
"""
def ProduceSubgraph(shapefile):
	g=nx.read_shp(shapefile)
	sgs=list(g.subgraph(c) for c in nx.connected_components(g.to_undirected()))
	idx_max=np.argmax([len(sg) for sg in sgs])
	return sgs[idx_max]

def ProduceCompleteSubgraph(shapefile,asm):
	g=nx.read_shp(shapefile)
	sgs=list(g.subgraph(c) for c in nx.connected_components(g.to_undirected()))
	idx_max=np.argmax([len(sg) for sg in sgs])
	sg=sgs[idx_max]
	sg=sg.to_undirected()
	sg=AddEdgeDistances(sg,asm)
	return sg

def get_path(n0, n1,sg):
	"""If n0 and n1 are connected nodes in the graph,
	this function returns an array of point
	coordinates along the road linking these two
	nodes."""
	return np.array(json.loads(sg[n0][n1]['Json'])['coordinates'])

def haversine(lon1, lat1, lon2, lat2):
	r = 6372800. #[m]

	dLat = np.radians(lat2 - lat1)
	dLon = np.radians(lon2 - lon1)
	lat1 = np.radians(lat1)
	lat2 = np.radians(lat2)

	a = np.sin(dLat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dLon/2)**2
	c = 2*np.arcsin(np.sqrt(a))
	return c * r

def get_path_length(path):
	return np.sum(haversine(path[1:, 0], path[1:, 1],path[:-1, 0], path[:-1, 1]))

def FindClosestNodes(n0,n1,nodes):
	# print((nodes[:, ::-1] - n0)**2)
	n0_a = np.argmin(np.sum((nodes - n0)**2, axis=1))
	n1_a = np.argmin(np.sum((nodes - n1)**2, axis=1))
	# print(np.sum((nodes - n0)**2, axis=1))
	# print(nodes[0],n0)
	return n0_a,n1_a

def AddEdgeDistances(sg,asm):
	for n0, n1 in sg.edges:
		path = get_path(n0, n1,sg)
		distance = get_path_length(path)
		# print(distance)
		
		sg.edges[n0, n1]['Distance [m]'] = distance
		# print(sg.edges[n0, n1]['MaxSpeed'])
		try:
			speed=float(re.findall(r'\b\d+\b',sg.edges[n0, n1]['MaxSpeed'])[0])*0.44704
		except Exception as e:
			# print(e)
			speed=asm(sg.edges[n0, n1]['Category']) #[m/s]
		# print(distance/speed)
		sg.edges[n0, n1]['Time [sec]'] = distance/speed
		sg.edges[n0, n1]['Speed [m/s]'] = speed

		# return
	return sg

def Path(start,finish,sg,weight='Time'):
	nodes=np.array(sg.nodes)
	s1,f1=FindClosestNodes(start,finish,nodes)
	t0=time.time()
	if weight=='Distance':
		path = nx.shortest_path(sg,source=tuple(nodes[s1]),target=tuple(nodes[f1]),weight='Distance [m]',
			method='bellman-ford')
		# path = nx.shortest_path(sg,source=tuple(nodes[s1]),target=tuple(nodes[f1]),weight='Distance [m]',
		# 	method='dijkstra')
	elif weight=='Time':
		path = nx.shortest_path(sg,source=tuple(nodes[s1]),target=tuple(nodes[f1]),weight='Time [sec]',
			method='bellman-ford')
		# path = nx.shortest_path(sg,source=tuple(nodes[s1]),target=tuple(nodes[f1]),weight='Time [sec]',
		# 	method='dijkstra')
	print('b',time.time()-t0)
	# path = nx.shortest_path(sg,source=tuple(nodes[s1]),target=tuple(nodes[f1]),weight='Distance',
	# 	method='dijkstra')
	t0=time.time()
	roads = pd.DataFrame([sg.edges[path[i], path[i + 1]] for i in range(len(path) - 1)],
	columns=['Name','MaxSpeed','Distance [m]','Time [sec]','Speed [m/s]'])
	print('c',time.time()-t0)
	return path,roads

def get_full_path(path,sg):
	"""Return the positions along a path."""
	p_list = []
	curp = None
	for i in range(len(path) - 1):
		p = get_path(path[i], path[i + 1],sg)
		if curp is None:
			curp = p
		if (np.sum((p[0] - curp) ** 2) >
				np.sum((p[-1] - curp) ** 2)):
			p = p[::-1, :]
		p_list.append(p)
		curp = p[-1]
	return np.vstack(p_list)

def SmopyPlot(path,pos0,pos1,sg):
	# print(path,pos0,pos1)
	# linepath = get_full_path(path,sg)
	linepath=np.array([*path])
	# print(linepath)
	
	bounds=(np.min(linepath[:,1]),np.min(linepath[:,0]),
		np.max(linepath[:,1]),np.max(linepath[:,0]))

	m = smopy.Map(bounds, z=15, margin=.1)
	x, y = m.to_pixels(linepath[:, 1], linepath[:, 0])
	
	ax = m.show_mpl(figsize=(8, 8))
	# Mark our two positions.
	ax.plot(x[0], y[0], 'ob', ms=20)
	ax.plot(x[-1], y[-1], 'or', ms=20)
	# Plot the itinerary.
	ax.plot(x, y, '-k', lw=3)
	return x,y

if __name__ == "__main__":
	sg=pkl.load(open('sg.pkl'))
