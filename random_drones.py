import logging
import random
import bisect

from copy import deepcopy
from sage.all import *
import random

def genCentersGridGraph(n, m):
    G = {}
    deltas = [[0, 2], [2, 0], [0, -2], [-2, 0]]
    for x in range(1, 2*n+1, 2):
        for y in range(1, 2*m+1, 2):
            for dx, dy in deltas:
                if 1 <= x+dx <= 2*n and 1 <= y+dy <= 2*m:
                    G.setdefault((x, y), []).append((x+dx, y+dy))
                    G.setdefault((x+dx, y+dy), []).append((x, y))
    for k in G:
        G[k] = list(set(G[k]))
    # print "Finished Centers", G.keys()
    G = Graph(G)
    # print "After", G.vertices(), len(G.vertices())
    return Graph(G)

def genDiGridFromCenters(centers):
    G = {}
    cycle = [[0, 1], [1, 0], [0, -1], [-1, 0]]
    for x, y in centers.vertices():
    # for x in range(1, 2*n+1, 2):
    #     for y in range(1, 2*m+1, 2):
        if ((x-1)/2)%2 == ((y-1)/2)%2:
            aux = cycle
        else:
            aux = cycle[::-1]
        for i in xrange(4):
            dux, duy = aux[i]
            dvx, dvy = aux[(i+1)%4]
            G.setdefault((x+dux, y+duy), []).append((x+dvx, y+dvy))
    return DiGraph(G)

def randDFS(G, depth):
    T = {}
    def DFS(x, y, depth):
        if y in T or depth <= 0:
            return

        T.setdefault(x, []).append(y)
        N = deepcopy(G[y])
        random.shuffle(N)
        for w in N:
            DFS(y, w, depth-1)

    u = random.choice(G.vertices())
    N = deepcopy(G[u])
    random.shuffle(N)
    for v in N:
        DFS(u, v, depth)
    return Graph(T)

def randomWalk(G, size):
    u = random.choice(G.vertices())
    S = set()
    S.add(u)
    current = u
    while len(S) != size:
        current = random.choice(G[current])
        S.add(current)
    return G.subgraph(list(S))

def removeVertices(G, k):
    V = random.sample(G.vertices(), k)
    H = G.subgraph(V)
    assert H.connected_components_subgraphs()[0].order() == max(H.connected_components_sizes())
    return H.connected_components_subgraphs()[0]

def getRandomGraph(n, m, k, method=0, depth=10):
    GridCenters = genCentersGridGraph(n, m)
    # print "Grid size is", GridCenters.order(), len(G.vertices()), G.vertices()
    if method == 0:
        H = randomWalk(GridCenters, k)
    elif method == 1:
        H = removeVertices(G, k)
    elif method == 2:
        H = randDFS(G, depth)
    else:
        raise Exception("Wrong value for method")
    return genDiGridFromCenters(H)


def comulative_distribution_function(probabilities):
	result = [0]*len(probabilities)
	csum = 0
	tsum = sum(probabilities)
	for i in range(len(probabilities)):
		csum += probabilities[i]
		result[i] = float(csum)/float(tsum)
	return result


def pick_element(elements, probabilities):
	assert len(elements) == len(probabilities)
	assert sum(probabilities) == 1
	cdf_vals = comulative_distribution_function(probabilities)
	x = random.random()
	idx = bisect.bisect(cdf_vals, x)
	return elements[idx]


def create_covering_statistics(g):
	cover_statistics = {}
	for p in g:
		for d in g[p]:
			cover_statistics[(p,d)] = 0
	return cover_statistics


def create_gossiping_statistics(robots):
	gossiping_statistics = {}
	for r in robots:
		gossiping_statistics[r] = {}
		for d in robots:
			gossiping_statistics[r][d] = 0
	return gossiping_statistics


def simula_drones(g, init_pos_robot, bouncing=False, ni=None, edges_circles=None, p = None):
	logging.basicConfig(format='%(message)s',filename='statistics.log',level=logging.DEBUG)
	c = 0
	current_pos_robot = init_pos_robot
	choices = ['stay','shift']
	if p is None:
		p = 0.5
	weights = [1-p, p]
	cover_statistics = create_covering_statistics(g)
	gsp_statistics = create_gossiping_statistics([x if type(x) is not tuple else x[0] for p in init_pos_robot for x in init_pos_robot[p]])

	logging.info('Statictics format:')

	logging.info('---------------------------')
	logging.info('<iteration>')
	logging.info('** Uncovering info **')
	logging.info('\t<average>')
	logging.info('\t<maximum>')
	edges = ''
	for e in cover_statistics:
		edges += str(e)+' '
	logging.info('\t<edges> '+str(edges))
	logging.info('')
	logging.info('** Gossiping info **')
	logging.info('\t<average>')
	logging.info('\t<maximum>')
	logging.info('')
	logging.info('')

	while ni is None or c<ni:
		#print(current_pos_robot)
		#raw_input()
		
		for e in cover_statistics:
			cover_statistics[e] += 1

		for r in gsp_statistics:
			gsp_statistics[r][r] = c

		new_pos_robot = {}		
		for p in current_pos_robot:
			c_robots = current_pos_robot[p]
			if len(c_robots)>1:
				for r1 in c_robots:
					robot1 = r1
					if type(r1) is tuple:
						robot1 = r1[0]
					for r2 in c_robots:
						if r2 == r1:
							continue
						robot2 = r2
						if type(r2) is tuple:
							robot2 = r2[0]
						for r in gsp_statistics:
							if gsp_statistics[robot1][r]<gsp_statistics[robot2][r]:
								gsp_statistics[robot1][r] = gsp_statistics[robot2][r]


			if not bouncing or len(c_robots)==1:
				#pick randomly a destiny node for every robot in this position
				for r in c_robots:
					if type(r) is tuple: #if r is a tuple then r = (robot, circle) where 'robot' is the label of the robot and 'circle' is the label of the circle where the robot is						
						picked = 0;
						if len(g[p])>1:
							action = pick_element(choices, weights)
							picked = 0 if ((action == 'stay' and edges_circles[(p,g[p][0])] == r[1]) or (action == 'shift' and edges_circles[(p,g[p][0])] != r[1])) else 1
					else:
						picked  = random.randint(0,len(g[p])-1)

					new_pos = g[p][picked]
					cover_statistics[(p,new_pos)] = 0
					new_r = r
					if type(r) is tuple:
						new_r = (r[0], edges_circles[(p,new_pos)])
					if new_pos in new_pos_robot:
						new_pos_robot[new_pos] += [new_r]
					else:
						new_pos_robot[new_pos] = [new_r]
			else:
				#there are at most two robots per node and two possible destiny nodes per node, then, send a robot per node 
				for i in [0,1]:
					r = c_robots[i];
					if type(r) is tuple: #if r is a tuple then r = (robot, circle) where 'robot' is the label of the robot and 'circle' is the label of the circle where the robot is
						new_pos  = g[p][0] if edges_circles[(p, g[p][0])] == r[1] else g[p][1]
					else:
						new_pos = g[p][i]	
					cover_statistics[(p,new_pos)] = 0				
					if new_pos in new_pos_robot:
						new_pos_robot[new_pos] += [r]
					else:
						new_pos_robot[new_pos] = [r]
		c += 1
		current_pos_robot = new_pos_robot
		logging.info('---------------------------')
		logging.info('iter %d',c)
		logging.info('** Uncovering info **')
		stat = [cover_statistics[e] for e in cover_statistics]
		logging.info('\tavg %f', sum(stat)/float(len(stat)))
		logging.info('\tmax %d',max(stat))
		logging.info('\t'+str(stat))
		logging.info('')
		gsp_values = [c-1-gsp_statistics[i][x] for i in gsp_statistics for x in gsp_statistics[i]]
		logging.info('** Gossiping info **')
		logging.info('\tavg %f',sum(gsp_values)/float(len(gsp_values)))
		logging.info('\tmax %d',max(gsp_values))
		logging.info('')
		logging.info('')
		



my_g = {1:[3],2:[1],3:[4],4:[2,6],5:[8],6:[9],7:[4,5],8:[10],9:[7,12],10:[7],11:[9],12:[13],13:[11]}
e_circles = {(1,3):'A',(3,4):'A',(4,2):'A',(2,1):'A',(4,6):'B',(6,9):'B',(9,7):'B',(7,4):'B',(7,5):'C',(5,8):'C',(8,10):'C',(10,7):'C',(11,9):'D',(9,12):'D',(12,13):'D',(13,11):'D'}
i_p_r = {4:[('r1','A'),('r2','B')],13:[('r3','D')]}
simula_drones(my_g, i_p_r, True, None, e_circles)


#i_p_r = {4:['r1','r2'],13:['r3']}
#simula_drones(my_g, i_p_r, True, None)