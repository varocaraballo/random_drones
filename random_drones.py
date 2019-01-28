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


def initialPositions(n, m, getAssociatedEdge=False):
    initial = {}
    associatedEdges = []
    for x in range(1, 2*n+1, 2):
        for y in range(1, 2*m+1, 2):
            if (x/2) % 2 == 1:
                pos = (x+1, y)
                # initial[(x, y)] = (x+1, y)
            else:
                pos = (x-1, y)
                # initial[(x, y)] = (x-1, y)
            initial[(x, y)] = pos
            if getAssociatedEdge:
                if ((x-1)/2) % 2 == 0 and  ((y-1)/2)%2 == 0:
                    associatedEdges.append((pos, (pos[0]+1, pos[1]+1)))
                    # associatedEdges[pos] = [pos[0]+1, pos[1]+1]
                elif ((x-1)/2) % 2 == 0 and  ((y-1)/2)%2 == 1:
                    associatedEdges.append((pos, (pos[0]+1, pos[1]-1)))
                    # associatedEdges[pos] = [pos[0]+1, pos[1]-1]
                elif ((x-1)/2) % 2 == 1 and  ((y-1)/2)%2 == 1:
                    associatedEdges.append((pos, (pos[0]-1, pos[1]-1)))
                    # associatedEdges[pos] = [pos[0]-1, pos[1]-1]
                else:
                    associatedEdges.append((pos, (pos[0]-1, pos[1]+1)))
                    # associatedEdges[pos] = [pos[0]-1, pos[1]+1]
    if getAssociatedEdge:
        return initial, associatedEdges

    return initial


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


def getRandomGraph(n, m, k, robots, method=0, depth=10):
    GridCenters = genCentersGridGraph(n, m)
    initial = initialPositions(n, m)
    # print "Grid size is", GridCenters.order(), len(G.vertices()), G.vertices()
    if method == 0:
        H = randomWalk(GridCenters, k)
    elif method == 1:
        H = removeVertices(GridCenters, k)
    elif method == 2:
        H = randDFS(GridCenters, depth)
    else:
        raise Exception("Wrong value for method")
    if method == 0:
        assert H.order() == k
    robotCenters = random.sample(H.vertices(), robots)
    positions = {x:initial[x] for x in robotCenters}
    invPos = {}
    for k, v in positions.iteritems():
        invPos.setdefault(v, []).append(k)
    return genDiGridFromCenters(H), invPos, H  # The grid graph and the graph generated from the centers of the circles


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
            cover_statistics[(p, d)] = 0
    return cover_statistics


def create_gossiping_statistics(robots):
    gossiping_statistics = {}
    for r in robots:
        gossiping_statistics[r] = {}
        for d in robots:
            gossiping_statistics[r][d] = 0
    return gossiping_statistics


def update_simulation_state(g, current_pos, robot, new_pos, new_pos_robot, covering_statistics, robot_circles, edges_circles):
	covering_statistics[(current_pos,new_pos)] = 0

	if robot_circles is not None:
		robot_circles[robot] = edges_circles[(current_pos,new_pos)]

	if new_pos in new_pos_robot:
		new_pos_robot[new_pos] += [robot]
	else:
		new_pos_robot[new_pos] = [robot]

def createTransitionMatrix(G):

    for v in G:
        pass


def simula_drones(g, init_pos_robot, bouncing=False, ni=None, robot_circles=None, edges_circles=None, p = None):
	# p is the probability to shift to a neighboring circle
    logging.basicConfig(format='%(message)s',filename='statistics.log',level=logging.DEBUG)
    c = 0
    current_pos_robot = init_pos_robot
    choices = ['stay','shift']
    if p is None:
        p = 0.5
    weights = [1-p, p]
    cover_statistics = create_covering_statistics(g)
    gsp_statistics = create_gossiping_statistics([x for p in init_pos_robot for x in init_pos_robot[p]])

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
                    for r2 in c_robots:
                        if r2 == r1:
                            continue
                        for r in gsp_statistics:
                            if gsp_statistics[r1][r]<gsp_statistics[r2][r]:
                                gsp_statistics[r1][r] = gsp_statistics[r2][r]


            if not bouncing or len(c_robots)==1:
                #pick randomly a destiny node for every robot in this position
                for r in c_robots:
                    if robot_circles is not None: #if r is a tuple then r = (robot, circle) where 'robot' is the label of the robot and 'circle' is the label of the circle where the robot is                       
                        picked = 0;
                        if len(g[p])>1:
                            action = pick_element(choices, weights)
                            picked = 0 if ((action == 'stay' and edges_circles[(p,g[p][0])] == robot_circles[r]) or (action == 'shift' and edges_circles[(p,g[p][0])] != robot_circles[r])) else 1
                    else:
                        picked  = random.randint(0,len(g[p])-1)

                    new_pos = g[p][picked]
                    
                    update_simulation_state(g, p, r, new_pos, new_pos_robot, cover_statistics, robot_circles, edges_circles)
            else:
                #there are at most two robots per node and two possible destiny nodes per node, then, send a robot per node 
                for i in [0,1]:
                    r = c_robots[i];
                    if robot_circles is not None: #if r is a tuple then r = (robot, circle) where 'robot' is the label of the robot and 'circle' is the label of the circle where the robot is
                        new_pos  = g[p][0] if edges_circles[(p, g[p][0])] == robot_circles[r] else g[p][1]
                    else:
                        new_pos = g[p][i]   

                    update_simulation_state(g, p, r, new_pos, new_pos_robot, cover_statistics, robot_circles, edges_circles)
                    
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
        



# my_g = {1:[3],2:[1],3:[4],4:[2,6],5:[8],6:[9],7:[4,5],8:[10],9:[7,12],10:[7],11:[9],12:[13],13:[11]}
# i_p_r = {4:['r1','r2'],13:['r3']}

# e_circles = {(1,3):'A',(3,4):'A',(4,2):'A',(2,1):'A',(4,6):'B',(6,9):'B',(9,7):'B',(7,4):'B',(7,5):'C',(5,8):'C',(8,10):'C',(10,7):'C',(11,9):'D',(9,12):'D',(12,13):'D',(13,11):'D'}
# simula_drones(my_g, i_p_r, True, None,{'r1':'A','r2':'B','r3':'D'},e_circles)



# simula_drones(my_g, i_p_r, True, None)

def DFS(G, u, v, depth, p):
    # print "doing", u, v, depth
    reachable = set()

    def innerDFS(u, v, depth, p):
        # print "       inner doing", u, v, depth
        if depth == 0:
            reachable.add((u, v, p))
            return
        for w in G.neighbors_out(v):
            innerDFS(v, w, depth-1, p*Rational(1/float(len(G.neighbors_out(v)))))

    innerDFS(u, v, depth, p)

    return reachable


def getMatrix(G, initialEdges):
    n = len(initialEdges)
    # assert len(initialEdges) == n
    M = [[0 for j in xrange(n)] for i in xrange(n)]
    i = 0
    indices = {}

    for e in initialEdges:
        indices[e] = i
        i += 1
    # indices = [u for u, v in initialEdges]
    for e in initialEdges:
        reachable = DFS(G, e[0], e[1], 4, 1)
        # print "from", e, " reachable:"
        for f in reachable:
            # print f
            M[indices[e[:2]]][indices[f[:2]]] = f[-1]
    return matrix(M)


def distToStationary(M):
    dist = 0
    u = Rational(1.0/M.ncols())
    # print "goal", u
    for r in M:
        for i in r:
            # if abs(i-u) > dist:
            #     print i
            dist = max(dist, abs(i-u))
    return dist


def stepsToEpsilon(n, epsilon):
    grid = genCentersGridGraph(n, n)
    grid = genDiGridFromCenters(grid)
    initPos, assocEdges = initialPositions(n, n, True)
    M = getMatrix(grid, assocEdges)
    i = 1
    X = deepcopy(M)
    dist = distToStationary(X)
    while dist > epsilon:
        X = X*M
        dist = distToStationary(X)
        i += 1
    return i


def dirGridMatrix(n):
    grid = genCentersGridGraph(n, n)
    grid = genDiGridFromCenters(grid)
    initPos, assocEdges = initialPositions(n, n, True)
    M = getMatrix(grid, assocEdges)
    return M
