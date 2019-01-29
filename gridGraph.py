from fractions import Fraction as frac

CLOCKWISE = 0
COUNTERCLOCKWISE = 1


def getDiamondGraph(n, m):
    centers = getDiamondCenters(n, m)
    G = {}
    associatedCircles = {}
    cycle = [[0, 1], [1, 0], [0, -1], [-1, 0]]
    for x, y in centers.keys():
        direction = CLOCKWISE
        if ((x-1)/2) % 2 == ((y-1)/2) % 2:
            aux = cycle  # CW oriented
        else:
            aux = cycle[::-1]  # CCW oriented
            direction = COUNTERCLOCKWISE
        for i in range(4):
            dux, duy = aux[i]
            dvx, dvy = aux[(i+1) % 4]
            G.setdefault((x+dux, y+duy), []).append((x+dvx, y+dvy))
            associatedCircles[((x+dux, y+duy), (x+dvx, y+dvy))] = [(x, y), direction]
    return G, associatedCircles


def getDiamondCenters(n, m):
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
    return G


def dfs(G, v, pathLength=4):
    paths = []

    def DFS(u, currentPath):
        for w in G[u]:
            currentPath.append(w)
            if len(currentPath) < pathLength:
                DFS(w, currentPath)
            else:
                paths.append([v]+currentPath[:])
            currentPath.pop()

    DFS(v, [])
    return paths


def getPathProbability(G, path):
    p = frac(1, 1)
    for i in range(len(path)-1):
        p *= frac(1, len(G[path[i]]))
    return p


def getDronesGridGraph(n, m):
    '''Returns a triple G, w, paths, where G is the graph obtained in the 2pi model,
    w is a dictionary mapping each edge in G to its probability, and paths id a dictionary
    mapping each edge in G to the path in the circles model'''
    diamondGraph, associatedCircle = getDiamondGraph(n, m)
    G = {}
    w = {}
    paths = {}
    circles = set(tuple(x) for x in associatedCircle.values())
    # sync, edge = None, None
    for center, orientation in circles:
        sync, u = None, None
        if (center[1]+1) % 4 == 2:  # Sinchronization point below
            sync = (center[0], center[1]-1)
            if orientation == COUNTERCLOCKWISE:
                u = (sync[0]+1, sync[1]+1)
            else:  # orientation == CLOCKWISE
                u = (sync[0]-1, sync[1]+1)
        else:                     # Synchronization point above
            sync = (center[0], center[1]+1)
            if orientation == COUNTERCLOCKWISE:
                u = ((sync[0]-1, sync[1]-1))
            else:  # orientation == CLOCKWISE
                u = ((sync[0]+1, sync[1]-1))
        syncPointpaths = dfs(diamondGraph, u, 4)

        syncPointProb = frac(0, 1)
        for path in syncPointpaths:
            G.setdefault((sync, u), []).append((path[-2], path[-1]))
            p = getPathProbability(diamondGraph, path)
            syncPointProb += p
            w[((sync, u), (path[-2], path[-1]))] = p
            paths[((sync, u), (path[-2], path[-1]))] = [sync]+path
        assert float(syncPointProb) == 1.0
    return G, w, paths
