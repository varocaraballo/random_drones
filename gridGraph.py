from fractions import Fraction as frac
import random
import itertools
import bisect
import pickle

CLOCKWISE = 0
COUNTERCLOCKWISE = 1


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
        else:                       # Synchronization point above
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
        # assert float(syncPointProb) == 1.0
    return G, w, paths


def randomWalk(n, m, k, osteps):
    steps = osteps
    diamondGraph, circles = getDiamondGraph(n, m)
    G, w, paths = getDronesGridGraph(n, m)

    edges = [(v, u) for v in diamondGraph for u in diamondGraph[v]]
    # print("edges", len(edges))
    assert len(edges) == n*m*4

    timeSinceLastCover = {(u, v): 0 for (u, v) in edges}
    maxUncoveredTime = {(u, v): -1 for (u, v) in edges}
    minUncoveredTime = {(u, v): float('inf') for (u, v) in edges}
    avgUncoveredTime = {(u, v): 0 for (u, v) in edges}
    totalUncoveredTime = {(u, v): 0 for (u, v) in edges}
    timesVisited = {(u, v): 0 for (u, v) in edges}

    timesCommunicated = {i: 0 for i in range(k)}
    timeSinceLastCom = {i: 0 for i in range(k)}
    maxTimeSinceLastCom = {i: -1 for i in range(k)}
    minTimeSinceLastCom = {i: float('inf') for i in range(k)}
    avgTimeSinceLastCom = {i: 0 for i in range(k)}
    totalUncomTime = {i: 0 for i in range(k)}

    currentPositions = random.sample(G.keys(), k=k)

    while steps:
        newPositions = [None for i in range(k)]
        pathsTaken = [None for i in range(k)]

        for i in range(k):
            u = currentPositions[i]
            # Python 3.5
            choices = list(G[u])
            weights = [w[(u, x)] for x in choices]
            cumdist = list(itertools.accumulate(weights))
            x = random.random() * cumdist[-1]
            v = choices[bisect.bisect(cumdist, x)]
            # Python 3.7:
            # weights = [w[(u, x)] for x in G[u]]
            # v = random.choices(G[u], weights=weights)[0]
            newPositions[i] = v
            pathsTaken[i] = paths[(u, v)]

        traversedEdges = set()

        #  Edge cover

        for path in pathsTaken:
            for i in range(len(path)-2):
                e = (path[i], path[i+1])
                traversedEdges.add(e)

        for e in edges:
            if e not in traversedEdges:
                timeSinceLastCover[e] += 1
            else:
                timesVisited[e] += 1
                aux = timesVisited[e]
                totalUncoveredTime[e] += timeSinceLastCover[e]
                maxUncoveredTime[e] = max(maxUncoveredTime[e], timeSinceLastCover[e])
                minUncoveredTime[e] = min(minUncoveredTime[e], timeSinceLastCover[e])
                avgUncoveredTime[e] = avgUncoveredTime[e]*((aux-1)/aux)+timeSinceLastCover[e]/aux
                timeSinceLastCover[e] = 0

        #  Communication

        communicatingDrones = set()

        for i in range(k):
            for j in range(i+1, k):
                if currentPositions[i] == currentPositions[j]:
                    if i not in communicatingDrones:
                        timesCommunicated[i] += 1
                        aux = timesCommunicated[i]
                        avgTimeSinceLastCom[i] = avgTimeSinceLastCom[i]*((aux-1)/aux)+timeSinceLastCom[i]/aux
                    if j not in communicatingDrones:
                        timesCommunicated[j] += 1
                        aux = timesCommunicated[j]
                        avgTimeSinceLastCom[j] = avgTimeSinceLastCom[j]*((aux-1)/aux)+timeSinceLastCom[j]/aux
                    communicatingDrones.add(i)
                    communicatingDrones.add(j)
                    totalUncomTime[i] += timeSinceLastCom[i]
                    totalUncomTime[j] += timeSinceLastCom[j]
                    maxTimeSinceLastCom[i] = max(maxTimeSinceLastCom[i], timeSinceLastCom[i])
                    minTimeSinceLastCom[i] = min(minTimeSinceLastCom[i], timeSinceLastCom[i])
                    maxTimeSinceLastCom[j] = max(maxTimeSinceLastCom[j], timeSinceLastCom[j])
                    minTimeSinceLastCom[j] = min(minTimeSinceLastCom[j], timeSinceLastCom[j])
                    timeSinceLastCom[i] = 0
                    timeSinceLastCom[j] = 0

        for drone in set(range(k)).difference(communicatingDrones):
            timeSinceLastCom[drone] += 1

        currentPositions = newPositions
        steps -= 1

    for e in totalUncoveredTime:
        totalUncoveredTime[e] += timeSinceLastCover[e]

    nEdges = len(edges)

    # print("Uncovered edges:", sum(1 for v in timesVisited.values() if v == 0))
    averageUncovered = sum(totalUncoveredTime.values())/nEdges
    # print("Average total time for uncovered edges:", averageUncovered)
    averageMaxUncovered = sum(maxUncoveredTime.values())/nEdges
    # print("Average max time for uncovered edges:", averageMaxUncovered)
    averageMinUncovered = sum(minUncoveredTime.values())/nEdges
    # print("Average min time for uncovered edges:", averageMinUncovered)
    averageAverageUncovered = sum(avgUncoveredTime.values())/nEdges
    # print("Average average time for uncovered edges:", averageAverageUncovered)
    # print("Proportion of time:", averageUncovered/osteps)

    # print("\nIsolated drones:", sum(1 for v in timesCommunicated.values() if v == 0))
    averageUncom = sum(totalUncomTime.values())/k
    # print("Average total time for isolated drones:", averageUncom)
    averageMaxUncom = sum(maxTimeSinceLastCom.values())/k
    # print("Average max time for isolated drones:", averageMaxUncom)
    averageMinUncom = sum(minTimeSinceLastCom.values())/k
    # print("Average min time for isolated drones:", averageMinUncom)
    averageAverageUncom = sum(avgTimeSinceLastCom.values())/k
    # print("Average average time for isolated drones:", averageAverageUncom)
    # print("Proportion of time:", averageUncom/osteps)

    # return (totalUncoveredTime, maxUncoveredTime, minUncoveredTime, avgUncoveredTime), (totalUncomTime, maxTimeSinceLastCom, minTimeSinceLastCom, avgTimeSinceLastCom), (timesVisited, timesCommunicated)
    return (averageUncovered, averageMaxUncovered, averageMinUncovered, averageAverageUncovered), (averageUncom, averageMaxUncom, averageMinUncom, averageAverageUncom)


def getCols(d):
    cols = {}
    maxVal = max(d.values())
    minVal = min(d.values())
    for k, v in d.items():
        col = -int((v-minVal)/(maxVal-minVal)*255)+255
        col = hex(col).split('x')[-1]
        while len(col) < 2:
            col = "0"+col
        cols[k] = "#ff"+col+col
    return cols


def simulate(n, m, steps):
    res = []
    total = n*m
    delta = int(n*m/100)
    for k in range(1, total+1, delta):
        res.append(randomWalk(n, m, k, steps))
    return res


def simulateK(n, m, repetitions, steps):
    totalSim = []
    for i in range(repetitions):
        print("repetition", i+1)
        res = []
        total = n*m
        delta = int(n*m/100)
        for k in range(1, total+1, delta):
            print("Random Walk with", k, "drones")
            res.append(randomWalk(n, m, k, steps))
        totalSim.append(res)
    avg = []
    # return totalSim
    for d in range(len(totalSim[0])):
        auxCover = zip(*[res[d][0] for res in totalSim])
        # print("auxCover", list(auxCover))
        # print("zip", list(zip(*auxCover)))
        avgCover = tuple(sum(x) / len(x) for x in auxCover)
        auxCom = zip(*[res[d][1] for res in totalSim])
        avgCom = tuple(sum(x) / len(x) for x in auxCom)
        avg.append((avgCover, avgCom))

    return totalSim, avg


def saveData(data, filename):
    f = open(filename, "wb")
    pickle.dump(data, f)
    f.close()
    print("Saved.")


def loadData(filename):
    f = open(filename, "rb")
    res = pickle.load(f)
    f.close()
    print("Loaded.")
    return res
