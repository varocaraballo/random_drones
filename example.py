import gridGraph as gs
import sys

if __name__ == '__main__':
    aux, n, m, k, steps = sys.argv
    uncovered, isolated, times = gs.randomWalk(10, 10, 10, 10000)
