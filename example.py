import gridGraph as gs
import sys

if __name__ == '__main__':
    n, m, k, steps = map(int, sys.argv[1:])
    uncovered, isolated, times = gs.randomWalk(n, m, k, steps)
