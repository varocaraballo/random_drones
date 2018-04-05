from sage.all import *


def get_neighbors(o):
	accesible = {
					(0,0):((-2,-2),(1,1)),
					(0,1):((-2,-1),(1,2)),
					(1,0):((-1,-2),(2,1)),
					(1,1):((-1,-1),(2,2))
				}
	(i,j) = o
	for r in range(accesible[(i % 2, j % 2)][0][0], accesible[(i % 2, j % 2)][1][0]+1):
		for c in range(accesible[(i % 2, j % 2)][0][1], accesible[(i % 2, j % 2)][1][1]+1):
			yield (r, c)


def get_transiton_matrix(n, adj=False):
	if n % 2 != 0:
		raise Exception('The given parameter must be even')
	M  = matrix(QQ,n**2)
	for i in range(n):
		for j in range(n):
			o = i*n+j
			for neighbor in get_neighbors((i,j)):
				(r,c) = neighbor
				d = ((i+r) % n)*n + ((j+c) % n)
				M[o,d] = 1 if adj else 1/16.0
	return M


def check_transition_matrix(M,adj=False):
	n =  M.nrows()
	for i in range(n):
		c = M.column(i)
		if (sum(c) != 1 and not adj) or (sum(c)!=16 and adj):
			return False
		c = M.row(i)
		if (sum(c) != 1 and not adj) or (sum(c)!=16 and adj):
			return False
	return True


def distance_to_uniform(M):
	n = M.nrows()
	nd = matrix(QQ,n,lambda i,j:1/float(n))
	D = nd - M
	d = 0
	for r in D:
		m = max(r)
		mi = abs(min(r))
		m = max(m,mi)
		if m>d:
			d = m
	return d


class LinkedNode:
	def __init__(self, cargo=None, nextNode=None):
		self.cargo = cargo
		self.nextNode = nextNode 

	def __str__(self):
		return str(self.cargo)

	def display(lst):
		if lst:
			w("%s " % lst)
			display(lst.nextNode)
		else:
			w("nil\n")


def round_paths_count(o,n,m=1000):
	current_dict = {o:1}	
	for i in range(n):
		next_dict = {}
		for k in current_dict:
			for nv in get_neighbors(k):
				neighbor = ((k[0]+nv[0]) % m, (k[1]+nv[1])%m)
				if neighbor in next_dict:
					next_dict[neighbor] += current_dict[k]
				else:
					next_dict[neighbor] = current_dict[k]
		current_dict = next_dict	
	return 0 if o not in current_dict else current_dict[o]


def paths_count(o,d,n):
	current_dict = {o:1}	
	for i in range(n):
		next_dict = {}
		for k in current_dict:
			for nv in get_neighbors(k):
				neighbor = (k[0]+nv[0], k[1]+nv[1])
				if neighbor in next_dict:
					next_dict[neighbor] += current_dict[k]
				else:
					next_dict[neighbor] = current_dict[k]
		current_dict = next_dict	
	return 0 if d not in current_dict else current_dict[d]


def getTorus9Matrix(n):
	M = [[0 for i in xrange(n**2)] for j in xrange(n**2)]
	indices = {}
	c = 0

	for i in xrange(n):
		for j in xrange(n):
			indices[(i, j)] = c
			c += 1

	for i in xrange(n):
		for j in xrange(n):
			directions = [1, 0, -1]
			for dx in directions:
				for dy in directions:
					if dx == dy and dy == 0:
						continue
					x = (i+dx)%n
					y = (j+dy)%n
					ind1 = indices[(i, j)]
					ind2 = indices[(x, y)]
					# print i, j, "to", x, y
					M[ind1][ind2] = 1
	return matrix(M)