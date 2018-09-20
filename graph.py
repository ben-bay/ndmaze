from collections import defaultdict

class graph(defaultdict):
	pass


def ndarray_to_graph(arr):
	graph = defaultdict(list)
	for i, square in enumerate(arr):
		get_blank_neighbors = {...}
		if square == blank:
			graph[i] = {blank_neighbors}
	return graph

def graph_to_ndarray(graph):
	pass

"""     
#####   O   O
#.#.#   |   |
#...#   O-O-O
#.#.#   |   |
#####   O   O
"""
