
class nd_maze():
    def __init__(self, dimensions):
                self.dimensionality = len(dimensions)
                self.orth_directions = 2*self.dimensionality
                self.diag_directions = 3**self.dimensionality-1
                self.dimensions = dimensions

class matrix(nd_maze)
    def __init__(self, dimensions):
		self.maze = np.ones(dimensions)

class graph(nd_maze):
    def __init__(self, dimensions):
                self.maze = dict() 

def matrix_to_graph(matrix):
    graph = dict()
    for i, cell in enumerate(matrix):
        if cell == 0:
            graph(i) = getNeighbors(matrix)
