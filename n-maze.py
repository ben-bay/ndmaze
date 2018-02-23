import numpy as np
from enum import Enum
from random import *

class n_maze():
	def __init__(self, dimensions):
		self.dimensionality = len(dimensions)
		self.orth_directions = 2*self.dimensionality
		self.diag_directions = 3**self.dimensionality-1
		self.dimensions = dimensions
		self.maze = np.ones(dimensions)

	def draw(self):
		if self.dimensionality == 3:
			for k in range(self.dimensions[2]):
				print("\nheight {}".format(k))
				for i in range(self.dimensions[0]):
					for j in range(self.dimensions[1]):
						if (self.maze[i,j,k] == 1):
							print("#", end="")
						elif (self.maze[i][j][k] == 2):
							print("S", end="")
						elif (self.maze[i][j][k] == 3):
							print("F", end="")
						else:
							print(" ",  end="")
					print("")
		elif self.dimensionality != 2:
			print("Can only display 2D and 3D mazes in terminal.")
		else:
			print("")
			for i in range(self.dimensions[0]):
				for j in range(self.dimensions[1]):
					if (self.maze[i][j] == 1):
						print("#", end="")
					elif (self.maze[i][j] == 2):
						print("S", end="")
					elif (self.maze[i][j] == 3):
						print("F", end="")
					else:
						print(" ",  end="")
				print("")

	def fillBorders(self):
		for index, val in np.ndenumerate(self.maze):
			for i, dim in enumerate(index):
				if dim == self.dimensions[i] - 1 or dim == 0 :
					self.maze[index] = 1

	def densityIsland(self, complexity=.5, density=.95): #TODO fix for n dimensions
		# Only odd shapes
		shape = ((self.dimensions[0] // 2) * 2 + 1, (self.dimensions[1] // 2) * 2 + 1)
		# Adjust complexity and density relative to maze size
		complexity = int(complexity * (5 * (shape[0] + shape[1])))
		density	= int(density * ((shape[0] // 2) * (shape[1] // 2)))
		# Build actual maze
		self.maze = np.zeros(shape, dtype=bool)
		# Fill borders
		self.fillBorders()
		# Make aisles
		for i in range(density):
			x, y = randint(0, shape[1] // 2) * 2, randint(0, shape[0] // 2) * 2
			self.maze[y, x] = 1
			for j in range(complexity):
				neighbours = []
				if x > 1:
					neighbours.append((y, x - 2))
				if x < shape[1] - 2:
					neighbours.append((y, x + 2))
				if y > 1:
					neighbours.append((y - 2, x))
				if y < shape[0] - 2:
					neighbours.append((y + 2, x))
				if len(neighbours):
					y_,x_ = neighbours[randint(0, len(neighbours) - 1)]
					if self.maze[y_, x_] == 0:
						self.maze[y_, x_] = 1
						self.maze[y_ + (y - y_) // 2, x_ + (x - x_) // 2] = 1
						x, y = x_, y_

	def monteCarloMaze(self, density=.1):
		for index, x in np.ndenumerate(self.maze):
			rnd = randint(0,100)
			if rnd < 100*density:
				self.maze[index] = 1
			else:
				self.maze[index] = 0

	def reset(self, fill=1):
		if fill==1:
			self.maze = np.ones(dimensions)
		else:
			self.maze = np.zeros(dimensions)

	def getAllBorderCells(self):
		border_cells = []
		for index, val in np.ndenumerate(self.maze):
			for i, dim in enumerate(index):
				if dim == self.dimensions[i] - 1 or dim == 0 :
					border_cells.append(index)
		return border_cells

	def getBorderCell(self):
		border_cells = self.getAllBorderCells()
		rnd = randint(0,len(border_cells)-1)
		return border_cells[rnd]

	def addEntrance(self):
		c = self.getBorderCell()
		while (self.maze[c] == 2 or self.maze[c] == 3):
			c = self.getBorderSquare()
		self.maze[c] = 2

	def addExit(self):
		c = self.getBorderCell()
		while (self.maze[c] == 2 or self.maze[c] == 3):
			c = self.getBorderCell()
		self.maze[c] = 3

	def addEntranceAndExit(self):
		self.addEntrance()
		self.addExit()

	def excludeExteriors(self, indexes):
		result = []
		for cell_index in indexes:
			cell_is_exterior = False
			for i, index_num in enumerate(cell_index):
				if index_num < 0 or index_num >= self.dimensions[i]:
					cell_is_exterior = True
					break
			if cell_is_exterior == False:
				result.append(cell_index)
		return result

	def getOrthogonalNeighbors(self, cell_index):
		orthogonal_directions = []
		for i, element in enumerate(cell_index):
			temp1 = list(cell_index)
			temp1[i] = element + 1
			temp2 = list(cell_index)
			temp2[i] = element - 1
			orthogonal_directions.append(tuple(temp1))
			orthogonal_directions.append(tuple(temp2))
		return self.excludeExteriors(orthogonal_directions)

	def getAdjacentNeighbors(self, cell_index):
		pass

	def __isSolvableRecursive(self, cell_index, visited):
		if (cell_index == None or len(cell_index) != self.dimensionality):
			return False
		if self.maze[cell_index] == 1:
			return False
		elif visited[cell_index] == 1:
			return False
		elif self.maze[cell_index] == 3:
			return True

		visited[cell_index] = 1

		#recurse to orthogonal neighbors
		result = False
		for orth_index in self.getOrthogonalNeighbors(cell_index):
			# print("orth_index: {}".format(orth_index))
			result = (result or self.__isSolvableRecursive(orth_index, visited))
			if result == True:
				break
		return result

	def findEntrance(self):
		for index, val in np.ndenumerate(self.maze):
			if val == 2:
				return index

	def isSolvable(self):
		entrance = self.findEntrance()
		visited = np.zeros(self.dimensions)
		return self.__isSolvableRecursive(entrance, visited)

	def __allAreConnectedRecursive(self, cell_index, visitedPaths, allPaths, visited, cell_type):
		cell_pos = [1,2,3]
		cell_neg = [0]
		if cell_type == "path":
			cell_pos = [0,2,3]
			cell_neg = [1]
		if (cell_index == None  or len(cell_index) != self.dimensionality):
			return False
		if visited[cell_index] == 1:
			return False
		elif self.maze[cell_index] in cell_neg:
			return False
		elif self.maze[cell_index] in cell_pos:
			visited[cell_index] = 1
			visitedPaths.append(cell_index)

		for orth_index in self.getOrthogonalNeighbors(cell_index):
			self.__allAreConnectedRecursive(orth_index, visitedPaths, allPaths, visited, cell_type)

		return (set(visitedPaths) == set(allPaths))

	def getAllCoordsOf(self, types):
		result = []
		for index, val in np.ndenumerate(self.maze):
			if self.maze[index] in types:
				result.append(index)
		return result

	def allAreConnected(self, cell_type="path"):
		visited = np.zeros(self.dimensions)
		cell_pos = [1,2,3]
		if cell_type == "path":
			cell_pos = [0,2,3]
		allPos = self.getAllCoordsOf(cell_pos)
		entrance = self.findEntrance()
		visitedPaths = []
		return self.__allAreConnectedRecursive(entrance, visitedPaths, allPos, visited, cell_type)

	def printSolvability(self):
		if self.isSolvable():
			print("Solvable!")
		else:
			print("NOT solvable!")

	def printConnectedness(self):
		if self.allAreConnected("path"):
			print("All paths are connected!")
		else:
			print("NOT all paths are connected!")
		if self.allAreConnected("wall"):
			print("All walls are connected!")
		else:
			print("NOT all walls are connected!")

	def getDensity(self):
		cell = 0
		wall = 0
		for index, val in np.ndenumerate(self.maze):
			if val == 1:
				wall += 1
			cell += 1
		return float(wall) / float(cell)

	def printDensity(self):
		print("Density: {}".format(self.getDensity()))

	def bruteForceMonteCarlo(self, low_threshold=0, high_threshold=1, paths_connect = True, walls_connect = True):
		self.monteCarloMaze()
		self.fillBorders()
		self.addEntranceAndExit()
		while self.isSolvable() == False or (self.getDensity() < low_threshold or self.getDensity() > high_threshold) or self.allAreConnected("path") == False or self.allAreConnected("wall") == False:
			self.monteCarloMaze()
			self.fillBorders()
			self.addEntranceAndExit()

	def printMazeInfo(self):
		self.draw()
		self.printSolvability()
		self.printConnectedness()
		self.printDensity()

maze = n_maze([7, 15, 5])
# maze.densityIsland()
# maze.monteCarloMaze(.3)
maze.bruteForceMonteCarlo()
maze.printMazeInfo()
