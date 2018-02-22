import numpy as np
from enum import Enum
from random import *

class Square(Enum):
	PATH = 0
	WALL = 1
	ENTRANCE = 2
	EXIT = 3

class Coord():
	def __init__(self, x, y):
		self.x = x
		self.y = y

class Maze2D():
	def __init__(self, length, width):
		self.length = length
		self.width = width
		self.maze = np.ones([self.length, self.width])

	def reset(self):
		self.maze = np.ones([self.length, self.width])

	def draw(self):
		print("")
		for i in range(self.length):
			for j in range(self.width):
				if (self.maze[i][j] == 1):
					print("#", end="")
				elif (self.maze[i][j] == 2):
					print("S", end="")
				elif (self.maze[i][j] == 3):
					print("F", end="")
				else:
					print(" ",  end="")
			print("")

	def getAllCoordsOf(self, types):
		result = {}
		for i in range(self.length):
			for j in range(self.width):
				for the_type in types:
					if self.maze[i,j] == the_type:
						result.add(Coord(i,j))
		return result

	def hasCoord(self, the_set, coord):
		for c in the_set:
			if c.x == coord.x and c.y == coord.y:
				return True
		return False

	def findEntrance(self):
		for i in range(self.length):
			for j in range(self.width):
				if (self.maze[i][j] == 2):
					return Coord(i, j)
		return None

	def __allPathsAreReachableRecursive(self, coord, visitedPaths, allPaths, visited):
		if (coord == None):
			return False
		x = coord.x
		y = coord.y
		try:
			s = self.maze[x][y]
		except (Exception):
			return False
		if visited[x][y] == 1:
			return False
		elif self.maze[x][y] == 1:
			return False
		elif self.maze[x][y] == 0 or self.maze[x][y] == 2 or self.maze[x][y] == 3:
			visited[x][y] = 1
			visitedPaths.add(Coord(x,y))

		# forward
		self.__allPathsAreReachableRecursive(Coord(x+1, y), visitedPaths, allPaths, visited)

		# right
		self.__allPathsAreReachableRecursive(Coord(x, y+1), visitedPaths, allPaths, visited)

		# backward
		self.__allPathsAreReachableRecursive(Coord(x-1, y), visitedPaths, allPaths, visited)

		# left
		self.__allPathsAreReachableRecursive(Coord(x, y-1), visitedPaths, allPaths, visited)

		return visitedPaths.Equals(allPaths)

	def allPathsAreReachable(self):
		visited = np.zeros([self.length, self.width])

		allPaths = self.getAllCoordsOf([0, 2, 3])
		entrance = self.findEntrance()
		visitedPaths = {}
		return self.__allPathsAreReachableRecursive(entrance, visitedPaths, allPaths, visited)

	def allWallsAreReachable(self):
		pass

	def has(self, c, type):
		if (maze[c.x][c.y] == type):
			return True
		return False

	def __isSolvableRecursive(self, sq, visited):
		if (sq == None):
			return False
		x = sq.x
		y = sq.y
		try:
			s = self.maze[x][y]
		except Exception:
			return False
		if self.maze[x][y] == 1:
			return False
		elif visited[x][y] == 1:
			return False
		elif self.maze[x][y] == 3:
			return True

		visited[x][y] = 1

		# forward
		return (self.__isSolvableRecursive(Coord(x+1,y), visited)) or (self.__isSolvableRecursive(Coord(x,y+1), visited)) or (self.__isSolvableRecursive(Coord(x-1,y), visited)) or (self.__isSolvableRecursive(Coord(x,y-1), visited))

	def isSolvable(self):
		entrance = self.findEntrance()
		visited = np.zeros([self.length, self.width])
		return self.__isSolvableRecursive(entrance, visited)

	def getBorderSquare(self):
		x = 0
		y = 0
		while (x == 0 and (y == 0 or y == self.width-1)) or (x == self.length-1 and (y == 0 or y == self.width-1)):
			x = randint(0,self.length-1)
			y = randint(0,self.width-1)
			side = randint(0,3)
			if side == 0:
				x = 0
			elif side == 1:
				x = self.length - 1
			elif side == 2:
				y = 0
			elif side == 3:
				y = self.width - 1
		return Coord(x,y)

	def addEntrance(self):
		c = self.getBorderSquare()
		while (self.maze[c.x][c.y] == 2 or self.maze[c.x][c.y] == 3):
			c = self.getBorderSquare()
		self.maze[c.x][c.y] = 2

	def addExit(self):
		c = self.getBorderSquare()
		while (self.maze[c.x][c.y] == 2 or self.maze[c.x][c.y] == 3):
			c = self.getBorderSquare()
		self.maze[c.x][c.y] = 3

	def rndMaze(self):
		for i in range(self.length):
			for j in range(self.width):
				r = randint(0,1)
				if r == 0:
					self.maze[i][j] = 0
		self.addEntrance()
		self.addExit()

	def solvableRndMaze(self):
		self.rndMaze()
		while ((not self.isSolvable()) or (not self.allPathsAreReachable)):
			self.reset()
			self.rndMaze()

	def fillBorders(self):
		self.maze[0, :] = self.maze[-1, :] = 1
		self.maze[:, 0] = self.maze[:, -1] = 1

	def densityIsland(self, complexity=.5, density=.95):
		# Only odd shapes
		shape = ((self.length // 2) * 2 + 1, (self.width // 2) * 2 + 1)
		# Adjust complexity and density relative to maze size
		complexity = int(complexity * (5 * (shape[0] + shape[1])))
		density	= int(density * ((shape[0] // 2) * (shape[1] // 2)))
		# Build actual maze
		self.maze = np.zeros(shape, dtype=bool)
		# Fill borders
		self.maze[0, :] = self.maze[-1, :] = 1
		self.maze[:, 0] = self.maze[:, -1] = 1
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

class Prim(Maze2D):
	def __init__(self, length, width):
		super().__init__(length, width)

	def addNeighborWalls(walls, center):
		neighbors = get4Neighbors(center)
		result = []
		for c in neighbors:
			if (maze[c.x, c.y] == Square.WALL and not hasCoord(walls, c)):
				result.add(c)

		walls.addRange(result)
		return walls

	def get8Neighbors(self, center):
		x = center.x
		y = center.y
		result = []
		# N
		if (y < self.width - 1):
			result.add(Coord(x, y+1))
		# NE
		if (y < self.width - 1 and x < self.length - 1):
			result.add(Coord(x+1, y+1))
		# E
		if (x < self.length - 1):
			result.add(Coord(x+1, y))
		# SE
		if (y > 0 and x < self.length - 1):
			result.add(Coord(x+1, y-1))
		# S
		if (y > 0):
			result.add(Coord(x, y-1))
		# SW
		if (y > 0 and x > 0):
			result.add(Coord(x-1, y-1))
		# W
		if (x > 0):
			result.add(Coord(x-1, y))
		# NW
		if (y < self.width - 1 and x > 0):
			result.add(Coord(x-1, y+1))
		return result

	def get4Neighbors(self, center):
		x = center.x
		y = center.y
		result = []
		# N
		if (y < self.width - 1):
			result.add(Coord(x, y+1))
		# E
		if (x < self.length - 1):
			result.add(Coord(x+1, y))
		# S
		if (y > 0):
			result.add(Coord(x, y-1))
		# W
		if (x > 0):
			result.add(Coord(x-1, y))
		return result

	def only1Visited(self, neighbors, visited):
		matches = 0
		for c1 in neighbors:
			# 			if (c1.x == 0 or c1.y == 0 or c1.x == self.length - 1 or c1.y == self.width - 1) return false
			for c2 in visited:
				if (c1.x == c2.x and c1.y == c2.y):
					matches += 1

		if (matches == 1):
			return true
		return false

	def getPassageCoord(self, center):
		result = None
		neighbors = get4Neighbors(center)
		for path in neighbors:
			if (maze[path.x,path.y] == Square.PATH):
				if (path.x == center.x):
					if path.y > center.y and center.y - 1 > 0:
						result = Coord(path.x, center.y - 1)

					elif center.y + 1 < self.width-1:
						result = Coord(path.x, center.y + 1)

				else:
					if (path.x > center.x and center.x-1 > 0):
						result = Coord(center.x-1,path.y)

					elif (center.x+1 < self.length-1):
						return Coord (center.x+1, path.y)

		if (result == None or result.x == 0 or result.y == 0 or result.x >= self.length-1 or result.y >= self.width-1):
			return None

		return result

	def prim(self):
		initFull()
		walls = []
		visited = []
		start = Coord(4,4)
		maze[start.x][start.y] = 0
		visited.add(start)
		walls = addNeighborWalls(walls, start)
		#Pick a cell, mark it as part of the maze. add the walls of the cell to the wall list.
		while(len(walls) > 0):
			#Pick a random wall from the list.
			wall_num = randint(0,len(walls)-1)
			rnd_wall = walls[wall_num]
			neighbors = get4Neighbors(rnd_wall)
			if (only1Visited(neighbors, visited)):#If only one of the two cells that the wall divides is visited, then:
				#Make the wall a passage and mark the unvisited cell as part of the maze.
				passage = getPassageCoord(rnd_wall)
				if (passage == None):
					del walls[wall_num]
					continue
				visited.add(passage)
				maze[passage.x][passage.y] = 0
				visited.add(rnd_wall)
				maze[rnd_wall.x][rnd_wall.y] = 0

				#add the neighboring walls of the cell to the wall list.
				walls = addNeighborWalls(walls, passage)
				#				walls = addNeighborWalls(walls, rnd_wall)
			#Remove the wall from the list.
			del walls[wall_num]


maze = Maze2D(11,21)
maze.densityIsland()
# maze.rndMaze()

# maze = Prim(11,21)

maze.draw()
