
import numpy as np
from enum import Enum
from random import randint
import argparse
import sys

import maze_utils










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
                            print(".",  end="")
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
                        print(".",  end="")
                print("")

    def getAllCells(self):
        return maze_utils.getAllCells(self)

    def fillBorders(self):
        maze_utils.fillBorders(self)

    def addEntrance(self):
        maze_utils.addEntrance(self)

    def addExit(self):
        maze_utils.addExit(self)

    def addEntranceAndExit(self):
        self.addEntrance()
        self.addExit()

    def primMaze(self):
        maze_utils.primMaze(self)

    def getPassage(self, path, wall):
        return maze_utils.getPassage(self, path, wall)

    def only1OrthVisited(self, neighbors, visited):
        return maze_utils.only1OrthVisited(self, neighbors, visited)

    def densityIsland(self, complexity=.5, density=.95): #TODO fix for n dimensions
        # Assumes odd shape
        # Adjust complexity and density relative to maze size
        complexity = int(complexity * (5 * (sum(self.dimensions))))
        product = 1
        for dim in self.dimensions:
            product *= (dim // 2)
        density    = int(density * product)
        # Build actual maze
        self.maze = np.zeros(self.dimensions)
        # Fill borders
        self.fillBorders()
        # Make aisles
        for i in range(density):
            variables = []
            for x in reversed(self.dimensions):
                variables.append(randint(0, x // 2) * 2)
            print(variables)
            print(self.maze)
            # x, y = randint(0, shape[1] // 2) * 2, randint(0, shape[0] // 2) * 2
            self.maze[list(reversed(variables))] = 1
            print(self.maze)
            # self.maze[y,x] = 1
            for j in range(complexity):
                neighbors = []
                for index, k in enumerate(variables):
                    if k > 1:
                        temp = list(reversed(variables))
                        temp[self.dimensionality-1-index] = temp[self.dimensionality-1-index]-2
                        neighbors.append(temp)
                    if k < self.dimensions[self.dimensionality-1-index] - 2:
                        temp = list(reversed(variables))
                        temp[self.dimensionality-1-index] = temp[self.dimensionality-1-index]+2
                        neighbors.append(temp)
                    if len(neighbors):
                        print(neighbors)
                        variables2 = neighbors[randint(0, len(neighbors) - 1)]
                        print(variables2)
                        print(self.maze)
                        print(self.maze[variables2])
                        if (self.maze[variables2] == 0):
                            self.maze[variables2] = 1
                            # self.maze[y_ + (y - y_) // 2, x_ + (x - x_) // 2] = 1
                            variables3 = []
                            for index, x in enumerate(variables):
                                variables3[index] = variables2[index] + (variables[self.dimensionality-1-index] - variables2[index]) // 2
                            self.maze[variables3] = 1
                            variables = reversed(variables2)
        self.addEntranceAndExit()

    def densityIsland2D(self, complexity=.5, density=.95):
        # Only odd shapes
        shape = ((self.dimensions[0] // 2) * 2 + 1, (self.dimensions[1] // 2) * 2 + 1)
        # Adjust complexity and density relative to maze size
        complexity = int(complexity * (5 * (shape[0] + shape[1])))
        density    = int(density * ((shape[0] // 2) * (shape[1] // 2)))
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

    def monteCarloMaze(self, density=.1):
        for index, x in np.ndenumerate(self.maze):
            rnd = randint(0,0)
            if rnd < 0*density:
                self.maze[index] = 1
            else:
                self.maze[index] = 0

    def getAllBorderCells(self):
        border_cells = []
        for index, val in np.ndenumerate(self.maze):
            for i, dim in enumerate(index):
                if dim == self.dimensions[i] - 1 or dim == 0 :
                    border_cells.append(index)
        return border_cells

    def getAllOddBorderBorderCells(self):
        border_cells = []
        for index, val in np.ndenumerate(self.maze):
            good = True
            for i, dim in enumerate(index):
                if (dim % 2 == 0):
                    good == False
                    break
            if good == False:
                continue
            for i, dim in enumerate(index):
                if (dim == self.dimensions[i] - 2 or dim == 1) and (0 not in index and self.dimensions[i] - 1 not in index):
                    border_cells.append(index)
        return border_cells

    def excludeCorners(self, indexes):
        result = []
        for cell_index in indexes:
            cell_is_corner = True
            for i, index_num in enumerate(cell_index):
                if index_num != 0 and index_num != self.dimensions[i] - 1:
                    cell_is_corner = False
                    break
            if cell_is_corner == False:
                result.append(cell_index)
        return result

    def rndBorderCell(self):
        border_cells = self.getAllBorderCells()
        no_corners = self.excludeCorners(border_cells)
        rnd = randint(0,len(no_corners)-1)
        return no_corners[rnd]

    def rndBorderBorderCell(self):
        border_border_cells = self.getAllOddBorderBorderCells()
        rnd = randint(0,len(border_border_cells)-1)
        return border_border_cells[rnd]

    def checkerboardMaze(self):
        for index, x in np.ndenumerate(self.maze):
            if sum(index) % 2 == 0:
                self.maze[index] = 1
            else:
                self.maze[index] = 0

    def getAllCoordsOf(self, types):
        result = []
        for index, val in np.ndenumerate(self.maze):
            if self.maze[index] in types:
                result.append(index)
            return result

        def bruteForceStep(self, low_threshold=0, high_threshold=1, paths_connect = True, walls_connect = True):
            self.checkerboardMaze()
            self.fillBorders()
            self.addEntranceAndExit()
            cell_indexes = set(self.getAllCoordsOf([0,1])) - set(self.getAllBorderCells())
            cell_indexes = list(cell_indexes)
            while self.isSolvable() == False: #or (self.getDensity() < low_threshold or self.getDensity() > high_threshold) or self.allAreConnected("path") == False or self.allAreConnected("wall") == False:
                rnd = randint(0, len(cell_indexes)-1)
                if self.maze[cell_indexes[rnd]] == 1:
                    self.maze[cell_indexes[rnd]] = 0
                else:
                    self.maze[cell_indexes[rnd]] = 1

        def reset(self, fill=1):
            if fill==1:
                self.maze = np.ones(dimensions)
            else:
                self.maze = np.zeros(dimensions)

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

    # def bruteForceAssignEntraceAndExit(self, qualifications):
    #     bad = True
    #     while bad == True:
    #         self.fillBorders()
    #         self.addEntranceAndExit()
    #         for qualification in qualifications:
    #             bad = bad and (not qualification.__call__)

    def bruteForceAssignEntraceAndExit(self):
        self.fillBorders()
        self.addEntranceAndExit()
        while self.isSolvable() == False:
            self.fillBorders()
            self.addEntranceAndExit()

    def printMazeInfo(self):
        self.printSolvability()
        self.printConnectedness()
        self.printDensity()

    def __DijkstrasShortestPath(self, cell_index, visited, length):
        if (cell_index == None or len(cell_index) != self.dimensionality):
            return False
        if self.maze[cell_index] == 1:
            return False
        elif visited[cell_index] == 1:
            return False
        elif self.maze[cell_index] == 3:
            return True

        length += 1
        visited[cell_index] = 1

        #recurse to orthogonal neighbors
        result = False
        for orth_index in self.getOrthogonalNeighbors(cell_index):
            result = (result or self.__isSolvableRecursive(orth_index, visited))
            if result == True:
                break
        return result

def matrix_to_graph(matrix):
    graph = dict()
    for i, cell in enumerate(matrix):
        if cell == 0:
            graph[i] = matrix.getOrthogonalNeighbors()

#! /usr/bin/env python3 

import numpy as np
from enum import Enum
from random import randint
import argparse

def draw(maze):
    if maze.dimensionality == 3:
        for k in range(maze.dimensions[2]):
            print("\nheight {}".format(k))
            for i in range(maze.dimensions[0]):
                for j in range(maze.dimensions[1]):
                    if (maze.maze[i,j,k] == 1):
                        print("#", end="")
                    elif (maze.maze[i][j][k] == 2):
                        print("S", end="")
                    elif (maze.maze[i][j][k] == 3):
                        print("F", end="")
                    else:
                        print(".",  end="")
                print("")
    elif maze.dimensionality != 2:
        print("Can only display 2D and 3D mazes in terminal.")
    else:
        print("")
        for i in range(maze.dimensions[0]):
            for j in range(maze.dimensions[1]):
                if (maze.maze[i][j] == 1):
                    print("#", end="")
                elif (maze.maze[i][j] == 2):
                    print("S", end="")
                elif (maze.maze[i][j] == 3):
                    print("F", end="")
                else:
                    print(".",  end="")
            print("")

def getAllCells(maze):
    result = []
    for index, val in np.ndenumerate(maze.maze):
        result.append(index)
    return result

def fillBorders(maze):
    for index, val in np.ndenumerate(maze.maze):
        for i, dim in enumerate(index):
            if dim == maze.dimensions[i] - 1 or dim == 0 :
                maze.maze[index] = 1

def addEntrance(maze):
    c = maze.rndBorderCell()
    while (maze.maze[c] == 2 or maze.maze[c] == 3):
        c = maze.rndBorderCell()
    maze.maze[c] = 2

def addExit(maze):
    c = maze.rndBorderCell()
    while (maze.maze[c] == 2 or maze.maze[c] == 3):
        c = maze.rndBorderCell()
    maze.maze[c] = 3

def addEntranceAndExit(maze):
    maze.addEntrance()
    maze.addExit()

def primMaze(maze):
    walls = []
    visited = []
    start = maze.rndBorderBorderCell()
    print(start)
    maze.maze[start] = 0
    visited.append(start)
    neighbor_walls = maze.getOrthogonalNeighbors(start)
    walls += neighbor_walls
    # Pick a cell, mark it as part of the maze. Add the walls of the cell to the wall list.
    while len(walls) > 0:
        # print("")
        # print(str(len(walls))+" walls")
        # Pick a random wall from the list.
        wall_num = randint(0,len(walls)-1)
        rnd_wall = walls[wall_num]
        neighbor_walls = maze.getOrthogonalNeighbors(rnd_wall)
        only_path = maze.only1OrthVisited(neighbor_walls, visited)
        # If only one of the two cells that the wall divides is visited, then:
        if (only_path != -1 and rnd_wall not in maze.getAllBorderCells()):
            # print("Only 1 visited path!")
            # print("wall: "+str(rnd_wall))
            # Make the wall a passage and mark the unvisited cell as part of the maze.
            passage = maze.getPassage(only_path, rnd_wall)
            # if (passage == None):
            #     print("Passage is null!")
            #     del walls[wall_num]
            #     continue
            # print("Passage NOT null!")
            if (passage != None):
                visited.append(passage)
                maze.maze[passage] = 0
                # Add the neighboring walls of the cell to the wall list.
                neighbor_walls = maze.getOrthogonalNeighbors(passage)
                unique_walls = set(neighbor_walls) - set(walls)
                # print("unique_walls length: "+str(len(unique_walls)))
                walls += list(unique_walls)
            visited.append(rnd_wall)
            maze.maze[rnd_wall] = 0
            # Remove the wall from the list.
            del walls[wall_num]

def getPassage(maze, path, wall):
    result = None
    neighbors = maze.getOrthogonalNeighbors(wall)
    diff_dim = -1
    direction = 0
    for i in range(maze.dimensionality):
        if path[i] != wall[i]:
            diff_dim = i
            direction = wall[i] - path[i]
    result = list(path)
    result[diff_dim] = result[diff_dim]+(direction*2)
    # print("passage: " + str(result))

    result = tuple(result)

    if (result in maze.getAllBorderCells()) or (result not in maze.getAllCells()):
    # if (result not in maze.getAllCells()):
        # print("passage rejected")
        return None
    return result

def only1OrthVisited(maze, neighbors, visited):
    matches = 0
    the_path = []
    for index1 in neighbors:
        # if (c1.x == 0 or c1.y == 0 or c1.x == length - 1 or c1.y == width - 1) return false
        for index2 in visited:
            if (index1 == index2):
                matches += 1
                the_path = index1
                if matches > 1:
                    break
    if matches == 1:
        return the_path
    return -1

def densityIsland(maze, complexity=.5, density=.95): #TODO fix for n dimensions
    # Assumes odd shape
    # Adjust complexity and density relative to maze size
    complexity = int(complexity * (5 * (sum(maze.dimensions))))
    product = 1
    for dim in maze.dimensions:
        product *= (dim // 2)
    density    = int(density * product)
    # Build actual maze
    maze.maze = np.zeros(maze.dimensions)
    # Fill borders
    maze.fillBorders()
    # Make aisles
    for i in range(density):
        variables = []
        for x in reversed(maze.dimensions):
            variables.append(randint(0, x // 2) * 2)
            print(variables)
            print(maze.maze)
            # x, y = randint(0, shape[1] // 2) * 2, randint(0, shape[0] // 2) * 2
            maze.maze[list(reversed(variables))] = 1
            print(maze.maze)
            # maze.maze[y,x] = 1
            for j in range(complexity):
                neighbors = []
                for index, k in enumerate(variables):
                    if k > 1:
                        temp = list(reversed(variables))
                        temp[maze.dimensionality-1-index] = temp[maze.dimensionality-1-index]-2
                        neighbors.append(temp)
                    if k < maze.dimensions[maze.dimensionality-1-index] - 2:
                        temp = list(reversed(variables))
                        temp[maze.dimensionality-1-index] = temp[maze.dimensionality-1-index]+2
                        neighbors.append(temp)
                    if len(neighbors):
                        print(neighbors)
                        variables2 = neighbors[randint(0, len(neighbors) - 1)]
                        print(variables2)
                        print(maze.maze)
                        print(maze.maze[variables2])
                    if (maze.maze[variables2] == 0):
                        maze.maze[variables2] = 1
                        # maze.maze[y_ + (y - y_) // 2, x_ + (x - x_) // 2] = 1
                        variables3 = []
                        for index, x in enumerate(variables):
                            variables3[index] = variables2[index] + (variables[maze.dimensionality-1-index] - variables2[index]) // 2
                            maze.maze[variables3] = 1
                            variables = reversed(variables2)
    maze.addEntranceAndExit()

def densityIsland2D(maze, complexity=.5, density=.95):
    # Only odd shapes
    shape = ((maze.dimensions[0] // 2) * 2 + 1, (maze.dimensions[1] // 2) * 2 + 1)
    # Adjust complexity and density relative to maze size
    complexity = int(complexity * (5 * (shape[0] + shape[1])))
    density    = int(density * ((shape[0] // 2) * (shape[1] // 2)))
    # Build actual maze
    maze.maze = np.zeros(shape, dtype=bool)
    # Fill borders
    maze.maze[0, :] = maze.maze[-1, :] = 1
    maze.maze[:, 0] = maze.maze[:, -1] = 1
    # Make aisles
    for i in range(density):
        x, y = randint(0, shape[1] // 2) * 2, randint(0, shape[0] // 2) * 2
        maze.maze[y, x] = 1
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
                if maze.maze[y_, x_] == 0:
                    maze.maze[y_, x_] = 1
                    maze.maze[y_ + (y - y_) // 2, x_ + (x - x_) // 2] = 1
                    x, y = x_, y_

def monteCarloMaze(maze, density=.1):
    for index, x in np.ndenumerate(maze.maze):
        rnd = randint(0,0)
        if rnd < 0*density:
            maze.maze[index] = 1
        else:
            maze.maze[index] = 0

def getAllBorderCells(maze):
    border_cells = []
    for index, val in np.ndenumerate(maze.maze):
        for i, dim in enumerate(index):
            if dim == maze.dimensions[i] - 1 or dim == 0 :
                border_cells.append(index)
    return border_cells

def getAllOddBorderBorderCells(maze):
    border_cells = []
    for index, val in np.ndenumerate(maze.maze):
        good = True
        for i, dim in enumerate(index):
            if (dim % 2 == 0):
                good == False
                break
            if good == False:
                continue
        for i, dim in enumerate(index):
            if (dim == maze.dimensions[i] - 2 or dim == 1) and (0 not in index and maze.dimensions[i] - 1 not in index):
                border_cells.append(index)
    return border_cells

def excludeCorners(maze, indexes):
    result = []
    for cell_index in indexes:
        cell_is_corner = True
        for i, index_num in enumerate(cell_index):
            if index_num != 0 and index_num != maze.dimensions[i] - 1:
                cell_is_corner = False
                break
            if cell_is_corner == False:
                result.append(cell_index)
    return result

def rndBorderCell(maze):
    border_cells = maze.getAllBorderCells()
    no_corners = maze.excludeCorners(border_cells)
    rnd = randint(0,len(no_corners)-1)
    return no_corners[rnd]

def rndBorderBorderCell(maze):
    border_border_cells = maze.getAllOddBorderBorderCells()
    rnd = randint(0,len(border_border_cells)-1)
    return border_border_cells[rnd]

def checkerboardMaze(maze):
    for index, x in np.ndenumerate(maze.maze):
        if sum(index) % 2 == 0:
            maze.maze[index] = 1
        else:
            maze.maze[index] = 0

def getAllCoordsOf(maze, types):
    result = []
    for index, val in np.ndenumerate(maze.maze):
        if maze.maze[index] in types:
            result.append(index)
        return result

def bruteForceStep(maze, low_threshold=0, high_threshold=1, paths_connect = True, walls_connect = True):
        maze.checkerboardMaze()
        maze.fillBorders()
        maze.addEntranceAndExit()
        cell_indexes = set(maze.getAllCoordsOf([0,1])) - set(maze.getAllBorderCells())
        cell_indexes = list(cell_indexes)
        while maze.isSolvable() == False: #or (maze.getDensity() < low_threshold or maze.getDensity() > high_threshold) or maze.allAreConnected("path") == False or maze.allAreConnected("wall") == False:
            rnd = randint(0, len(cell_indexes)-1)
            if maze.maze[cell_indexes[rnd]] == 1:
                maze.maze[cell_indexes[rnd]] = 0
            else:
                maze.maze[cell_indexes[rnd]] = 1

def reset(maze, fill=1):
        if fill==1:
            maze.maze = np.ones(dimensions)
        else:
            maze.maze = np.zeros(dimensions)

def excludeExteriors(maze, indexes):
        result = []
        for cell_index in indexes:
            cell_is_exterior = False
            for i, index_num in enumerate(cell_index):
                if index_num < 0 or index_num >= maze.dimensions[i]:
                    cell_is_exterior = True
                    break
            if cell_is_exterior == False:
                result.append(cell_index)
        return result

def getOrthogonalNeighbors(maze, cell_index):
        orthogonal_directions = []
        for i, element in enumerate(cell_index):
            temp1 = list(cell_index)
            temp1[i] = element + 1
            temp2 = list(cell_index)
            temp2[i] = element - 1
            orthogonal_directions.append(tuple(temp1))
            orthogonal_directions.append(tuple(temp2))
        return maze.excludeExteriors(orthogonal_directions)

def getAdjacentNeighbors(maze, cell_index):
    pass

def __isSolvableRecursive(maze, cell_index, visited):
    if (cell_index == None or len(cell_index) != maze.dimensionality):
        return False
    if maze.maze[cell_index] == 1:
        return False
    elif visited[cell_index] == 1:
        return False
    elif maze.maze[cell_index] == 3:
        return True

    visited[cell_index] = 1

    #recurse to orthogonal neighbors
    result = False
    for orth_index in maze.getOrthogonalNeighbors(cell_index):
        # print("orth_index: {}".format(orth_index))
        result = (result or maze.__isSolvableRecursive(orth_index, visited))
        if result == True:
            break
    return result
def findEntrance(maze):
    for index, val in np.ndenumerate(maze.maze):
        if val == 2:
            return index

def isSolvable(maze):
    entrance = maze.findEntrance()
    visited = np.zeros(maze.dimensions)
    return maze.__isSolvableRecursive(entrance, visited)

def __allAreConnectedRecursive(maze, cell_index, visitedPaths, allPaths, visited, cell_type):
    cell_pos = [1,2,3]
    cell_neg = [0]
    if cell_type == "path":
        cell_pos = [0,2,3]
        cell_neg = [1]
    if (cell_index == None  or len(cell_index) != maze.dimensionality):
        return False
    if visited[cell_index] == 1:
        return False
    elif maze.maze[cell_index] in cell_neg:
        return False
    elif maze.maze[cell_index] in cell_pos:
        visited[cell_index] = 1
        visitedPaths.append(cell_index)

    for orth_index in maze.getOrthogonalNeighbors(cell_index):
        maze.__allAreConnectedRecursive(orth_index, visitedPaths, allPaths, visited, cell_type)

    return (set(visitedPaths) == set(allPaths))

def allAreConnected(maze, cell_type="path"):
    visited = np.zeros(maze.dimensions)
    cell_pos = [1,2,3]
    if cell_type == "path":
        cell_pos = [0,2,3]
    allPos = maze.getAllCoordsOf(cell_pos)
    entrance = maze.findEntrance()
    visitedPaths = []
    return maze.__allAreConnectedRecursive(entrance, visitedPaths, allPos, visited, cell_type)

def printSolvability(maze):
    if maze.isSolvable():
        print("Solvable!")
    else:
        print("NOT solvable!")

def printConnectedness(maze):
    if maze.allAreConnected("path"):
        print("All paths are connected!")
    else:
        print("NOT all paths are connected!")
    if maze.allAreConnected("wall"):
        print("All walls are connected!")
    else:
        print("NOT all walls are connected!")

def getDensity(maze):
    cell = 0
    wall = 0
    for index, val in np.ndenumerate(maze.maze):
        if val == 1:
            wall += 1
            cell += 1
    return float(wall) / float(cell)

def printDensity(maze):
    print("Density: {}".format(maze.getDensity()))

def bruteForceMonteCarlo(maze, low_threshold=0, high_threshold=1, paths_connect = True, walls_connect = True):
    maze.monteCarloMaze()
    maze.fillBorders()
    maze.addEntranceAndExit()
    while maze.isSolvable() == False or (maze.getDensity() < low_threshold or maze.getDensity() > high_threshold) or maze.allAreConnected("path") == False or maze.allAreConnected("wall") == False:
        maze.monteCarloMaze()
        maze.fillBorders()
        maze.addEntranceAndExit()

    # def bruteForceAssignEntraceAndExit(maze, qualifications):
    #     bad = True
    #     while bad == True:
    #     maze.fillBorders()
    #     maze.addEntranceAndExit()
    #     for qualification in qualifications:
    #         bad = bad and (not qualification.__call__)

def bruteForceAssignEntraceAndExit(maze):
    maze.fillBorders()
    maze.addEntranceAndExit()
    while maze.isSolvable() == False:
        maze.fillBorders()
        maze.addEntranceAndExit()

def printMazeInfo(maze):
    maze.printSolvability()
    maze.printConnectedness()
    maze.printDensity()

def __DijkstrasShortestPath(maze, cell_index, visited, length):
    if (cell_index == None or len(cell_index) != maze.dimensionality):
        return False
    if maze.maze[cell_index] == 1:
        return False
    elif visited[cell_index] == 1:
        return False
    elif maze.maze[cell_index] == 3:
        return True

    length += 1
    visited[cell_index] = 1

    #recurse to orthogonal neighbors
    result = False
    for orth_index in maze.getOrthogonalNeighbors(cell_index):
        result = (result or maze.__isSolvableRecursive(orth_index, visited))
        if result == True:
            break
    return result

