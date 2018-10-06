#! /usr/bin/env python3

import numpy as np
from enum import Enum
from random import randint
import argparse
import sys

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
        result = []
        for index, val in np.ndenumerate(self.maze):
            result.append(index)
        return result

    def fillBorders(self):
        for index, val in np.ndenumerate(self.maze):
            for i, dim in enumerate(index):
                if dim == self.dimensions[i] - 1 or dim == 0 :
                    self.maze[index] = 1

    def addEntrance(self):
        c = self.rndBorderCell()
        while (self.maze[c] == 2 or self.maze[c] == 3):
            c = self.rndBorderCell()
        self.maze[c] = 2

    def addExit(self):
        c = self.rndBorderCell()
        while (self.maze[c] == 2 or self.maze[c] == 3):
            c = self.rndBorderCell()
        self.maze[c] = 3

    def addEntranceAndExit(self):
        self.addEntrance()
        self.addExit()

    def primMaze(self):
        walls = []
        visited = []
        start = self.rndBorderBorderCell()
        print(start)
        self.maze[start] = 0
        visited.append(start)
        neighbor_walls = self.getOrthogonalNeighbors(start)
        walls += neighbor_walls
        # Pick a cell, mark it as part of the maze. Add the walls of the cell to the wall list.
        while len(walls) > 0:
            # print("")
            # print(str(len(walls))+" walls")
            # Pick a random wall from the list.
            wall_num = randint(0,len(walls)-1)
            rnd_wall = walls[wall_num]
            neighbor_walls = self.getOrthogonalNeighbors(rnd_wall)
            only_path = self.only1OrthVisited(neighbor_walls, visited)
            # If only one of the two cells that the wall divides is visited, then:
            if (only_path != -1 and rnd_wall not in self.getAllBorderCells()):
                # print("Only 1 visited path!")
                # print("wall: "+str(rnd_wall))
                # Make the wall a passage and mark the unvisited cell as part of the maze.
                passage = self.getPassage(only_path, rnd_wall)
                # if (passage == None):
                #     print("Passage is null!")
                #     del walls[wall_num]
                #     continue
                # print("Passage NOT null!")
                if (passage != None):
                    visited.append(passage)
                    self.maze[passage] = 0
                    # Add the neighboring walls of the cell to the wall list.
                    neighbor_walls = self.getOrthogonalNeighbors(passage)
                    unique_walls = set(neighbor_walls) - set(walls)
                    # print("unique_walls length: "+str(len(unique_walls)))
                    walls += list(unique_walls)
                visited.append(rnd_wall)
                self.maze[rnd_wall] = 0
            # Remove the wall from the list.
            del walls[wall_num]

    def getPassage(self, path, wall):
        result = None
        neighbors = self.getOrthogonalNeighbors(wall)
        diff_dim = -1
        direction = 0
        for i in range(self.dimensionality):
            if path[i] != wall[i]:
                diff_dim = i
                direction = wall[i] - path[i]
        result = list(path)
        result[diff_dim] = result[diff_dim]+(direction*2)
        # print("passage: " + str(result))

        result = tuple(result)

        if (result in self.getAllBorderCells()) or (result not in self.getAllCells()):
        # if (result not in self.getAllCells()):
            # print("passage rejected")
            return None
        return result

    def only1OrthVisited(self, neighbors, visited):
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
            rnd = randint(0,100)
            if rnd < 100*density:
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

def setup_argparse():
    parser = argparse.ArgumentParser(prog='ndmaze', description='Tool for generating n-dimensional mazes')
    subparsers = parser.add_subparsers(dest='subparser')
    make = subparsers.add_parser("make", help="make a maze")
    make.add_argument("-e", "--entrance_exit", action='store_true', help="include entrance and exit")
    make.add_argument("-i", "--info", action='store_true', help="print out info about maze")
    make.add_argument("maze_algo", type=str, nargs='?', help="type of maze, choose between rnd, prim, kruskal, density_island", default="prim")
    make.add_argument("dimensions", type=int, nargs='+')
    return parser.parse_args()

def parse_args(args):
    if hasattr(args, 'subparser') and args.subparser == 'make':
        maze = n_maze(args.dimensions)
        if args.maze_algo == "prim":
            maze.primMaze()
        if args.entrance_exit == True:
            maze.bruteForceAssignEntraceAndExit()
        maze.draw()
        if args.info == True:
             maze.printMazeInfo()

def main():
    args = setup_argparse()
    parse_args(args)
    #maze = n_maze([7, 7, 7])
    # maze.densityIsland2D()
    # maze.bruteForceMonteCarlo()
    # maze.bruteForceStep()
#    maze.primMaze()
#    maze.bruteForceAssignEntraceAndExit()
#    maze.printMazeInfo()

if __name__ == "__main__":
    sys.exit(main())
