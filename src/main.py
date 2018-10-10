#! /usr/bin/env python3 

"""
GOALS
> Generate mazes
> Traverse mazes
> Save mazes
    Serialization? Human-readable?
> Support nd mazes
> Support matrices and graphs to represent mazes
"""

import sys
import argparse

import ndmaze as m

def setup_argparse():
    """
    Example: main.py make prim 7 7
    """
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
        maze = m.matrix(args.dimensions)
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

if __name__ == "__main__":
    sys.exit(main()) 
