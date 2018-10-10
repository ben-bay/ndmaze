#import pytest
import numpy as np

import ndmaze as n

def test_fillBorders():
    maze = n.matrix(2, np.zeros(10,10))
    maze.fillBorders()
    assert_borders_full(maze)

def assert_borders_full(maze):
    for index, val in np.ndenumerate(maze.maze):
        for i, dim in enumerate(index):
            if dim == maze.dimensions[i] - 1 or dim == 0 :
                assert(maze.maze[index] == 1)

