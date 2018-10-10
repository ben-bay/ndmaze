import sys
sys.path.insert(0, '../')
sys.path.insert(0, '../src/')
from .context import ndmaze as n
print(sys.path)

import numpy as np


def test_fillBorders():
    maze = n.matrix(2, np.zeros(10,10))
    maze.fillBorders()
    assert_borders_full(maze)

def assert_borders_full(maze):
    for index, val in np.ndenumerate(maze.maze):
        for i, dim in enumerate(index):
            if dim == maze.dimensions[i] - 1 or dim == 0 :
                assert(maze.maze[index] == 1)

