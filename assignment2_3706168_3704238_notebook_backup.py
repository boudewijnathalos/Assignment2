############ CODE BLOCK 0 ################

# DO NOT CHANGE THIS CELL.
# THESE ARE THE ONLY IMPORTS YOU ARE ALLOWED TO USE:

import numpy as np
import copy
from grid_maker import Map
from collections import defaultdict, deque

RNG = np.random.default_rng()

############ CODE BLOCK 1 ################

class FloodFillSolver():
    def __call__(self, road_grid, source, destination):
        self.road_grid = road_grid
        self.source = source
        self.destination = destination
        
        # Initialize queue with the source node
        self.queue = deque([source])
        # Initialize history to keep track of paths
        self.history = {source: None}
        print(f"Starting point: {source}")
        print(f"Destination point: {destination}")
        self.main_loop()
        return self.find_path()

    def find_path(self):
        if self.destination in self.history:
            path = []
            node = self.destination
            while node is not None:
                path.append(node)
                node = self.history[node]
            path.reverse()
            distance = len(path) - 1
            print(f"Path found: {path}")
            print(f"Distance: {distance}")
            return path, float(distance)
        else:
            print(f"Destination {self.destination} not reached.")
            print("No path found.")
            return [], float('inf')

    def main_loop(self):
        while self.queue:
            current = self.queue.popleft()
            if self.base_case(current):
                return
            for next_node in self.next_step(current):
                self.step(current, next_node)

    def base_case(self, node):
        return node == self.destination

    def step(self, node, new_node):
        if new_node not in self.history and self.road_grid[new_node[0], new_node[1]] != 0:
            self.queue.append(new_node)
            self.history[new_node] = node

    def next_step(self, node):
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        steps = []
        for dx, dy in directions:
            new_node = (node[0] + dx, node[1] + dy)
            if 0 <= new_node[0] < self.road_grid.shape[0] and 0 <= new_node[1] < self.road_grid.shape[1]:
                steps.append(new_node)
        return steps


############ END OF CODE BLOCKS, START SCRIPT BELOW! ################
