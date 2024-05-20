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

############ CODE BLOCK 10 ################

class GraphBluePrint:
    def find_nodes(self): pass
    def find_edges(self): pass

class Graph(GraphBluePrint):
    def __init__(self, map_, start=(0, 0)):
        self.adjacency_list = {}
        self.map = map_
        self.start = start

        self.find_nodes()
        self.find_edges()  # This will be implemented in the next notebook cell

    def find_nodes(self):
        """
        This method contains a breadth-frist search algorithm to find all the nodes in the graph.
        """
        queue = deque([self.start])
        history = {self.start}

        while queue:
            current = queue.popleft()
            actions = self.neighbour_coordinates(current)
            self.adjacency_list_add_node(current, actions)

            for next_node in actions:
                if next_node not in history:
                    history.add(next_node)
                    queue.append(next_node)

    def adjacency_list_add_node(self, coordinate, actions):
        """
        This is a helper function for the breadth-first search algorithm to add a coordinate to the `adjacency_list`.
        """
        if len(actions) != 2 or (len(actions) == 2 and actions[0][0] != actions[1][0] and actions[0][1] != actions[1][1]):
            self.adjacency_list[coordinate] = set()

    def neighbour_coordinates(self, coordinate):
        """
        This method returns the next possible actions and is part of the breadth-first search algorithm.
        """
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        steps = []

        for dx, dy in directions:
            new_node = (coordinate[0] + dx, coordinate[1] + dy)
            if 0 <= new_node[0] < self.map.grid.shape[0] and 0 <= new_node[1] < self.map.grid.shape[1]:
                if self.map.grid[new_node[0], new_node[1]] != 0:
                    steps.append(new_node)

        return steps

    def __repr__(self):
        return repr(dict(sorted(self.adjacency_list.items()))).replace("},", "},\n")

    def __getitem__(self, key):
        return self.adjacency_list[key]

    def __contains__(self, key):
        return key in self.adjacency_list

    def get_random_node(self):
        return tuple(np.random.choice(list(self.adjacency_list)))

    def show_coordinates(self, size=5, color='k'):
        nodes = self.adjacency_list.keys()
        plt.plot([n[1] for n in nodes], [n[0] for n in nodes], 'o', color=color, markersize=size)

    def show_edges(self, width=0.05, color='r'):
        for node, edge_list in self.adjacency_list.items():
            for next_node, _, _ in edge_list:
                plt.arrow(node[1], node[0], (next_node[1] - node[1]) * 0.975, (next_node[0] - node[0]) * 0.975, color=color, length_includes_head=True, width=width, head_width=4 * width)

############ CODE BLOCK 15 ################
    def find_edges(self):
        """
        This method does a depth-first/brute-force search for each node to find the edges of each node.
        """
        raise NotImplementedError("Please complete this method")

    def find_next_node_in_adjacency_list(self, node, direction):
        """
        This is a helper method for find_edges to find a single edge given a node and a direction.

        :param node: The node from which we try to find its "neighboring node" NOT its neighboring coordinates.
        :type node: tuple[int]
        :param direction: The direction we want to search in this can only be 4 values (0, 1), (1, 0), (0, -1) or (-1, 0).
        :type direction: tuple[int]
        :return: This returns the first node in this direction and the distance.
        :rtype: tuple[int], int 
        """
        raise NotImplementedError("Please complete this method")


############ END OF CODE BLOCKS, START SCRIPT BELOW! ################
