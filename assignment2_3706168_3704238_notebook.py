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
        return list(self.adjacency_list.keys())[np.random.randint(len(self.adjacency_list))]

    def show_coordinates(self, size=5, color='k'):
        nodes = self.adjacency_list.keys()
        plt.plot([n[1] for n in nodes], [n[0] for n in nodes], 'o', color=color, markersize=size)

    def show_edges(self, width=0.05, color='r'):
        for node, edge_list in self.adjacency_list.items():
            for next_node, _, _ in edge_list:
                plt.arrow(node[1], node[0], (next_node[1] - node[1]) * 0.975, (next_node[0] - node[0]) * 0.975, color=color, length_includes_head=True, width=width, head_width=4 * width)

############ CODE BLOCK 15 ################
    def __init__(self, map_, start=(0, 0)):
        self.adjacency_list = {}
        self.map = map_
        self.start = start
        self.road_grid = map_.grid

        self.find_nodes()
        self.find_edges()  # This will be implemented in the next notebook cell
          
    def find_edges(self):
        """
        This method does a depth-first/brute-force search for each node to find the edges of each node.
        """
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        for node in self.adjacency_list:
            for direction in directions:
                next_node, distance = self.find_next_node_in_adjacency_list(node, direction)
                if next_node in self.adjacency_list and distance > 0:
                    speed_limit = self.road_grid[node[0], node[1]]
                    self.adjacency_list[node].add((next_node, distance, speed_limit))
                    self.adjacency_list[next_node].add((node, distance, speed_limit))

    def find_next_node_in_adjacency_list(self, node, direction):
        """
        Find the next node in a given direction and the distance to it.

        :param node: The node from which we try to find its "neighboring node" NOT its neighboring coordinates.
        :type node: tuple[int]
        :param direction: The direction we want to search in; this can only be 4 values (0, 1), (1, 0), (0, -1), or (-1, 0).
        :type direction: tuple[int]
        :return: This returns the first node in this direction and the distance.
        :rtype: tuple[int], int 
        """
        current = node
        distance = 0
        rows, cols = self.road_grid.shape
        while True:
            next_node = (current[0] + direction[0], current[1] + direction[1])
            if (next_node[0] < 0 or next_node[0] >= rows or
                next_node[1] < 0 or next_node[1] >= cols or
                next_node in self.adjacency_list):
                return next_node, distance
            current = next_node
            distance += 1

############ CODE BLOCK 120 ################

class FloodFillSolverGraph(FloodFillSolver):
    """
    A class instance should at least contain the following attributes after being called:
        :param queue: A queue that contains all the nodes that need to be visited.
        :type queue: collections.deque
        :param history: A dictionary containing the coordinates that will be visited and as values the coordinate that lead to this coordinate.
        :type history: dict[tuple[int], tuple[int]]
    """
    def __call__(self, graph, source, destination):      
        """
        This method gives a shortest route through the grid from source to destination.
        You start at the source and the algorithm ends if you reach the destination, both nodes should be included in the path.
        A route consists of a list of nodes (which are coordinates).

        Hint: The history is already given as a dictionary with as keys the node in the state-space graph and
        as values the previous node from which this node was visited.

        :param graph: The graph that represents the map.
        :type graph: Graph
        :param source: The node where the path starts.
        :type source: tuple[int]
        :param destination: The node where the path ends.
        :type destination: tuple[int]
        :return: The shortest route, which consists of a list of nodes and the length of the route.
        :rtype: list[tuple[int]], float
        """       
        self.queue = deque([source])
        self.history = {source: None}
        self.graph = graph  # Changed from Graph to graph (an instance)
        self.destination = destination
        
        while self.queue:
            current_node = self.queue.popleft()
            if current_node == destination:
                return self.find_path()
            for next_node in self.next_step(current_node, graph):  # Pass graph to next_step
                if next_node not in self.history:
                    self.queue.append(next_node)
                    self.history[next_node] = current_node
        
        return [], float('inf')
        
    def find_path(self):
        """
        This method finds the shortest paths between the source node and the destination node.
        It also returns the length of the path. 
        
        Note, that going from one node to the next has a length of 1.

        :return: A path that is the optimal route from source to destination and its length.
        :rtype: list[tuple[int]], float
        """
        path = []
        current_node = self.destination
        while current_node is not None:
            path.append(current_node)
            current_node = self.history[current_node]
        path.reverse()
        return path, len(path) - 1      

    def next_step(self, node, graph):
        """
        This method returns the next possible actions.

        :param node: The current node.
        :type node: tuple[int]
        :param graph: The graph that represents the map.
        :type graph: Graph
        :return: A list with possible next nodes that can be visited from the current node.
        :rtype: list[tuple[int]]  
        """
        return [neighbor for neighbor, _, _ in graph[node]]

############ CODE BLOCK 130 ################

class BFSSolverShortestPath():
    def __call__(self, graph, source, destination):
        self.priorityqueue = [(source, 0)]
        self.history = {source: (None, 0)}
        self.destination = destination
        self.graph = graph
        self.main_loop()
        return self.find_path()

    def find_path(self):
        path = []
        current_node = self.destination
        while current_node is not None:
            path.append(current_node)
            current_node, _ = self.history[current_node]
        path.reverse()
        total_cost = self.history[self.destination][1]
        return path, total_cost

    def main_loop(self):
        while self.priorityqueue:
            self.priorityqueue.sort(key=lambda x: x[1])  # Sort priority queue by distance
            current_node, current_cost = self.priorityqueue.pop(0)
            if self.base_case(current_node):
                return
            for next_node, distance, speed_limit in self.next_step(current_node):
                self.step(current_node, next_node, distance, speed_limit)

    def base_case(self, node):
        return node == self.destination

    def new_cost(self, previous_node, distance, speed_limit):
        return self.history[previous_node][1] + distance

    def step(self, node, new_node, distance, speed_limit):
        new_cost = self.new_cost(node, distance, speed_limit)
        if new_node not in self.history or new_cost < self.history[new_node][1]:
            self.history[new_node] = (node, new_cost)
            self.priorityqueue.append((new_node, new_cost))

    def next_step(self, node):
        return self.graph.adjacency_list.get(node, [])

############ CODE BLOCK 200 ################
class BFSSolverFastestPath(BFSSolverShortestPath):
    """
    A class instance should at least contain the following attributes after being called:
        :param priorityqueue: A priority queue that contains all the nodes that need to be visited 
                              including the time it takes to reach these nodes.
        :type priorityqueue: list[tuple[tuple[int], float]]
        :param history: A dictionary containing the nodes that will be visited and 
                        as values the node that lead to this node and
                        the time it takes to get to this node.
        :type history: dict[tuple[int], tuple[tuple[int], float]]
    """   
    def __call__(self, graph, source, destination, vehicle_speed):      
        """
        This method gives a fastest route through the grid from source to destination.

        This is the same as the `__call__` method from `BFSSolverShortestPath` except that 
        we need to store the vehicle speed. 
        
        Here, you can see how we can overwrite the `__call__` method but 
        still use the `__call__` method of BFSSolverShortestPath using `super`.
        """
        self.vehicle_speed = vehicle_speed
        print(f"Finding fastest path from {source} to {destination}, Vehicle speed: {self.vehicle_speed}")  # Debug statement
        return super(BFSSolverFastestPath, self).__call__(graph, source, destination)

    def new_cost(self, previous_node, distance, speed_limit):
        """
        This is a helper method that calculates the new cost to go from the previous node to
        a new node with a distance and speed_limit between the previous node and new node.

        Use the `speed_limit` and `vehicle_speed` to determine the time/cost it takes to go to
        the new node from the previous_node and add the time it took to reach the previous_node to it..

        :param previous_node: The previous node that is the fastest way to get to the new node.
        :type previous_node: tuple[int]
        :param distance: The distance between the node and new_node
        :type distance: int
        :param speed_limit: The speed limit on the road from node to new_node. 
        :type speed_limit: float
        :return: The cost to reach the node.
        :rtype: float
        """
        effective_speed = min(self.vehicle_speed, speed_limit)
        travel_time = distance / effective_speed
        return self.history[previous_node][1] + travel_time

############ CODE BLOCK 210 ################

def coordinate_to_node(map_, graph, coordinate):
    """
    This function finds a path from a coordinate to its closest nodes.
    A closest node is defined as the first node you encounter if you go a certain direction.
    This means that unless the coordinate is a node, you will need to find two closest nodes.
    If the coordinate is a node then return a list with only the coordinate itself.

    :param map_: The map of the graph
    :type map_: Map
    :param graph: A Graph of the map
    :type graph: Graph
    :param coordinate: The coordinate from which we want to find the closest node in the graph
    :type coordinate: tuple[int]
    :return: This returns a list of closest nodes which contains either 1 or 2 nodes.
    :rtype: list[tuple[int]]
    """
    if coordinate in graph:
        return [coordinate]

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    closest_nodes = set()

    for dx, dy in directions:
        x, y = coordinate
        while 0 <= x < map_.grid.shape[0] and 0 <= y < map_.grid.shape[1]:
            if (x, y) in graph:
                closest_nodes.add((x, y))
                break
            x += dx
            y += dy

    return list(closest_nodes)

############ CODE BLOCK 220 ################

def create_country_graphs(map_):
    """
    This function returns a list of all graphs of a country map, where the first graph is the highways and de rest are the cities.

    :param map_: The country map
    :type map_: Map
    :return: A list of graphs
    :rtype: list[Graph]
    """
    raise NotImplementedError("Please complete this method")

############ CODE BLOCK 300 ################

def path_length(coordinate, closest_nodes, map_, vehicle_speed):
    return [(node, (abs(node[0] - coordinate[0]) + abs(node[1] - coordinate[1])) / min(vehicle_speed, map_[coordinate])) for node in closest_nodes] 

def find_path(coordinate_A, coordinate_B, map_, vehicle_speed, find_at_most=3):
    """
    Find the optimal path according to the divide and conquer strategy from coordinate A to coordinate B.

    See hints and rules above on how to do this.

    :param coordinate_A: The start coordinate
    :type coordinate_A: tuple[int]
    :param coordinate_B: The end coordinate
    :type coordinate_B: tuple[int]
    :param map_: The map on which the path needs to be found
    :type map_: Map
    :param vehicle_speed: The maximum vehicle speed
    :type vehicle_speed: float
    :param find_at_most: The number of routes to find for each path finding algorithm, defaults to 3. 
                         Note, that this is only needed if you did 2.3.
    :type find_at_most: int, optional
    :return: The path between coordinate_A and coordinate_B. Also, return the cost.
    :rtype: list[tuple[int]], float
    """
    
    # Initialize the Graph for the map
    graph = Graph(map_)

    # Find the closest nodes to coordinate A and coordinate B
    closest_nodes_A = coordinate_to_node(map_, graph, coordinate_A)
    closest_nodes_B = coordinate_to_node(map_, graph, coordinate_B)

    print(f"Closest nodes to {coordinate_A}: {closest_nodes_A}")
    print(f"Closest nodes to {coordinate_B}: {closest_nodes_B}")

    # Find the fastest paths from coordinate A to its closest nodes
    solver = BFSSolverFastestPath()
    paths_from_A = []
    for node_A in closest_nodes_A:
        print(f"Finding path from {coordinate_A} to {node_A}")
        path_A, cost_A = solver(graph, coordinate_A, node_A, vehicle_speed)
        print(f"Path from {coordinate_A} to {node_A}: {path_A} with cost {cost_A}")
        paths_from_A.append((path_A, cost_A))

    # Find the fastest paths from coordinate B to its closest nodes
    paths_from_B = []
    for node_B in closest_nodes_B:
        print(f"Finding path from {coordinate_B} to {node_B}")
        path_B, cost_B = solver(graph, coordinate_B, node_B, vehicle_speed)
        print(f"Path from {coordinate_B} to {node_B}: {path_B} with cost {cost_B}")
        paths_from_B.append((path_B, cost_B))

    # Find the best entry and exit points on the highway
    highway_paths = []
    for path_A, cost_A in paths_from_A:
        for path_B, cost_B in paths_from_B:
            print(f"Finding highway path from {path_A[-1]} to {path_B[0]}")
            highway_path, highway_cost = solver(graph, path_A[-1], path_B[0], vehicle_speed)
            print(f"Highway path from {path_A[-1]} to {path_B[0]}: {highway_path} with cost {highway_cost}")
            total_cost = cost_A + highway_cost + cost_B
            highway_paths.append((path_A + highway_path[1:] + path_B[1:], total_cost))

    # Select the path with the minimum cost
    best_path, best_cost = min(highway_paths, key=lambda x: x[1])
    print(f"Best path: {best_path} with cost {best_cost}")

    return best_path, best_cost


############ END OF CODE BLOCKS, START SCRIPT BELOW! ################
