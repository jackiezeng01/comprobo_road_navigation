import numpy as np
import networkx as nx
import heapq as hq
from itertools import permutations
import matplotlib.pyplot as plt

class PathPlanning():
    def __init__(self, start_node: tuple, end_node: tuple) -> None:
        """
        Initialize an instance of path planning. 

        Args:
            start_nodes: the coordinates of the node to start at
            end_node: the cordinates of the node to end at

        Attributes: 
            self.tag_map: a dictionary where the keys are tuples. Each tuple contains
                two tuples, representing the coordinates of two adjacent nodes.
                The values are the tag number that connects the two nodes. This can be
                an integer or a dictionary if a node has multiple tags. In those
                dictionaries, the keys are tuples representing the previous square and the 
                values are the tag number associated with that square. 
            
            self.map_grid: a binary representation of our map
            self.graph: our map converted into a graph, with edges between adjacent
                nodes
            self.path: a list where the nodes to visit will be stores 
            self.instructions: a list where the instructions will be stored
        """
        self.tag_map = {(2, 0): {(3, 0): 6, (1, 0): 6, (2, 1): 5}, 
            (2, 2): { (2, 1): 4, (2, 3): 4, (1, 2): 3},
            (0, 2): 2, (2, 5): 8, (4, 5): 7, (4, 6): 0, 
            (0, 6): 9, (0, 5): 1}
        self.map_grid = [1, 1, 1, 1, 1, 0, 
        1, 0, 1, 0, 1, 0,
        1, 1, 1, 0, 1, 1,
        1, 0, 1, 0, 0, 1,
        1, 0, 1, 0, 1, 1,
        1, 1, 1, 1, 1, 0,
        1, 1, 1, 1, 1, 0]
        self.graph = self.map_to_graph(6, 7)
        self.path = []
        nx.set_node_attributes(self.graph, self.tag_map, 'tag')
        self.node_to_node(start_node, end_node)

    def map_to_graph(self, width: int, height: int) -> nx.Graph:
        """
        Convert the binary map into a graph representation. 

        Args: 
            width: int representing the width of the map
            height: int representing the height of the map

        Returns: 
            a Networkx graph object representing the grid
        """
        graph = nx.Graph()
        # add a node for each free square
        for x in range(0, width):
            for y in range(0, height):
                if self.map_grid[y * width + x]:
                    graph.add_node((x, y), weight=1)
        for node in graph.nodes:
            x = node[0]
            y = node[1]

            right_node = (x + 1, y)
            bottom_node = (x, y + 1)
            # add an edge to the right, unless we're all the way over
            if (x != width - 1) & (right_node in graph.nodes):
                graph.add_edge(node, right_node)
            # add an edge to the bottom node, unless we're all the way down
            if (y != height - 1) & (bottom_node in graph.nodes):
                graph.add_edge(node, bottom_node)
        return graph

    def calculate_heur(self, start_node: tuple, end_node: tuple) -> float:
        """
        Helper function to calculate an estimated distance between nodes.
        
        Args: 
            start_node: tuple representing the coordinates of the node to start at
            end_node: tuple representing the coordinates of the node to end at

        Returns: 
            float representing the Euclidean distance between the nodes
        """
        # we're going for a straightforward Euclidean distance here
        x1 = start_node[0]
        y1 = start_node[1]

        x2 = end_node[0]
        y2 = end_node[1]

        return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    
    def node_to_node(self, start_node: tuple, end_node: tuple) -> None:
        """
        Find a path from the start node to the end node. 

        An implementation of the A* algorithm to find shortest distance. Uses
        Euclidean distance between nodes as a heuristic. Can be run multiple 
        times consectutively to calculate a path between multiple nodes. 
        Full path is stored in self.path

        Args: 
            start_node: tuple of coordinates representing the start node
            end_node: tuple of coordinates representing the end node
        """
        path = []
        path_node = None

        if start_node == end_node:
            return None

        nx.set_node_attributes(self.graph, np.Inf, 'dist')   # set all the node attributes to infinite distance 
        nx.set_node_attributes(self.graph, {start_node: 0}, 'dist')  # set start node dist to 0 so we choose it first 
        nx.set_node_attributes(self.graph, None, 'heur')
        nx.set_node_attributes(self.graph, None, 'est_dist')
        nx.set_node_attributes(self.graph, None, 'parent')

        open_nodes = [start_node]   # make a list of all the nodes to visit

        closed = set()
        while open_nodes:
            #open = sorted(open, key=lambda x: calculate_dist(x))    # sort the nodes by estimated distance to the end 
            if path_node is not None:
                break 
                # we're done
            
            curr_node = open_nodes.pop(0)
            if curr_node == end_node:
                path.append(curr_node)
                path_node = curr_node
                break
            
            adjacent_nodes = self.graph.edges(curr_node)
            curr_node_dist = self.graph.nodes[curr_node]['dist']
            next_node_dist = curr_node_dist + 1 
            for edge in adjacent_nodes:
                next_node = edge[1]
                if next_node not in closed:
                    self.graph.nodes[next_node]['parent'] = curr_node
                    if next_node == end_node:
                        path_node = next_node
                        if len(self.path) > 0:
                            if (self.path[-1] != next_node):
                                path.append(next_node)
                        break

                    if self.graph.nodes[next_node]['dist'] > next_node_dist:
                        est_dist = self.calculate_heur(next_node, end_node)
                        self.graph.nodes[next_node]['heur'] = est_dist
                        self.graph.nodes[next_node]['est_dist'] = next_node_dist + est_dist
                        self.graph.nodes[next_node]['dist'] = next_node_dist

                    if next_node not in open_nodes:
                        if next_node not in closed:
                            open_nodes.append(next_node)

            closed.add(curr_node)
            open_nodes.sort(key=lambda x: self.graph.nodes[x]['est_dist'])

        if path_node is None:
            print("No path found")
            return
        while self.graph.nodes[path_node]['parent'] is not None:
            next_node = self.graph.nodes[path_node]['parent']
            path.append(next_node)
            path_node = next_node
        self.path.extend(path[::-1])

    def generate_instructions(self):
        """
        Generate instructions for navigating intersections. 

        Uses the tag numbers at intersections and the direction of travel
        to calculate which way to turn at intersections. Appends a tuple 
        containing the tag number and either 'left', 'right', or 'straight' 
        to self.instructions
        """
        instructions = []
        for i, node in enumerate(self.path[1:-1]): 
            previous_node = self.path[i]
            next_node = self.path[i + 2]
            # we've reached an intersection:
            if 'tag' in self.graph.nodes[node]:
                tag = self.graph.nodes[node]['tag']
                # if multiple tags are possible, figure out which one we'll 
                # see based on the previous node
                if type(tag) == dict:
                    tag = tag[previous_node]

                # the direction of travel is whichever coord has changed 
                x_prev = node[0] - previous_node[0]
                y_prev = node[1] - previous_node[1]
                
                # if we were moving in an x direction, now we want to go y
                if abs(x_prev): 
                    dir = 'x'
                # if we were moving in a y direction, now we want to go x
                else: 
                    dir = 'y'

                # if we were traveling in a positive direction:
                if (x_prev > 0) | (y_prev > 0):
                    facing = 1
                # otherwise we're traveling in a negative direction
                else:
                    facing = -1

                # which direction are we travelling next? 
                x_next = next_node[0] - node[0]
                y_next = next_node[1] - node[1]

                # no change in direction, go straight
                if (x_next == x_prev) & (y_next == y_prev):
                    instructions.append((tag, 'straight'))
                # we used to be going x so now we're going y
                elif dir == 'x':
                    # if we're facing a positive direction
                    if facing > 0:
                        # turn right if we want to go positive
                        if y_next > 0:
                            instructions.append((tag, 'right'))
                        # turn left if we want to go negative
                        else: 
                            instructions.append((tag, 'left'))
                    # if we're facing a negative direction
                    else: 
                        # turn left if we want to go negative
                        if y_next < 0: 
                            instructions.append((tag, 'right'))
                        else: 
                            instructions.append((tag, 'left'))
                elif dir == 'y':
                    # if we're facing a positive direction
                    if facing > 0:
                        # turn right if we want to go positive
                        if x_next > 0:
                            instructions.append((tag, 'left'))
                        # turn left if we want to go negative
                        else: 
                            instructions.append((tag, 'right'))
                    # if we're facing a negative direction
                    else: 
                        # turn left if we want to go negative
                        if x_next < 0: 
                            instructions.append((tag, 'left'))
                        else: 
                            instructions.append((tag, 'right'))
        return instructions