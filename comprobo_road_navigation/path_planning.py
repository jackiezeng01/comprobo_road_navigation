import numpy as np
import networkx as nx
from itertools import permutations

map = [1, 1, 1, 1, 1, 0, 
        1, 0, 1, 0, 1, 0,
        1, 1, 1, 0, 1, 1,
        1, 0, 1, 0, 0, 1,
        1, 0, 1, 0, 1, 1,
        1, 1, 1, 1, 1, 0,
        1, 1, 1, 1, 1, 0]

def map_to_graph(map, width, height):
    graph = nx.Graph()
    for x in range(0, width):
        for y in range(0, height):
            if map[y * width + x]:
                graph.add_node((x, y), weight=1)
    for node in graph.nodes:
        x = node[0]
        y = node[1]

        right_node = (x + 1, y)
        bottom_node = (x, y + 1)
        # add an edge to the right, unless we're all the way over
        if (x != width - 1) & (right_node in graph.nodes):
            graph.add_edge(node, right_node)
        if (y != height - 1) & (bottom_node in graph.nodes):
            graph.add_edge(node, bottom_node)

    return graph
    

print(map_to_graph(map, 6, 7))



