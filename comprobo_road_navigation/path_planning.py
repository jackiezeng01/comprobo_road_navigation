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
    # add a node for each free square
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
        # add an edge to the bottom node, unless we're all the way down
        if (y != height - 1) & (bottom_node in graph.nodes):
            graph.add_edge(node, bottom_node)
    return graph

def calculate_heur(start_node, end_node):
    # we're going for a straightforward Euclidean distance here
    x1 = start_node[0]
    y1 = start_node[1]

    x2 = end_node[0]
    y2 = end_node[1]

    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    
def node_to_node(start_node, end_node, graph):
    path = []

    if start_node == end_node:
        return None

    nx.set_node_attributes(graph, np.Inf, 'dist')   # set all the node attributes to infinite distance 
    nx.set_node_attributes(graph, {start_node: 0}, 'dist')  # set start node dist to 0 so we choose it first 
    nx.set_node_attributes(graph, None, 'heur')
    nx.set_node_attributes(graph, None, 'parent')

    open = {start_node}   # make a list of all the nodes to visit

    closed = {}
    while open:
        #open = sorted(open, key=lambda x: calculate_dist(x))    # sort the nodes by estimated distance to the end 
        curr_node = open.pop(0)
        if curr_node == end_node: 
            # we're done
            path.append(curr_node)
            break
        if curr_node in closed:
            continue
        estimated_dist = calculate_heur(curr_node, end_node)
        #nx.set_node_attributes(graph, {curr_node: estimated_dist}, 'dist')   # set all the node attributes to infinite distance 

        # we want to check each adjacent node 
        adjacent_nodes = graph.edges(curr_node)
        exact_dists = nx.get_node_attributes(graph, 'dist') # pre fetch the exact distance for each node so we have it handy
        new_dist = exact_dists[curr_node] + 1      # cost is 1 to each adjacent node so dist is current dist + 1 for all adjacent nodes
        for edge in adjacent_nodes:
            next_node = edge[1]
            if next_node not in open:   # check if we've seen this node before
                heur = calculate_heur(next_node, end_node)  # if not, calculate its approximate distance to the end
                nx.set_node_attributes(graph, {next_node: heur}, 'heur')
                nx.set_node_attributes(graph, {next_node: new_dist}, 'dist')
                nx.set_node_attributes(graph, {next_node: curr_node}, 'parent')
                # then add it to the nodes we still need to visit
                open.add(next_node)
            # if we've already seen it, check if it's a better path 
            if next_node in open: 
                if new_dist < exact_dists[next_node]:
                    nx.set_node_attributes(graph, {next_node: curr_node}, 'parent')
                    nx.set_node_attributes(graph, {next_node: new_dist}, 'dist')
    parents = nx.get_node_attributes(graph, 'parent')
    path_node = end_node
    while parents[path_node] != None:
        path.append(parents[path_node])
        path_node = parents[path_node]
    return path[::-1]

graph = map_to_graph(map, 6, 7)
print(node_to_node((0, 0), (1, 0), graph))
