import numpy as np
import networkx as nx
import heapq as hq
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
    path_node = None

    if start_node == end_node:
        return None

    nx.set_node_attributes(graph, np.Inf, 'dist')   # set all the node attributes to infinite distance 
    nx.set_node_attributes(graph, {start_node: 0}, 'dist')  # set start node dist to 0 so we choose it first 
    nx.set_node_attributes(graph, None, 'heur')
    nx.set_node_attributes(graph, None, 'est_dist')
    nx.set_node_attributes(graph, None, 'parent')

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
        
        adjacent_nodes = graph.edges(curr_node)
        curr_node_dist = graph.nodes[curr_node]['dist']
        next_node_dist = curr_node_dist + 1 
        for edge in adjacent_nodes:
            next_node = edge[1]
            if next_node not in closed:
                graph.nodes[next_node]['parent'] = curr_node
                if next_node == end_node: 
                    path.append(next_node)
                    path_node = next_node
                    break

                if graph.nodes[next_node]['dist'] > next_node_dist:
                    est_dist = calculate_heur(next_node, end_node)
                    graph.nodes[next_node]['heur'] = est_dist
                    graph.nodes[next_node]['est_dist'] = next_node_dist + est_dist
                    graph.nodes[next_node]['dist'] = next_node_dist

                if next_node not in open_nodes:
                    if next_node not in closed:
                        open_nodes.append(next_node)

        closed.add(curr_node)
        open_nodes.sort(key=lambda x: graph.nodes[x]['est_dist'])

    if path_node is None:
        print("No path found")
        return
    while graph.nodes[path_node]['parent'] is not None:
        next_node = graph.nodes[path_node]['parent']
        path.append(next_node)
        path_node = next_node
    return path[::-1]

graph = map_to_graph(map, 6, 7)
print(node_to_node((0, 0), (5, 6), graph))
