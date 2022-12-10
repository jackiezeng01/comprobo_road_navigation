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

def generate_instructions(graph, path):
    instructions = []
    for i, node in enumerate(path[1:-1]): 
        previous_node = path[i]
        next_node = path[i + 2]

        if 'tag' in graph.edges[((node, next_node))]:
            tag = graph.edges[((node, next_node))]['tag']
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



graph = map_to_graph(map, 6, 7)
tag_map = {((1, 0), (2, 0)): 6, ((2, 0), (3, 0)): 6, ((2, 0), (2, 1)): 5, \
    ((0, 1), (0, 2)): 2, ((0, 2), (0, 3)): 2, ((1, 2), (2, 2)): 3, \
    ((2, 1), (2, 2)): 4, ((2, 2), (2, 3)): 4, ((0, 4), (0, 5)): 1, \
    ((2, 5), (3, 5)): 8}
nx.set_edge_attributes(graph, tag_map, 'tag')
path = node_to_node((4, 0), (0, 5), graph)
print(path)
print(generate_instructions(graph, path))
