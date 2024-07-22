from heapq import heappush, heappop
from itertools import count
import numpy as np


class GraphRouter():
    def __init__(self, graph, heuristic=lambda u, v: 0, is_symmetric=True):
        self.graph = graph
        self.heuristic = heuristic
        self.distances = {}
        self.is_symmetric = is_symmetric
        self.node_list = list(self.graph.nodes())
        self.node_to_idx = {v: i for i,v in enumerate(self.node_list)}

    # heuristic is a dictionary of keys (u,v) where u and v are two nodes, and the value is the approximated distance.
    # if the entry is not present, it needs to be calculated at runtime. can be saved between runs
    # Not used for now
    def get_dist(self, u, v):
        if self.is_symmetric:
            pair_key = tuple(sorted([u, v]))
        else:
            pair_key = (u, v)
        # pair_key = tuple(sorted([u, v]))
        if pair_key in self.distances:
            dist = self.distances[pair_key]
        else:
            dist = self.heuristic(u, v)
            self.distances[pair_key] = dist
        # print("Dist: {}".format(dist))
        return dist

    # Modified from https://networkx.org/documentation/stable/_modules/networkx/algorithms/shortest_paths/weighted.html#dijkstra_path
    def dijkstra(self, source, target, weight="length"):
        """Returns a list of nodes in a shortest path between source and target
        using Dijksta's algorithm.
        There may be more than one shortest path. This returns only one.
        Parameters
        ----------
        source : node
            Starting node for path

        target : node
            Ending node for path. Search is halted when target is found.
        weight : string or function
            If this is a string, then edge weights will be accessed via the
            edge attribute with this key (that is, the weight of the edge
            joining `u` to `v` will be ``G.edges[u, v][weight]``). If no
            such edge attribute exists, the weight of the edge is assumed to
            be one.
            If this is a function, the weight of an edge is the value
            returned by the function. The function must accept exactly three
            positional arguments: the two endpoints of an edge and the
            dictionary of edge attributes for that edge. The function must
            return a number.
        """
        if not callable(weight):
            weight = lambda u, v, data: data.get(weight, 1)
        pred = {source: None}
        # G_succ = G._succ if G.is_directed() else G._adj

        push = heappush
        pop = heappop
        dist = {}  # dictionary of final distances
        seen = {}
        # fringe is heapq with 3-tuples (distance,c,node)
        # use the count c to avoid comparing nodes (may not be able to)
        c = count()
        fringe = []

        # if source not in G:
        #     raise nx.NodeNotFound(f"Source {source} not in G")
        seen[source] = 0
        push(fringe, (0, next(c), source))

        while fringe:
            (d, _, v) = pop(fringe)
            if v in dist:
                continue  # already searched this node.
            dist[v] = d
            if v == target:
                path = []
                node = v
                while node is not None:
                    path.append(node)
                    node = pred[node]
                path.reverse()
                return path
            for u, e in self.graph[v].items():  # G_succ[v].items():
                cost = e[0]["length"]
                if cost is None:
                    continue
                vu_dist = dist[v] + cost
                if u in dist:
                    u_dist = dist[u]
                    if vu_dist < u_dist:
                        raise ValueError("Contradictory paths found:", "negative weights?")
                    elif pred is not None and vu_dist == u_dist:
                        pred[u].append(v)
                elif u not in seen or vu_dist < seen[u]:
                    seen[u] = vu_dist
                    push(fringe, (vu_dist, next(c), u))
                    pred[u] = v
                # elif vu_dist == seen[u]:
                #    pred[u].append(v)

        # The optional predecessor and path dictionaries can be accessed
        # by the caller via the pred and paths objects passed as arguments.
        return dist

        # Modified from https://networkx.org/documentation/stable/_modules/networkx/algorithms/shortest_paths/astar.html#astar_path

    def astar(self, source, target, weight="length", alpha=1):
        """Returns a list of nodes in a shortest path between source and target
        using the A* ("A-star") algorithm.
        There may be more than one shortest path.  This returns only one.
        Parameters
        ---------_
        source : node
            Starting node for path
        target : node
            Ending node for path. Search is halted when target is found.
        weight : string or function
            If this is a string, then edge weights will be accessed via the
            edge attribute with this key (that is, the weight of the edge
            joining `u` to `v` will be ``G.edges[u, v][weight]``). If no
            such edge attribute exists, the weight of the edge is assumed to
            be one.
            If this is a function, the weight of an edge is the value
            returned by the function. The function must accept exactly three
            positional arguments: the two endpoints of an edge and the
            dictionary of edge attributes for that edge. The function must
            return a number.
        """
        visited = []
        push = heappush
        pop = heappop
        if not callable(weight):
            weight = lambda u, v, data: data.get(weight, 1)

        # The queue stores priority, node, cost to reach, and parent.
        # Uses Python heapq to keep in priority order.
        # Add a counter to the queue to prevent the underlying heap from
        # attempting to compare the nodes themselves. The hash breaks ties in the
        # priority and is guaranteed unique for all nodes in the graph.
        c = count()
        queue = [(0, next(c), source, 0, None)]
        num_visited = 0

        # Maps enqueued nodes to distance of discovered paths and the
        # computed heuristics to target. We avoid computing the heuristics
        # more than once and inserting the node into the queue too many times.
        enqueued = {}
        # Maps explored nodes to parent closest to the source.
        explored = {}

        while queue:
            # Pop the smallest item from queue.
            length_est, __, curnode, dist, parent = pop(queue)
            # print(length_est-dist, dist, end=" || ")
            visited.append(curnode)
            num_visited += 1

            if curnode == target:
                path = [curnode]
                node = parent
                while node is not None:
                    path.append(node)
                    node = explored[node]
                path.reverse()
                return path, num_visited, visited

            if curnode in explored:
                # Do not override the parent of starting node
                if explored[curnode] is None:
                    continue

                # Skip bad paths that were enqueued before finding a better one
                qcost, h = enqueued[curnode]
                if qcost < dist:
                    continue

            explored[curnode] = parent

            for neighbor, w in self.graph[curnode].items():
                ncost = dist + w[0]["length"]# weight(curnode, neighbor, w)
                if neighbor in enqueued:
                    qcost, h = enqueued[neighbor]
                    # if qcost <= ncost, a less costly path from the
                    # neighbor to the source was already determined.
                    # Therefore, we won't attempt to push this neighbor
                    # to the queue
                    if qcost <= ncost:
                        continue
                else:
                    h = self.get_dist(neighbor, target)
                    # h = self.heuristic(neighbor, target)
                    # print("       " + str(h))
                if h != np.inf:
                    enqueued[neighbor] = ncost, h
                    push(queue, (ncost + alpha*h, next(c), neighbor, ncost, curnode))

        return None
        # raise nx.NetworkXNoPath(f"Node {target} not reachable from {source}")

    """def astar(self, source, dest):
        if not source or not dest:
            return {}, [] 
        sequence = []
        pred_list = {source : {'dist' : 0, 'pred' : None}}
        closed_set = set()
        unseen = [(0, source)]    # keeps a set and heap structure
        while unseen:
            _, vert = heappop(unseen)
            sequence.append((pred_list[vert]['pred'], [vert]))
            if vert in closed_set:
                continue
            elif vert == dest:
                return sequence[1:], pred_list
            closed_set.add(vert)
            for arc, arc_len in self.graph[vert]:
                if arc in closed_set: 
                    continue    # disgard nodes that already have optimal paths
                new_dist = pred_list[vert]['dist'] + arc_len
                if arc not in pred_list or new_dist < pred_list[arc]['dist']:
                    # the shortest path to the arc changed, record this
                    pred_list[arc] = {'pred' : vert, 'dist' : new_dist}
                    est = new_dist + self.get_dist(arc, dest)
                    heappush(unseen, (est, arc))
        return None    # no valid path found`"""

    def greedy(self, source, dest, alpha=1): # alpha is there just to make the code work, it'll be gotten rid of
        path = [source]
        seen = set()

        while path[-1] != dest:
            curr = path[-1]

            neighbors = [i[0] for i in self.graph[curr].items()]
            best_neighbor = neighbors[0]
            best_neighbor_dist = self.get_dist(best_neighbor, dest)
            for neighbor in neighbors:
                dist = self.get_dist(neighbor, dest)
                if dist < best_neighbor_dist:
                    best_neighbor_dist = dist
                    best_neighbor = neighbor

            path.append(best_neighbor)
            if best_neighbor in seen:
                return None

            seen.add(best_neighbor)

        return path

    def greedy_early_abort(self, source, dest, alpha=1):
        path = [source]

        while path[-1] != dest:
            curr = path[-1]

            neighbors = [i[0] for i in self.graph[curr].items()]
            neighbors.append(curr)
            best_neighbor = neighbors[0]
            best_neighbor_dist = self.get_dist(best_neighbor, dest)
            for neighbor in neighbors:
                dist = self.get_dist(neighbor, dest)
                if dist < best_neighbor_dist:
                    best_neighbor_dist = dist
                    best_neighbor = neighbor

            if best_neighbor == curr:
                return None

            path.append(best_neighbor)

        return path

    def node_is_greedy(self, node, dest):
        neighbors = [i[0] for i in self.graph[node].items()]
        neighbors.append(node)
        best_neighbor = neighbors[0]
        best_neighbor_dist = self.get_dist(best_neighbor, dest)
        for neighbor in neighbors:
            dist = self.get_dist(neighbor, dest)
            if dist < best_neighbor_dist:
                best_neighbor_dist = dist
                best_neighbor = neighbor

        if best_neighbor == node:
            return False

        return True

    def pair_is_greedy(self, first, second):
        return self.node_is_greedy(first, second) and self.node_is_greedy(second, first)


