"""Offline part of NGC: the graph distance between graph nodes.

The graph joins two nodes when each is among the ``neighbors`` nearest nodes
of the other, with an edge as long as their Euclidean distance. The graph
distance is the shortest-path length along these edges.
"""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial.distance import cdist


def build_graph(latents, neighbors):
    """Return float64 ``nodes`` [N, D] and ``graph_distance`` [N, N]."""
    nodes = np.asarray(latents, dtype=np.float64)
    assert nodes.ndim == 2 and len(nodes) > neighbors
    assert np.isfinite(nodes).all()
    distance = cdist(nodes, nodes)
    np.fill_diagonal(distance, np.inf)
    nearest = np.argsort(distance, axis=1, kind='stable')[:, :neighbors]
    assert (distance[np.arange(len(nodes)), nearest[:, 0]] > 0).all(), (
        'graph nodes must be distinct'
    )
    listed = np.zeros(distance.shape, dtype=bool)
    listed[np.arange(len(nodes))[:, None], nearest] = True
    row, col = np.nonzero(listed & listed.T)
    edges = csr_matrix((distance[row, col], (row, col)), shape=distance.shape)
    components = connected_components(edges, directed=False)[0]
    assert components == 1, f'graph has {components} components'
    graph_distance = dijkstra(edges, directed=False)
    return nodes, graph_distance
