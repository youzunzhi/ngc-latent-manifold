"""NGC_Euc, the ablation of NGC that keeps its graph nodes.

NGC_Euc is ``NGCCost`` with ``euclidean_table`` in place of the graph
distance: a candidate still attaches through its anchors and still takes the
Euclidean distance within both radii, but the route between two anchors is
their Euclidean distance instead of the shortest path along the graph.
"""

import numpy as np
from scipy.spatial.distance import cdist


def euclidean_table(nodes):
    """Euclidean distances [N, N] between graph nodes, in float64.

    The arithmetic is that of ``build_graph``, so every edge of the graph
    has the same length here.
    """
    nodes = np.asarray(nodes, dtype=np.float64)
    assert nodes.ndim == 2 and np.isfinite(nodes).all()
    return cdist(nodes, nodes)
