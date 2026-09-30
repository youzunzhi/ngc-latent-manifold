"""NGC on small graphs whose answers follow from the geometry.

``expected_cost`` is an independent NumPy statement of the NGC cost.
"""

import numpy as np
import pytest
import torch

from ngc import NGCCost, build_graph

STEP = 2 * np.pi / 16


def arc(count):
    """Unit-circle points one sixteenth of a turn apart, from angle 0."""
    angles = STEP * np.arange(count)
    return np.stack([np.cos(angles), np.sin(angles)], axis=1)


def expected_cost(nodes, graph_distance, q, g, neighbors, anchors):
    def attach(x):
        distance = np.linalg.norm(nodes - x, axis=1)
        order = np.argsort(distance, kind='stable')
        return (
            order[:anchors],
            distance[order[:anchors]],
            distance[order[neighbors - 1]],
        )

    direct = np.linalg.norm(q - g)
    q_ids, q_distance, q_radius = attach(q)
    g_ids, g_distance, g_radius = attach(g)
    if direct <= min(q_radius, g_radius):
        return direct
    return min(
        graph_distance[i, j] + q_distance[a] + g_distance[b]
        for a, i in enumerate(q_ids)
        for b, j in enumerate(g_ids)
    )


def score(model, q, g):
    """Cost [B, S] of candidates q [B, S, D] against goals g [B, D]."""
    info = {
        'predicted_emb': torch.as_tensor(q)[:, :, None],
        'goal_emb': torch.as_tensor(g)[:, None],
    }
    return model(info).numpy()


def test_graph_follows_the_arc():
    # Three quarters of a circle: the ends are 1.41 apart in the plane but
    # twelve steps apart along the arc, and only neighbors along the arc
    # are mutual 2-nearest neighbors.
    nodes, graph_distance, _ = build_graph(arc(13), neighbors=2)
    chord = 2 * np.sin(STEP / 2)
    steps = np.abs(np.arange(13)[:, None] - np.arange(13)[None, :])
    np.testing.assert_allclose(graph_distance, steps * chord, rtol=1e-12)
    assert np.linalg.norm(nodes[0] - nodes[12]) < 4 * chord


def test_graph_bridges_components_and_rejects_repeated_nodes():
    # Two quarter arcs 10 apart form two components. The one bridge joins
    # the closest pair: (1, 0) on the first arc, node 0, and (10, 1) on the
    # second, node 9. Its ends 4 and 5 are then four steps from each bridge
    # end.
    shifted = arc(5) + [10, 0]
    nodes, graph_distance, bridges = build_graph(
        np.concatenate([arc(5), shifted]), neighbors=2
    )
    assert bridges == [(0, 9)]
    chord = 2 * np.sin(STEP / 2)
    bridge = np.linalg.norm(nodes[0] - nodes[9])
    np.testing.assert_allclose(
        graph_distance[4, 5], 4 * chord + bridge + 4 * chord, rtol=1e-12
    )
    _, _, none = build_graph(arc(13), neighbors=2)
    assert none == []
    with pytest.raises(AssertionError, match='distinct'):
        build_graph(np.concatenate([arc(5), arc(1)]), neighbors=2)


def test_cost_takes_the_route_along_the_arc():
    nodes, graph_distance, _ = build_graph(arc(13), neighbors=2)
    model = NGCCost(nodes, graph_distance, neighbors=2, anchors=1)
    q = 1.05 * arc(1)
    g = 1.05 * nodes[12]
    route = np.linalg.norm(q - nodes[0]) + 12 * 2 * np.sin(STEP / 2)
    route += np.linalg.norm(nodes[12] - g)
    cost = score(model, q[None], g[None])
    np.testing.assert_allclose(cost, [[route]], rtol=1e-12)
    assert cost[0, 0] > 3 * np.linalg.norm(q - g)


def test_cost_is_euclidean_within_both_radii():
    nodes, graph_distance, _ = build_graph(arc(13), neighbors=2)
    model = NGCCost(nodes, graph_distance, neighbors=2, anchors=1)
    g = 1.02 * nodes[6]
    q = np.stack([g, 0.98 * nodes[6], 1.1 * nodes[7]])
    cost = score(model, q[None], g[None])
    np.testing.assert_array_equal(cost[0, 0], 0)
    np.testing.assert_allclose(
        cost[0, :2], np.linalg.norm(q[:2] - g, axis=1), rtol=1e-12
    )
    for c, candidate in zip(cost[0], q):
        np.testing.assert_allclose(
            c,
            expected_cost(nodes, graph_distance, candidate, g, 2, 1),
            rtol=1e-12,
        )


def test_tied_ranks_resolve_to_the_lower_node_index():
    # Queries on a unit lattice at half-integer offsets have exactly equal
    # distances across the anchor (4th/5th) and radius (15th/16th) ranks.
    grid = np.stack(np.meshgrid(np.arange(7), np.arange(7)), -1)
    nodes, graph_distance, _ = build_graph(grid.reshape(-1, 2), neighbors=15)
    model = NGCCost(nodes, graph_distance, neighbors=15, anchors=4)
    queries = np.array([[3.5, 3.0], [3.0, 3.5], [2.5, 3.5], [3.2, 2.9]])
    ids, distance, radius = (
        x.numpy() for x in model.attach(torch.as_tensor(queries))
    )
    for query, row_ids, row_distance, row_radius in zip(
        queries, ids, distance, radius
    ):
        exact = np.linalg.norm(nodes - query, axis=1)
        order = np.argsort(exact, kind='stable')
        np.testing.assert_array_equal(row_ids, order[:4])
        np.testing.assert_allclose(row_distance, exact[order[:4]], rtol=0)
        np.testing.assert_allclose(row_radius, exact[order[14]], rtol=0)
    tied = np.sort(np.linalg.norm(nodes - queries[0], axis=1))
    assert tied[3] == tied[4] and tied[14] == tied[15]


def test_batch_matches_single_goal_scoring():
    grid = np.stack(np.meshgrid(np.arange(7), np.arange(7)), -1)
    nodes, graph_distance, _ = build_graph(grid.reshape(-1, 2), neighbors=15)
    model = NGCCost(nodes, graph_distance, neighbors=15, anchors=4)
    rng = np.random.default_rng(0)
    q = rng.uniform(0, 6, size=(2, 5, 2))
    g = rng.uniform(0, 6, size=(2, 2))
    batch = score(model, q, g)
    for b in range(2):
        fresh = NGCCost(nodes, graph_distance, neighbors=15, anchors=4)
        np.testing.assert_array_equal(
            batch[b], score(fresh, q[b : b + 1], g[b : b + 1])[0]
        )
        for s in range(5):
            np.testing.assert_allclose(
                batch[b, s],
                expected_cost(nodes, graph_distance, q[b, s], g[b], 15, 4),
                rtol=1e-12,
            )


def test_goal_attachment_is_reused_until_the_goal_changes():
    grid = np.stack(np.meshgrid(np.arange(7), np.arange(7)), -1)
    nodes, graph_distance, _ = build_graph(grid.reshape(-1, 2), neighbors=15)
    model = NGCCost(nodes, graph_distance, neighbors=15, anchors=4)
    q = np.array([[[0.3, 0.4], [5.2, 5.9]]])
    first, second = np.array([[1.5, 1.2]]), np.array([[4.4, 4.1]])
    score(model, q, first)
    attachment = model.cached_goal_attachment
    score(model, q, first)
    assert model.cached_goal_attachment is attachment
    fresh = NGCCost(nodes, graph_distance, neighbors=15, anchors=4)
    np.testing.assert_array_equal(
        score(model, q, second), score(fresh, q, second)
    )
    assert model.cached_goal_attachment is not attachment


def test_cost_keeps_the_prediction_dtype():
    nodes, graph_distance, _ = build_graph(arc(13), neighbors=2)
    model = NGCCost(nodes, graph_distance, neighbors=2, anchors=1)
    q = arc(4).astype(np.float32)[None] * 1.01
    g = arc(1).astype(np.float32)
    cost = score(model, q, g)
    assert cost.dtype == np.float32 and cost.shape == (1, 4)
