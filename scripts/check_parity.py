"""Check NGC against the implementation that produced the paper's results.

The fixture holds, for each task, the graph nodes of one sealed graph, the
SHA-256 and sampled entries of its graph distance table, the first CEM pool
of every NGC evaluation cell with the cost the planner used (float64 and its
float32 cast), and the reference implementation's scores on the GPU for
synthetic queries that cover both cases of the cost. Every comparison is
exact. The costs were computed on CUDA, so the check runs there.

    python scripts/check_parity.py --fixture DIR
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from ngc import NGCCost, build_graph

NEIGHBORS = 15
ANCHORS = 4


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def score(model, predicted, goal):
    """Costs [C, S] of C pools, one goal per pool, in the input dtype."""
    return np.stack(
        [
            model(
                {
                    'predicted_emb': torch.from_numpy(p)[None, :, None].cuda(),
                    'goal_emb': torch.from_numpy(g)[None, None].cuda(),
                }
            )[0]
            .cpu()
            .numpy()
            for p, g in zip(predicted, goal)
        ]
    )


def check_task(fixture, task, record):
    path = fixture / f'{task}.npz'
    assert file_sha256(path) == record['fixture_sha256'], path
    data = np.load(path, allow_pickle=False)

    nodes, graph_distance = build_graph(data['latent'], NEIGHBORS)
    assert list(graph_distance.shape) == record['paths_shape']
    table_sha = hashlib.sha256(graph_distance.tobytes()).hexdigest()
    assert table_sha == record['paths_sha256'], 'graph distance table'
    sampled = graph_distance[data['path_sample_i'], data['path_sample_j']]
    assert np.array_equal(sampled, data['path_sample_value'])

    model = NGCCost(nodes, graph_distance, NEIGHBORS, ANCHORS).cuda()
    real32 = score(model, data['real_pred'], data['real_goal'])
    assert np.array_equal(real32, data['real_cost_sealed_f32']), 'float32'
    real64 = score(
        model,
        data['real_pred'].astype(np.float64),
        data['real_goal'].astype(np.float64),
    )
    assert np.array_equal(real64, data['real_cost_sealed']), 'float64'
    synthetic = score(model, data['synthetic_pred'], data['synthetic_goal'])
    assert np.array_equal(synthetic, data['synthetic_cost_reference_cuda']), (
        'synthetic'
    )
    return {
        'nodes': len(nodes),
        'pools': len(real32),
        'candidates': real32.size + synthetic.size,
        'synthetic_local': float(
            data['synthetic_local_reference_cuda'].mean()
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture', type=Path, required=True)
    args = parser.parse_args()
    assert torch.cuda.is_available(), 'the parity check runs on CUDA'
    record = json.loads((args.fixture / 'fixture.json').read_text())
    print('device:', torch.cuda.get_device_name(0), 'torch', torch.__version__)
    print('fixture device:', record['device'])
    for task, task_record in record['tasks'].items():
        result = check_task(args.fixture, task, task_record)
        print(f'{task}: identical', json.dumps(result))
    print('parity: all tasks identical')


if __name__ == '__main__':
    main()
