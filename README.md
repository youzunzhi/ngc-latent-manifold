# Neighborhood Graph Cost (NGC)

NGC is a terminal cost for planning with a latent world model. Instead of
the Euclidean distance between a predicted latent and the goal latent, it
measures the distance along the latent manifold: the shortest path over a
graph of training latents in which each latent is joined to its mutual
nearest neighbors. The graph is built once, offline; at planning time a
predicted latent and the goal attach to the graph through their nearest
graph nodes. NGC needs no training and replaces the cost of an existing
planner.

This repository contains NGC, the commands that build its graph from a
dataset and a [LeWM](https://arxiv.org/abs/2603.19312) checkpoint, and
the planning evaluation of the paper, on top of
[stable-worldmodel](https://github.com/galilai-group/stable-worldmodel).

## Installation

Linux with a CUDA GPU is required. With [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

This installs Python 3.10 and the pinned versions in `uv.lock`, with
PyTorch built for CUDA 13.0.

## Using NGC in a planner

`NGCCost` is a stable-worldmodel `Objective`: like the L2 cost `GoalMSE`, it
reads `predicted_emb` and `goal_emb` and returns one cost per candidate.

```python
import numpy as np
import stable_worldmodel as swm
from ngc import NGCCost, build_graph

latents = np.load('nodes.npz')['latents']  # [N, D] latents of training frames
nodes, graph_distance, bridges = build_graph(latents, neighbors=15)
objective = NGCCost(nodes, graph_distance, neighbors=15, anchors=4).cuda()
cost = swm.planning.ShootingCostEvaluator(model, objective)
```

`build_graph` joins two nodes when each is among the `neighbors` nearest
nodes of the other. If the graph has more than one component, it adds the
shortest edges between components until it is connected and returns them
as `bridges`. `NGCCost` scores a predicted latent `q` against the goal `g`
by their Euclidean distance when `q` lies within both of their
neighborhoods (the distance to their `neighbors`-th nearest node), and
otherwise by the shortest route `q -> v_i -> ... -> v_j -> g` through the
`anchors` nearest nodes of each.

## From a dataset to success rates

The three commands below sample evaluation pairs, build the graph nodes,
and plan. They read a stable-worldmodel HDF5 dataset and a LeWM checkpoint
(a local `.pt` file or a HuggingFace repository such as
`quentinll/lewm-cube`).

**1. Sample start-goal pairs.** Each pair is a start step in one episode
and the frame `--goal-offset` steps later; every pair comes from a
different episode.

```bash
uv run python scripts/sample_pairs.py --dataset cube_single_expert.h5 \
    --goal-offset 50 --count 100 --seed 0 --out pairs/cube.json
```

**2. Build the graph nodes.** One frame from each of `--count` training
episodes, encoded by the checkpoint. The training split is redrawn from the
split fraction and seed of the checkpoint's training run, over clips of its
frameskip and length, and the episodes of the given pair files are left
out, so no evaluation episode enters the graph. The values below are those
of the LeWM training recipe.

```bash
uv run python scripts/build_nodes.py --dataset cube_single_expert.h5 \
    --checkpoint lewm-cube --train-fraction 0.9 --train-seed 3072 \
    --frameskip 5 --num-steps 4 --exclude pairs/cube.json \
    --count 5000 --seed 0 --image-size 224 --batch-size 32 \
    --out nodes/cube.npz
```

**3. Plan.** `--cost ngc` or `--cost l2` (the cost LeWM plans with).

```bash
uv run python scripts/evaluate.py --config configs/cube.yaml \
    --pairs pairs/cube.json --dataset cube_single_expert.h5 \
    --checkpoint lewm-cube --cost ngc --nodes nodes/cube.npz \
    --neighbors 15 --anchors 4 --eval-budget 100 --planner-seed 42 \
    --out results/cube-ngc.json
```

The result lists every pair with its success and the success rate.

### Evaluation protocol

- Each environment starts from the recorded state of the pair's start row
  with its velocity set to zero (rest start), since the planner sees only
  the start frame.
- CEM (300 samples, 30 iterations, 30 elites) plans five action blocks of
  five steps and executes all of them before replanning. Each replan sees
  up to three real frames of the executed rollout and the blocks between
  them. Actions are clipped to the action space.
- A pair succeeds when the task's success rule holds at any step within
  `--eval-budget` steps. The paper uses twice the goal offset.

| Task | Config | Success rule |
| --- | --- | --- |
| Cube | `configs/cube.yaml` | cube within 0.04 m of the goal position (stable-worldmodel) |
| PushT | `configs/pusht.yaml` | joint agent and block position error below 20 px and block angle within 20 degrees of the goal (stable-worldmodel) |
| Reacher | `configs/reacher.yaml` | both joints within 0.05 rad of the goal pose, the shoulder angle compared on the circle, checked after each environment step (`evaluation/reacher.py`) |
| TwoRoom | `configs/tworoom.yaml` | agent within 16 px of the goal (stable-worldmodel) |

### Pair files

```json
{"goal_offset": 50, "pairs": [{"episode_id": 12, "start_step": 40}, ...]}
```

`episode_id` is the position of the episode in the dataset. Extra fields
of a pair (such as a `pair_id`) are copied into the result. The pair sets
of the paper were drawn with the CLEAR-LeWM pair generator, which also
drops pairs that already succeed at the start and, on TwoRoom, keeps only
pairs whose start and goal lie in different rooms; `sample_pairs.py` applies
neither filter.

## Numerical notes

- The package versions follow the environment the paper's results ran in.
  OpenCV and Pillow resize the rendered frames, so other versions change
  the pixels the model sees.
- Encoder latents depend on the GPU architecture at the level of 1e-6
  (the same frames encoded on an A100 and an RTX 4090 differ in about 90%
  of their entries). Plans depend on the latents, so success lists from
  different GPU architectures need not agree pair by pair.
- PushT resets draw a random goal pose before the recorded state is set,
  and stable-worldmodel does not seed that draw. Rerunning a PushT
  evaluation with every seed fixed can change the outcome of a few pairs.
