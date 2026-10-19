# Reproducing the main results

Use the published checkpoints and fixed evaluation pairs to rerun the main
L2-versus-NGC evaluation on four tasks. Retraining and supplementary studies
are optional. Install the environment using the [README](README.md) first.

## Data and weights

Use the original LeWM HDF5 datasets with their episode ordering unchanged.
The Cube, PushT, and Reacher dataset repositories and revisions used in the
research configuration are:

| Task | Hugging Face dataset | Revision | File |
| --- | --- | --- | --- |
| Cube | `quentinll/lewm-cube` | `02a19a67a0dc8c9d6215f89c19e0a597691e152a` | `cube_single_expert.h5` |
| PushT | `quentinll/lewm-pusht` | `655cd446b9929369d7d406001da85c15d1457850` | `pusht_expert_train.h5` |
| Reacher | `quentinll/lewm-reacher` | `e70a080d0d04c6072123c9ebd343acf7fff28dbf` | `reacher.h5` |

For example, with the Hugging Face CLI installed:

```bash
hf download quentinll/lewm-cube cube_single_expert.tar.zst --repo-type dataset \
  --revision 02a19a67a0dc8c9d6215f89c19e0a597691e152a --local-dir data
```

Download TwoRoom from the official dataset repository
[`quentinll/lewm-tworooms`](https://huggingface.co/datasets/quentinll/lewm-tworooms).
The upstream revision inspected on 2026-10-18 is
`6903a2de048b13819d812da0b4dd661290bc01e4`, with `tworoom.tar.zst`.
Its extracted contents have not yet been compared with the paper's local
TwoRoom input. This revision identifies an available official download.

Official files are compressed: Cube uses `cube_single_expert.tar.zst`,
Reacher uses `reacher.tar.zst`, PushT uses `pusht_expert_train.h5.zst`, and
TwoRoom uses `tworoom.tar.zst`. Install `zstd` and decompress/extract them:

```bash
tar --zstd -xf data/cube_single_expert.tar.zst -C data
# For the downloaded PushT file:
zstd -d data/pusht_expert_train.h5.zst
```

Set the resulting HDF5 paths in `experiments/local.yaml` and preserve the
original episode ordering. This release hosts no datasets or conversions.

The main results use our checkpoints, training seed 3072, epoch 10.
The released LeWM checkpoints are a separate comparison. For every local
weight file, keep its matching `config.json` in the same directory.
`experiments/example.yaml` describes the expected layout:

- Cube, PushT, Reacher: `checkpoints/project/<task>/weights_epoch_10.pt`.
- TwoRoom: `checkpoints/training/checkpoints/tworoom-seed3072/weights_epoch_10.pt`.

Checkpoint sizes and SHA-256 digests are in
`artifacts/checkpoint-inventory.json` and `artifacts/CHECKPOINT_SHA256SUMS`.
The 12 checkpoints and matching configs are publicly available at
[Zunzhi/ngc-checkpoints](https://huggingface.co/Zunzhi/ngc-checkpoints), pinned
to revision `c62f91b9d6012766191776ba589591b4cb357854`. From the code root:

```bash
uv run python scripts/download_checkpoints.py
```

This command fetches the four main models (about 276 MiB). The downloader verifies file
sizes and SHA-256 and fails on mismatches. No login is required for these
public files. To choose a different destination, pass `--root <directory>`
and adjust the experiment configuration accordingly.

Use the published weights for the main evaluation. Retraining with the same
seed need not produce the same model.

## Main table

Copy the example configuration and set the four HDF5 paths:

```bash
cp experiments/example.yaml experiments/local.yaml
```

The default checkpoint paths match the downloader. The `training_datasets`
and `official` checkpoint entries are not used for the main evaluation.
Run from the repository root, building nodes before evaluating:

```bash
uv run python -m experiments.run --config experiments/local.yaml --stage main-nodes --execute
uv run python -m experiments.run --config experiments/local.yaml --stage main-cells --execute
uv run python scripts/summarize.py --root outputs/paper > outputs/main-table.json
```

These stages produce 12 graph-node files and 216 evaluations. Omit
`--execute` to inspect the commands first. To use a different output root,
pass `--root <directory>` to both stages and to the summarizer. The runner
stops on existing output files, so use a fresh root for a new run.

The summarizer reports all 12 task/offset rows, including success rates, mean per-pair differences, intervals, and
pair counts. It uses the same pair-level aggregation and 50,000 bootstrap
resamples as the research code. A degenerate interval is represented by
`null`, as in the paper's statistical procedure.

The 36 committed pair files contain 100 pairs each. Their SHA-256 digests
are in `artifacts.json`. To draw a new set for another experiment:

```bash
uv run python scripts/sample_pairs.py --dataset data/cube_single_expert.h5 \
  --task cube --rule protocol --goal-offset 50 --count 100 --seed 10 \
  --out outputs/cube-offset50-seed10.json
```

Do not replace the committed pairs when reproducing the paper. Build a
separate graph that excludes the new evaluation episodes.

## Compare with the published results

`artifacts/main-results.zip` contains the 216 main-table evaluation records.
These compact records preserve per-pair successes, settings, and timings.
`artifacts.json` records the archive checksum.

Extract to a new directory and aggregate:

```bash
python -m zipfile -e artifacts/main-results.zip outputs/published
uv run python scripts/summarize.py --root outputs/published > outputs/reference-main-table.json
```

This checks the published outcomes and statistics; it does not rerun the
planner or verify checkpoint contents. Use a separate output root for new
evaluations.

Compare `outputs/main-table.json` with `outputs/reference-main-table.json`
for each task and goal offset: L2 and NGC success rates, their paired
success-rate difference, confidence intervals, and pair counts. Each row
should contain 300 pairs, averaging three planner seeds within each pair.
Use the same checkpoints, fixed pairs, and settings for this comparison.
Individual successes and aggregate rates may vary across hardware and
software stacks; investigate material differences rather than requiring
identical trajectories, latent values, or output hashes.

A fresh Linux/A100 smoke run verified all four main checkpoint downloads,
rebuilt 5,000 Cube nodes, and evaluated two Cube pairs with L2 and NGC.
Aggregating the bundled outcomes matched all 12 reference rows. The full
main evaluation has not been rerun in that environment.

## Optional supplementary studies

The portable runner uses `experiments/plan.py`, which specifies the exact
study settings. The full evaluation sequence is `prepare`, `nodes`,
`subsets`, `cells`, then `analysis`. Each stage uses the same runner syntax
as the main table. This full sequence includes the main table, so use a
fresh root rather than running it over existing main-table outputs.

| Study | Implementation |
| --- | --- |
| NGC with Euclidean node distances | `ngc/ablations.py`, `--cost ngc-euc` |
| CEM budget | `CEM_MULTIPLIERS` in `experiments/plan.py` |
| Graph size, neighbor count, union rule, redraw | `scripts/subset_nodes.py`, `ngc/graph.py`, experiment plan |
| Simulator oracle | `evaluation/oracle.py`, `--dynamics oracle` |
| Released checkpoints and LeWM protocol | `--protocol comparability`, `configs/reacher-native.yaml` |
| Geometry, physical distance, timing | `analysis/geometry.py`, `analysis/strata.py`, `analysis/timing.py` |
| TwoRoom maps and embeddings | `analysis/tworoom_maps.py` |

Oracle evaluation uses one pair per invocation. The runner materializes
those single-pair inputs only when executing, never while printing commands.
The oracle preserves simulator state, strips executed-history prefixes,
and executes candidate trajectories to their end.

For the training-seed study, download the eight additional checkpoints
(total download including main models: about 827 MiB):

```bash
uv run python scripts/download_checkpoints.py --include-replicates
```

The released LeWM comparison uses separate upstream weights at
`checkpoints/official/<task>/weights.pt`.

`replicate-train`, `replicate-nodes`, and `replicate-cells` run the two
additional training seeds separately. The full training/evaluation matrix
is expensive and is not required for the main results.

The paper table formatter is `paper/scripts/tables.py`. Set
`NGC_RESULTS_ROOT` to a complete run root, optionally `NGC_ORACLE_ROOT` to a
separate oracle run root, and `NGC_STRATA_ROOT` to a separate strata folder.
Otherwise all studies are read from `outputs/paper`. It requires all study
outputs, unlike `scripts/summarize.py`, which needs only the main table.
The main-table outcome records are bundled as described above. Other
historical study records and figure assets are not bundled.

## Optional retraining

Retraining is not needed to reproduce the main evaluation. The training
implementation and configuration are in `training/`. For a
new 10-epoch model:

```bash
uv run python -m training.lewm data=cube \
  data.dataset.name=data/cube_single_expert.h5 seed=3072 \
  run_name=cube-seed3072 checkpoint_root=checkpoints/training
```

Use an absolute dataset path if overriding Hydra's working directory.
`train-smoke` runs eight training/validation batches on TwoRoom and PushT.
`train` trains the main TwoRoom checkpoint. The other main weights must be
provided separately. Models, configurations, and resumable training state
are written below the checkpoint root. No training is launched by install
or by printing an experiment stage.

PushT training in the recorded runs used a local Lance conversion of the
same official HDF5 dataset. Evaluation and graph sampling use HDF5 directly.
For optional retraining, prepare a conversion preserving columns and
row/episode order, then set `training_datasets.pusht`. A verified conversion
command remains to be documented.
