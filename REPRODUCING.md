# Reproduce the main results

This guide reruns the paper's L2-versus-NGC evaluation on Cube, PushT,
Reacher, and TwoRoom using the published checkpoints. No retraining is
needed. Follow the [installation instructions](README.md#installation),
then run the commands below from the repository root.

## 1. Download the data and checkpoints

Download the official LeWM datasets from the linked revisions and extract
with `zstd`. Keep the original episode ordering.

| Task | Dataset revision | Download | Extracted HDF5 |
| --- | --- | --- | --- |
| Cube | [lewm-cube](https://huggingface.co/datasets/quentinll/lewm-cube/tree/02a19a67a0dc8c9d6215f89c19e0a597691e152a) | `cube_single_expert.tar.zst` | `cube_single_expert.h5` |
| PushT | [lewm-pusht](https://huggingface.co/datasets/quentinll/lewm-pusht/tree/655cd446b9929369d7d406001da85c15d1457850) | `pusht_expert_train.h5.zst` | `pusht_expert_train.h5` |
| Reacher | [lewm-reacher](https://huggingface.co/datasets/quentinll/lewm-reacher/tree/e70a080d0d04c6072123c9ebd343acf7fff28dbf) | `reacher.tar.zst` | `reacher.h5` |
| TwoRoom | [lewm-tworooms](https://huggingface.co/datasets/quentinll/lewm-tworooms/tree/6903a2de048b13819d812da0b4dd661290bc01e4) | `tworoom.tar.zst` | `tworoom.h5` |

For example, after placing the archives in `data/`:

```bash
tar --zstd -xf data/cube_single_expert.tar.zst -C data
zstd -d data/pusht_expert_train.h5.zst
tar --zstd -xf data/reacher.tar.zst -C data
tar --zstd -xf data/tworoom.tar.zst -C data
```

Download our four main checkpoints (about 276 MiB, no login required):

```bash
uv run python scripts/download_checkpoints.py
```

The downloader fetches weights and matching `config.json` files from
[Zunzhi/ngc-checkpoints](https://huggingface.co/Zunzhi/ngc-checkpoints),
using the pinned revision and SHA-256 checksums in
[the checkpoint inventory](artifacts/checkpoint-inventory.json).
Keep each weight file and its configuration together.

## 2. Set the data paths

```bash
cp experiments/example.yaml experiments/local.yaml
```

Edit the four `datasets` entries in `experiments/local.yaml` to point to
the extracted HDF5 files. The default checkpoint paths already match the
downloader. If you store checkpoints elsewhere, update the corresponding
`checkpoints.project` paths.

Use the committed files in `pairs/` to evaluate the paper's fixed pairs.
The runner handles graph sampling and excludes the evaluation episodes.

## 3. Build the graphs and run evaluation

Run these stages in order:

```bash
uv run python -m experiments.run --config experiments/local.yaml --stage main-nodes --execute
uv run python -m experiments.run --config experiments/local.yaml --stage main-cells --execute
uv run python scripts/summarize.py --root outputs/paper > outputs/main-table.json
```

This produces 12 graph-node files and 216 evaluations. Omit `--execute`
to preview commands. Existing outputs are not overwritten; for a new run,
pass the same fresh `--root <directory>` to both stages and the summarizer.

## 4. Compare the main table

Generate a reference table from the bundled paper outcomes. This step
needs no GPU and can also be run independently of steps 1–3:

```bash
uv run python -m zipfile -e artifacts/main-results.zip outputs/published
uv run python scripts/summarize.py --root outputs/published > outputs/reference-main-table.json
```

Compare `outputs/main-table.json` with `outputs/reference-main-table.json`.
Each contains 12 task/offset rows with L2 and NGC success rates, their
paired differences, and confidence intervals. Each row covers 300 pairs,
with three planner seeds averaged within each pair. A `null` interval
indicates a degenerate bootstrap distribution.

Use the same checkpoints, pairs, and protocol when comparing results.
Hardware and numerical differences can change individual trajectories;
bitwise-identical outputs are not required. Protocol details and success
criteria are in the [README](README.md#protocol-and-numerical-settings).

## Optional extensions

For training and supplementary studies, start from
[experiments/supplementary.example.yaml](experiments/supplementary.example.yaml).
Experiment settings and stages are documented in
[experiments/plan.py](experiments/plan.py). Training code is in
[training/lewm.py](training/lewm.py), and analysis code is in `analysis/`.
These are not required for the main table. To download the additional
training-seed checkpoints, use `scripts/download_checkpoints.py --include-replicates`.
