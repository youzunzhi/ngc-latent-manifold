"""Graph nodes of NGC: encoded observations from the training episodes.

Each node is one frame of a distinct training episode: a uniformly drawn
training clip of the episode, then a uniformly drawn frame of that clip.
Excluded episodes, such as those an evaluation starts from, never contribute
a node.

``clips`` is a stable-worldmodel dataset that enumerates the world model's
training clips, loaded with the frameskip and number of steps of training.
"""

import numpy as np
import stable_pretraining as spt
import torch
from torchvision.transforms import v2 as transforms


def training_clips(clips, train_fraction, seed):
    """Ids of the clips in the training split, as training drew them."""
    train, _ = spt.data.random_split(
        range(len(clips)),
        [train_fraction, 1 - train_fraction],
        generator=torch.Generator().manual_seed(seed),
    )
    return np.asarray(train.indices)


def sample_rows(clips, train_clip_ids, excluded_episodes, count, seed):
    """Ascending dataset rows [count], one frame from each of count episodes."""
    episode_start = np.asarray(clips.clip_indices)[np.sort(train_clip_ids)]
    episodes, starts = episode_start[:, 0], episode_start[:, 1]
    eligible = np.setdiff1d(episodes, sorted(excluded_episodes))
    assert 1 <= count <= len(eligible), (
        f'{count} nodes requested from {len(eligible)} eligible episodes'
    )
    rng = np.random.default_rng(seed)
    rows = []
    for episode in rng.choice(eligible, count, replace=False):
        low, high = np.searchsorted(episodes, [episode, episode + 1])
        start = rng.choice(starts[low:high])
        step = start + clips.frameskip * rng.integers(clips.num_steps)
        rows.append(clips.offsets[episode] + step)
    return np.sort(rows)


def image_transform(image_size):
    """Pixel preprocessing of LeWM: ImageNet normalization, then resize."""
    return transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=image_size),
        ]
    )


def encode(model, clips, rows, image_size, batch_size):
    """Float32 latents [len(rows), D] of the pixel observations at rows."""
    assert not model.training
    device = next(model.parameters()).device
    transform = image_transform(image_size)
    latents = []
    with torch.inference_mode():
        for i in range(0, len(rows), batch_size):
            images = clips.get_row_data(rows[i : i + batch_size].tolist())
            pixels = torch.from_numpy(images['pixels']).permute(0, 3, 1, 2)
            info = {'pixels': transform(pixels)[:, None].to(device)}
            latents.append(model.encode(info)['emb'][:, 0].float().cpu())
    return torch.cat(latents).numpy()
