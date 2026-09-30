"""Graph-node sampling on a small dataset whose rows record their origin.

Row ``r`` of ``Frames`` is frame ``r - offsets[e]`` of episode ``e``, so a
sampled row tells which episode and step it came from.
"""

import numpy as np
import pytest
import torch
from stable_worldmodel.data.dataset import Dataset
from torch import nn

from ngc.nodes import encode, sample_rows, training_clips

FRAMESKIP = 5
NUM_STEPS = 4


class Frames(Dataset):
    """Episodes of random 8x8 RGB frames, one row per frame."""

    def __init__(self, lengths):
        lengths = np.asarray(lengths)
        offsets = np.concatenate([[0], np.cumsum(lengths)[:-1]])
        super().__init__(lengths, offsets, FRAMESKIP, NUM_STEPS)
        rng = np.random.default_rng(0)
        self.pixels = rng.integers(
            0, 256, size=(lengths.sum(), 8, 8, 3), dtype=np.uint8
        )

    def get_row_data(self, row_idx):
        return {'pixels': self.pixels[row_idx]}

    def origin(self, row):
        """Episode and step of one row."""
        episode = np.searchsorted(self.offsets, row, side='right') - 1
        return episode, row - self.offsets[episode]


class MeanColor(nn.Module):
    """Encoder stand-in whose latent is the mean of each color channel."""

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))

    def encode(self, info):
        return {'emb': self.scale * info['pixels'].mean(dim=(-1, -2))}


def test_training_clips_split_every_clip_once():
    # 11 + 26 + 2 + 41 = 80 clips. The fractions are [0.9, 1 - 0.9] as in
    # LeWM training, and 80 * (1 - 0.9) rounds down to 7, so the remaining
    # clip goes to the training split: 73 and 7.
    clips = Frames([30, 45, 21, 60])
    train = training_clips(clips, 0.9, seed=3)
    rest = np.setdiff1d(np.arange(len(clips)), train)
    assert len(train) == 73 and len(rest) == 7
    assert len(np.unique(train)) == len(train)
    assert np.array_equal(train, training_clips(clips, 0.9, seed=3))
    assert not np.array_equal(train, training_clips(clips, 0.9, seed=4))


def test_sample_rows_draws_one_training_frame_per_episode():
    # Episode 2 is too short for a clip and episode 5 is excluded.
    clips = Frames([30, 45, 19, 60, 25, 40, 33, 50])
    ids = np.arange(len(clips))
    # Leave episode 6 without training clips.
    train = ids[[episode != 6 for episode, _ in clips.clip_indices]]
    starts = {}
    for clip in train:
        episode, start = clips.clip_indices[clip]
        starts.setdefault(episode, set()).add(start)
    rows = sample_rows(clips, train, {5}, count=5, seed=0)
    assert np.all(np.diff(rows) > 0)
    episodes = set()
    for row in rows:
        episode, step = clips.origin(row)
        episodes.add(episode)
        assert any(
            step - start in FRAMESKIP * np.arange(NUM_STEPS)
            for start in starts[episode]
        )
    assert episodes == {0, 1, 3, 4, 7}
    assert np.array_equal(rows, sample_rows(clips, train, {5}, 5, seed=0))
    with pytest.raises(AssertionError, match='6 nodes requested from 5'):
        sample_rows(clips, train, {5}, count=6, seed=0)


def test_sample_rows_covers_every_frame_of_the_clip():
    # One episode with one clip: the node is one of the clip's frames, and
    # every one of them is drawn under some seed.
    clips = Frames([20])
    steps = {
        clips.origin(sample_rows(clips, [0], set(), 1, seed)[0])[1]
        for seed in range(50)
    }
    assert steps == set(FRAMESKIP * np.arange(NUM_STEPS))


def test_encode_preprocesses_each_row_independently_of_batching():
    clips = Frames([30, 45])
    rows = np.array([2, 7, 11, 40, 52, 70])
    model = MeanColor().eval()
    # ImageNet normalization of [0, 1] colors; the frames are already 8x8.
    mean, std = (
        np.array([0.485, 0.456, 0.406]),
        np.array([0.229, 0.224, 0.225]),
    )
    colors = clips.pixels[rows].astype(np.float64) / 255
    expected = ((colors - mean) / std).mean(axis=(1, 2))
    for batch_size in (1, 4, 6):
        latents = encode(model, clips, rows, 8, batch_size)
        assert latents.dtype == np.float32 and latents.shape == (6, 3)
        np.testing.assert_allclose(latents, expected, rtol=1e-5)
    with pytest.raises(AssertionError):
        encode(model.train(), clips, rows, 8, 2)
