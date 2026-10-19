"""Train a LeWM world model with the recipe of the project checkpoints.

The training seed draws the train/validation split of the clips, the order
of the batches and the initial weights. ``scripts/build_nodes.py`` redraws
the same split from the same fraction and seed. The checkpoint of every
epoch is written to ``<checkpoint_root>/checkpoints/<run_name>/
weights_epoch_<n>.pt`` with its ``config.json``; the training state of the
last finished epoch is kept beside it, and a rerun of the same command
resumes from it.

    python -m training.lewm data=cube \\
        data.dataset.name=cube_single_expert.h5 seed=3073 \\
        run_name=lewm-cube-seed3073 checkpoint_root=<directory>
"""

import os
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from lightning.pytorch.callbacks import Callback
from omegaconf import OmegaConf, open_dict
from stable_pretraining import data as dt
from stable_worldmodel.data import column_normalizer
from stable_worldmodel.wm.loss import SIGReg
from stable_worldmodel.wm.utils import save_pretrained


def image_preprocessor(img_size):
    """uint8 frames to ImageNet-normalized float images of ``img_size``."""
    to_image = dt.transforms.ToImage(
        **dt.dataset_stats.ImageNet, source='pixels', target='pixels'
    )
    resize = dt.transforms.Resize(img_size, source='pixels', target='pixels')
    return dt.transforms.Compose(to_image, resize)


class SaveWeights(Callback):
    """Save the world model and its config after every epoch."""

    def __init__(self, run_name, checkpoint_root, model_config):
        super().__init__()
        self.run_name = run_name
        self.checkpoint_root = checkpoint_root
        self.model_config = model_config

    def on_train_epoch_end(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return
        save_pretrained(
            pl_module.model,
            run_name=self.run_name,
            config=self.model_config,
            filename=f'weights_epoch_{trainer.current_epoch + 1}.pt',
            cache_dir=self.checkpoint_root,
        )


class SaveTrainingState(Callback):
    """Atomically save everything a resumed run needs after every epoch."""

    def __init__(self, path, data_generator):
        super().__init__()
        self.path = path
        self.data_generator = data_generator

    def state_dict(self):
        return {'data_generator_state': self.data_generator.get_state()}

    def load_state_dict(self, state_dict):
        self.data_generator.set_state(state_dict['data_generator_state'])

    def on_train_epoch_end(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return
        partial_path = self.path.with_suffix(self.path.suffix + '.partial')
        assert not partial_path.exists(), (
            f'a partial training state is left over: {partial_path}'
        )
        trainer.save_checkpoint(partial_path, weights_only=False)
        os.replace(partial_path, self.path)


def lewm_forward(self, batch, stage, cfg):
    """Encode the clip, predict the next latents and compute the losses."""
    history = cfg.wm.history_size
    # Actions past the end of an episode are NaN.
    batch['action'] = torch.nan_to_num(batch['action'], 0.0)
    output = self.model.encode(batch)
    emb = output['emb']  # (B, T, D)
    predicted = self.model.predict(
        emb[:, :history], output['act_emb'][:, :history]
    )
    target = emb[:, cfg.wm.num_preds :]
    output['pred_loss'] = (predicted - target).pow(2).mean()
    output['sigreg_loss'] = self.sigreg(emb.transpose(0, 1))
    output['loss'] = (
        output['pred_loss'] + cfg.loss.sigreg.weight * output['sigreg_loss']
    )
    self.log_dict(
        {f'{stage}/{k}': v.detach() for k, v in output.items() if 'loss' in k},
        on_step=True,
        sync_dist=True,
    )
    return output


@hydra.main(version_base='1.3', config_path='config', config_name='lewm')
def run(cfg):
    pl.seed_everything(cfg.seed, workers=True)

    dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    dataset = swm.data.load_dataset(
        dataset_cfg.pop('name'), transform=None, **dataset_cfg
    )
    transforms = [image_preprocessor(cfg.img_size)]
    with open_dict(cfg):
        for column in cfg.data.dataset.keys_to_load:
            if not column.startswith('pixels'):
                transforms.append(column_normalizer(dataset, column, column))
        cfg.model.action_encoder.input_dim = (
            cfg.data.dataset.frameskip * dataset.get_dim('action')
        )
    dataset.transform = spt.data.transforms.Compose(*transforms)

    generator = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset,
        lengths=[cfg.train_split, 1 - cfg.train_split],
        generator=generator,
    )
    train = torch.utils.data.DataLoader(
        train_set, **cfg.loader, generator=generator
    )
    val = torch.utils.data.DataLoader(
        val_set, **{**cfg.loader, 'shuffle': False, 'drop_last': False}
    )

    schedule_steps = cfg.trainer.max_epochs * len(train)
    optimizers = {
        'model_opt': {
            'modules': 'model',
            'optimizer': dict(cfg.optimizer),
            'scheduler': {
                'type': 'LinearWarmupCosineAnnealingLR',
                'warmup_steps': max(1, int(0.01 * schedule_steps)),
                'max_steps': schedule_steps,
            },
        },
    }
    module = spt.Module(
        model=hydra.utils.instantiate(cfg.model),
        sigreg=SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(lewm_forward, cfg=cfg),
        optim=optimizers,
    )

    run_dir = Path(cfg.checkpoint_root, 'checkpoints', cfg.run_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, run_dir / 'config.yaml')
    state_path = run_dir / 'training_state.ckpt'
    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[
            SaveWeights(cfg.run_name, cfg.checkpoint_root, cfg.model),
            SaveTrainingState(state_path, generator),
        ],
        num_sanity_val_steps=1,
        logger=False,
        enable_checkpointing=True,
    )
    manager = spt.Manager(
        trainer=trainer,
        module=module,
        data=spt.data.DataModule(train=train, val=val),
        seed=cfg.seed,
        ckpt_path=state_path if state_path.exists() else None,
        weights_only=False,
    )
    manager()


if __name__ == '__main__':
    run()
