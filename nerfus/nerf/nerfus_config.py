"""
Define the Nerfus config.
"""

from __future__ import annotations

from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManagerConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.scheduler import ExponentialDecaySchedulerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.models.vanilla_nerf import VanillaModelConfig
from nerfstudio.plugins.types import MethodSpecification

from nerfus.nerf.nerfus_pipeline import NerfusPipelineConfig

nerfus_config = MethodSpecification(
    TrainerConfig(
        method_name="nerfus",
        project_name="nerfus-project",
        steps_per_eval_batch=1000,
        steps_per_eval_image=1000,
        steps_per_save=5000,
        steps_per_eval_all_images=0,
        save_only_latest_checkpoint=False,
        max_num_iterations=5001,
        mixed_precision=True,
        pipeline=NerfusPipelineConfig(
            datamanager=VanillaDataManagerConfig(
                dataparser=NerfstudioDataParserConfig(eval_mode="eval-frame-index"),
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
            ),
            model=VanillaModelConfig(
                eval_num_rays_per_chunk=1 << 15,
            ),
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=5e-4, eps=1e-8),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=5e-5),
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=5e-4, eps=1e-8),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=5e-5),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15, websocket_port=None),
        vis="viewer",
    ),
    description="Uses the Nerfus pipeline.",
)
