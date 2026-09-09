"""Select an official ProcTHOR house with ManiSkill's native scene builder.

Only environment construction uses scene metadata. No scene geometry or actor
coordinates are exposed to the learned policy by this adapter.
"""
from pathlib import Path

from mani_skill.utils.scene_builder.ai2thor.variants import ProcTHORSceneBuilder


def selected_procthor_builder(scene_name):
    class SelectedProcTHOR(ProcTHORSceneBuilder):
        def __init__(self, env, robot_init_qpos_noise=.02):
            super().__init__(env, robot_init_qpos_noise=robot_init_qpos_noise)
            matching = [c for c in self.build_configs
                        if Path(c.config_file).name == f"{scene_name}.scene_instance.json"]
            if len(matching) != 1:
                raise ValueError(f"Expected one native ProcTHOR config: {scene_name}")
            # A fresh list avoids modifying ManiSkill's global metadata cache.
            # Local index zero also uses its supported Fetch initialization;
            # the task spawn is set once before the first simulation step.
            self.build_configs = matching
            self._navigable_positions = [None]
    return SelectedProcTHOR
