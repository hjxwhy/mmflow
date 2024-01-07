# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.model import is_model_wrapper
from mmengine.registry import HOOKS
from mmengine.runner import IterBasedTrainLoop, load_state_dict
from mmengine.hooks import Hook


@HOOKS.register_module()
class LiteFlowNetStageLoadHook(Hook):
    """Stage loading hook for LiteFlowNet.

    This hook works for loading weights at the previous stage to the additional
    stage in this training.

    Args:
        src_level (str): The source level to be loaded.
        dst_level (str): The level that will load the weights.
    """

    def __init__(self, src_level: str, dst_level: str) -> None:
        super().__init__()

        self.src_level = src_level
        self.dst_level = dst_level

    def before_run(self, runner: IterBasedTrainLoop) -> None:
        """Before running function of Hook.

        Args:
            runner (IterBasedTrainLoop): The runner for this training. This hook
                only has be tested in IterBasedTrainLoop.
        """
        runner.logger.info(
            f'Submodule of LiteFlowNet decoder at {self.dst_level} loads ' +
            f'LiteFlowNet\'s decoder at {self.src_level}')
        if is_model_wrapper(runner.model):
            load_state_dict(
                runner.model.module.decoder.decoders[self.dst_level],
                runner.model.module.decoder.decoders[
                    self.src_level].state_dict())
        else:
            load_state_dict(
                runner.model.decoder.decoders[self.dst_level],
                runner.model.decoder.decoders[self.src_level].state_dict())
