# -*- coding: utf-8 -*-

from __future__ import annotations

import torch
import fla  # noqa
# import liger
# import lolcats
from lm_eval.__main__ import cli_evaluate
from lm_eval.api.registry import register_model
from lm_eval.models.huggingface import HFLM
from lm_eval.models.utils import get_dtype

from distill_model.config_distilled_student import StudentConfig
from distill_model.modeling_distilled_student import StudentModel, StudentForCausalLM
from transformers import AutoConfig, AutoModelForCausalLM

AutoConfig.register('student', StudentConfig, exist_ok=True)
AutoModelForCausalLM.register(StudentConfig, StudentForCausalLM, exist_ok=True)

@register_model('fla')
class FlashLinearAttentionLMWrapper(HFLM):
    def __init__(self, **kwargs) -> FlashLinearAttentionLMWrapper:
        # TODO: provide options for doing inference with different kernels
        super().__init__(**kwargs)

    def _create_model(
        self,
        pretrained: str,
        dtype: str | torch.dtype | None = "auto",
        **kwargs,
    ) -> None:
        """Override to fix dtype parameter name for HuggingFace transformers.

        HFLM passes 'dtype' but HF transformers expects 'torch_dtype'.
        This is a workaround for that incompatibility.
        """
        # IMPORTANT: HuggingFace from_pretrained expects 'torch_dtype', not 'dtype'
        # Always set torch_dtype in kwargs to ensure proper dtype handling
        if "torch_dtype" not in kwargs:
            if dtype is None or dtype == "auto":
                # Let from_pretrained read from config.json
                kwargs["torch_dtype"] = "auto"
            else:
                # Convert the dtype to proper format
                kwargs["torch_dtype"] = get_dtype(dtype)

        # Call parent but pass None for dtype since we're using torch_dtype
        return super()._create_model(pretrained=pretrained, dtype=None, **kwargs)


if __name__ == "__main__":
    cli_evaluate()