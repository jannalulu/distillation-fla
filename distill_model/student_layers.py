# student_only_attention.py
# -*- coding: utf-8 -*-
# Copyright (c) 2025, Songlin Yang, Yanhong Li

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import logging
import sys
import os


def get_logger(name: str = None) -> logging.Logger:
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S"
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    if 'RANK' in os.environ and int(os.environ['RANK']) == 0:
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)

    return logger
logger = get_logger(__name__)


def get_num_heads(obj):
    """Get number of attention heads - handles both Qwen (num_heads) and Llama (num_attention_heads)."""
    return getattr(obj, 'num_heads', None) or getattr(obj, 'num_attention_heads', None)

def get_num_kv_heads(obj):
    """Get number of KV heads - handles both Qwen (num_kv_heads) and Llama (num_key_value_heads)."""
    return getattr(obj, 'num_kv_heads', None) or getattr(obj, 'num_key_value_heads', None)

def get_kv_repeat_factor(teacher_attn):
    """
    Calculate the KV repeat factor (num_heads // num_kv_heads) from weight shapes.
    This is more robust than relying on attribute names which differ between model families.
    """
    q_out_features = teacher_attn.q_proj.weight.shape[0]
    k_out_features = teacher_attn.k_proj.weight.shape[0]
    return q_out_features // k_out_features


from fla.layers.path_attn import PaTHAttention
class PaTHAttentionStudentV1(PaTHAttention):
    def __init__(self, config, layer_idx: int):
        super().__init__(hidden_size=config.hidden_size,
                         num_heads=get_num_heads(config),
                         num_kv_heads=get_num_kv_heads(config),
                         layer_idx=layer_idx,
                         use_w_shortconv=False,
                         use_low_rank_w=False,
                         )
    def init_from_teacher(self, teacher_attn):
        self.q_proj.weight.data.copy_(teacher_attn.q_proj.weight.data)
        self.k_proj.weight.data.copy_(teacher_attn.k_proj.weight.data)
        self.w_proj.weight.data.copy_(teacher_attn.w_proj.weight.data)
        self.v_proj.weight.data.copy_(teacher_attn.v_proj.weight.data)
        self.o_proj.weight.data.copy_(teacher_attn.o_proj.weight.data)

class PaTHFoXAttentionStudentV1(PaTHAttention):
    def __init__(self, config, layer_idx: int):
        super().__init__(hidden_size=config.hidden_size,
                         num_heads=get_num_heads(config),
                         num_kv_heads=get_num_kv_heads(config),
                         layer_idx=layer_idx,
                         use_w_shortconv=False,
                         use_low_rank_w=False,
                         use_forget_gate=True,
                         )
    def init_from_teacher(self, teacher_attn):
        self.q_proj.weight.data.copy_(teacher_attn.q_proj.weight.data)
        self.k_proj.weight.data.copy_(teacher_attn.k_proj.weight.data)
        self.w_proj.weight.data.copy_(teacher_attn.w_proj.weight.data)
        self.v_proj.weight.data.copy_(teacher_attn.v_proj.weight.data)
        self.o_proj.weight.data.copy_(teacher_attn.o_proj.weight.data)


from fla.layers.gated_deltanet import GatedDeltaNet
class GatedDeltaNetStudentV1(GatedDeltaNet):
    def __init__(self, config, layer_idx: int):
        super().__init__(hidden_size=config.hidden_size,
                         expand_v=1,
                         head_dim=config.hidden_size // get_num_heads(config),
                         use_gate=False,
                         use_short_conv=True,
                         num_heads=get_num_heads(config),
                         layer_idx=layer_idx,
                         )

    def init_from_teacher(self, teacher_attn):        
        k_weight = teacher_attn.k_proj.weight.data
        v_weight = teacher_attn.v_proj.weight.data

        k_weight_repeat = repeat(k_weight, 'h d -> (h g) d', g=get_kv_repeat_factor(teacher_attn))
        v_weight_repeat = repeat(v_weight, 'h d -> (h g) d', g=get_kv_repeat_factor(teacher_attn))
        
        self.q_proj.weight.data.copy_(teacher_attn.q_proj.weight.data)
        self.o_proj.weight.data.copy_(teacher_attn.o_proj.weight.data)
        self.k_proj.weight.data.copy_(k_weight_repeat)
        self.v_proj.weight.data.copy_(v_weight_repeat)
        logger.info(f"✅ GDN layer {self.layer_idx} init from teacher done.")


from fla.layers.gated_deltanet import GatedDeltaNet
class GatedDeltaNetStudentV2(GatedDeltaNet):
    def __init__(self, config, layer_idx: int):
        super().__init__(hidden_size=config.hidden_size,
                         expand_v=1,
                         head_dim=config.hidden_size // get_num_heads(config),
                         use_gate=False,
                         use_short_conv=False, 
                         num_heads=get_num_heads(config),
                         layer_idx=layer_idx,
                         )
        self.description = "Compared with GDN V1, this version does not use short conv"


    def init_from_teacher(self, teacher_attn):        
        k_weight = teacher_attn.k_proj.weight.data
        v_weight = teacher_attn.v_proj.weight.data

        k_weight_repeat = repeat(k_weight, 'h d -> (h g) d', g=get_kv_repeat_factor(teacher_attn))
        v_weight_repeat = repeat(v_weight, 'h d -> (h g) d', g=get_kv_repeat_factor(teacher_attn))
        
        self.q_proj.weight.data.copy_(teacher_attn.q_proj.weight.data)
        self.o_proj.weight.data.copy_(teacher_attn.o_proj.weight.data)
        self.k_proj.weight.data.copy_(k_weight_repeat)
        self.v_proj.weight.data.copy_(v_weight_repeat)
        logger.info(f"✅ GDN V2 layer {self.layer_idx} init from teacher done.")


class GatedDeltaNetStudentV3(GatedDeltaNet):
    def __init__(self, config, layer_idx: int):
        super().__init__(hidden_size=config.hidden_size,
                         expand_v=1,
                         head_dim=config.hidden_size // get_num_heads(config),
                         use_short_conv=True, 
                         num_heads=get_num_heads(config),
                         layer_idx=layer_idx,
                         use_gate=True,
                         )
        self.description = "Compared with GDN V1, this version uses output gate and short conv"
    

    def init_from_teacher(self, teacher_attn):        
        k_weight = teacher_attn.k_proj.weight.data
        v_weight = teacher_attn.v_proj.weight.data

        k_weight_repeat = repeat(k_weight, 'h d -> (h g) d', g=get_kv_repeat_factor(teacher_attn))
        v_weight_repeat = repeat(v_weight, 'h d -> (h g) d', g=get_kv_repeat_factor(teacher_attn))
        
        self.q_proj.weight.data.copy_(teacher_attn.q_proj.weight.data)
        self.o_proj.weight.data.copy_(teacher_attn.o_proj.weight.data)
        self.k_proj.weight.data.copy_(k_weight_repeat)
        self.v_proj.weight.data.copy_(v_weight_repeat)
        logger.info(f"✅ GDN V3 layer {self.layer_idx} init from teacher done.")

from fla.layers.gsa import GatedSlotAttention
class GatedSlotAttentionStudentV1(GatedSlotAttention):
    def __init__(self, config, layer_idx: int):
        super().__init__(hidden_size=config.hidden_size,
                         num_heads=get_num_heads(config),
                         num_kv_heads=get_num_heads(config),
                         num_slots=64,
                         layer_idx=layer_idx,                        
                         )

    def init_from_teacher(self, teacher_attn):        
        k_weight = teacher_attn.k_proj.weight.data
        v_weight = teacher_attn.v_proj.weight.data

        k_weight_repeat = repeat(k_weight, 'h d -> (h g) d', g=get_kv_repeat_factor(teacher_attn))
        v_weight_repeat = repeat(v_weight, 'h d -> (h g) d', g=get_kv_repeat_factor(teacher_attn))
        
        self.q_proj.weight.data.copy_(teacher_attn.q_proj.weight.data)
        self.o_proj.weight.data.copy_(teacher_attn.o_proj.weight.data)
        self.k_proj.weight.data.copy_(k_weight_repeat)
        self.v_proj.weight.data.copy_(v_weight_repeat)
        logger.info(f"✅ GSA V1 layer {self.layer_idx} init from teacher done.")



from fla.layers.gla import GatedLinearAttention
class GatedLinearAttentionStudentV1(GatedLinearAttention):
    def __init__(self, config, layer_idx: int):
        super().__init__(hidden_size=config.hidden_size,
                         expand_k=1,
                         expand_v=1,
                         num_heads=get_num_heads(config),
                         layer_idx=layer_idx,
                         )

    def init_from_teacher(self, teacher_attn):        
        k_weight = teacher_attn.k_proj.weight.data
        v_weight = teacher_attn.v_proj.weight.data

        k_weight_repeat = repeat(k_weight, 'h d -> (h g) d', g=get_kv_repeat_factor(teacher_attn))
        v_weight_repeat = repeat(v_weight, 'h d -> (h g) d', g=get_kv_repeat_factor(teacher_attn))
        
        self.q_proj.weight.data.copy_(teacher_attn.q_proj.weight.data)
        self.o_proj.weight.data.copy_(teacher_attn.o_proj.weight.data)
        self.k_proj.weight.data.copy_(k_weight_repeat)
        self.v_proj.weight.data.copy_(v_weight_repeat)
        logger.info(f"✅ GLA V1 layer {self.layer_idx} init from teacher done.")

