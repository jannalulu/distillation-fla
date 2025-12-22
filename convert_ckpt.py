import transformers
from transformers import AutoConfig, AutoModelForCausalLM
import torch
import os
import json
from safetensors.torch import load_file
from accelerate import init_empty_weights
from omegaconf import OmegaConf
import argparse
import json
import os
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import load_file
from distill_model.config_distilled_student import StudentConfig
from distill_model.modeling_distilled_student import StudentForCausalLM, get_student_attention_class

AutoConfig.register('student', StudentConfig, exist_ok=True)
AutoModelForCausalLM.register(StudentConfig, StudentForCausalLM, exist_ok=True)

import yaml

def parse_config(path: str):
    with open(path) as f:
        return yaml.safe_load(f)

def find_latest_checkpoint(base_dir: str) -> str:
    """Find the latest `checkpoint-*` directory under base_dir."""
    ckpts = [d for d in os.listdir(base_dir) if d.startswith("checkpoint-")]
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint-* found under {base_dir}")
    ckpts.sort(key=lambda x: int(x.split("-")[1]))
    return os.path.join(base_dir, ckpts[-1])

def convert_deepspeed_checkpoint_to_clean_student(
    base_dir: str,
    student_attn_class_name: str,
    keep_full_attention_layers: list[int] = [],
    teacher_model_name: str = None,
):
    ckpt_dir = find_latest_checkpoint(base_dir)
    print(f"🔍 Using latest checkpoint: {ckpt_dir}")

    config = AutoConfig.from_pretrained(ckpt_dir)
    config.use_cache = True

    student_attn_class = get_student_attention_class(student_attn_class_name)
    print(f"✅ Building student model with: {student_attn_class.__name__}")
    
    # Check for architecture changes between checkpoint and target config
    ckpt_keep_layers = set(config.to_dict().get('keep_full_attention_layers', []))
    target_keep_layers = set(keep_full_attention_layers)
    layers_to_init_from_teacher = target_keep_layers - ckpt_keep_layers

    if layers_to_init_from_teacher:
        print(f"⚠️ Layers transitioning student → full attention: {sorted(layers_to_init_from_teacher)}")

    config_dict = config.to_dict()
    config_dict['student_name'] = student_attn_class_name
    config_dict['name'] = 'student'
    config_dict['keep_full_attention_layers'] = keep_full_attention_layers
    config = StudentConfig(**config_dict)

    with init_empty_weights():
        student_model = AutoModelForCausalLM.from_config(config)
    student_model.to_empty(device='cpu')
    student_model = student_model.to(torch.bfloat16)
    model_state = student_model.state_dict()

    # load weights
    index_path = os.path.join(ckpt_dir, 'model.safetensors.index.json')
    safetensors_path = os.path.join(ckpt_dir, 'model.safetensors')
    pytorch_bin_path = os.path.join(ckpt_dir, 'pytorch_model.bin')

    state_dict = {}
    if os.path.exists(index_path):
        with open(index_path, 'r') as f:
            index = json.load(f)
        for shard_file in set(index['weight_map'].values()):
            state_dict.update(load_file(os.path.join(ckpt_dir, shard_file), device="cpu"))
    elif os.path.exists(safetensors_path):
        state_dict = load_file(safetensors_path, device="cpu")
    elif os.path.exists(pytorch_bin_path):
        state_dict = torch.load(pytorch_bin_path, map_location="cpu")
    else:
        raise FileNotFoundError("No weights found.")

    keys_to_remap = [k for k in state_dict if k.startswith("module.") or k.startswith("_forward_module.")]
    for k in keys_to_remap:
        state_dict[k.replace("module.", "").replace("_forward_module.", "")] = state_dict.pop(k)

    purified_state_dict = {}
    for k, v in state_dict.items():
        if ".student_attn." in k:
            new_key = k.replace(".student_attn", "")
            # Only include if shapes match
            if new_key in model_state and model_state[new_key].shape == v.shape:
                purified_state_dict[new_key] = v
        elif ".teacher_attn" not in k:
            # Only include if shapes match
            if k in model_state and model_state[k].shape == v.shape:
                purified_state_dict[k] = v

    # Load teacher weights for transitioning layers
    if layers_to_init_from_teacher and teacher_model_name:
        print(f"🔄 Loading teacher weights for transitioning layers from: {teacher_model_name}")
        teacher_model = AutoModelForCausalLM.from_pretrained(
            teacher_model_name, torch_dtype=torch.bfloat16, device_map="cpu"
        )
        teacher_state = teacher_model.state_dict()
        for k, v in teacher_state.items():
            if ".layers." in k and ".attn." in k:
                parts = k.split(".layers.")
                if len(parts) > 1:
                    layer_idx = int(parts[1].split(".")[0])
                    if layer_idx in layers_to_init_from_teacher:
                        if k in model_state and model_state[k].shape == v.shape:
                            purified_state_dict[k] = v
        del teacher_model
        print("✅ Teacher weights loaded for transitioning layers")
    elif layers_to_init_from_teacher:
        print("⚠️ No teacher model specified - transitioning layers will be randomly initialized")

    student_model.load_state_dict(purified_state_dict, strict=False)

    # save to {base_dir}/converted-hf/
    hf_output_dir = os.path.join(base_dir, "converted-hf")
    os.makedirs(hf_output_dir, exist_ok=True)
    student_model.save_pretrained(hf_output_dir, safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
    tokenizer.save_pretrained(hf_output_dir)

    print(f"✅ Saved clean student model to: {hf_output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, required=True)

    args = parser.parse_args()
    cfg_dict = parse_config(args.cfg)
    cfg = OmegaConf.create(cfg_dict)
    cfg = OmegaConf.to_container(cfg, resolve=True)
    convert_deepspeed_checkpoint_to_clean_student(
        base_dir=cfg['train']['output_dir'],
        student_attn_class_name=cfg['student_model']['name'],
        keep_full_attention_layers=cfg['student_model']['keep_full_attention_layers'],
        teacher_model_name=cfg.get('teacher_model', {}).get('name')
    )



