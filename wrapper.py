import torch
import torch.nn as nn

class AttentionDistillationWrapper(nn.Module):
    def __init__(self, teacher_attn, student_cls, config, layer_idx):
        super().__init__()
        self.teacher_attn = teacher_attn.eval()
        for p in self.teacher_attn.parameters():
            p.requires_grad_(False)

        self.student_attn = student_cls(config, layer_idx)
        self.student_attn.init_from_teacher(self.teacher_attn)
        self.distill_loss = torch.tensor(0.0)

    def forward(self, *args, **kwargs):
        kwargs["output_attentions"] = False
        kwargs["use_cache"] = False  # Disable cache for teacher pass during training

        # Teacher pass - no gradients
        with torch.no_grad():
            teacher_outputs = self.teacher_attn(*args, **kwargs)
            t_hidden = teacher_outputs[0]

        # Student pass (this is what must flow back to the decoder)
        student_outputs = self.student_attn(*args, **kwargs)
        s_hidden = student_outputs[0]

        # Stash distillation loss for the caller to consume
        self.distill_loss = torch.linalg.vector_norm(
            t_hidden - s_hidden, dim=-1
        ).mean() * (t_hidden.size(-1) ** -0.5)

        # Return same number of outputs as teacher (Llama returns 2, Qwen returns 3)
        # First element is teacher hidden state, rest are None placeholders
        return (t_hidden,) + (None,) * (len(teacher_outputs) - 1)
