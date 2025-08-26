import json
import math
import os
from dataclasses import dataclass

import torch
import torch.distributed as dist

def print_gpu_usage(tag=""):
    device = torch.device("cuda:0")
    allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)  # GB
    reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)    # GB
    print(f"[{tag}] GPU memory allocated: {allocated:.3f} GB, reserved: {reserved:.3f} GB")

@dataclass
class ModelConfig:
    num_hidden_layers: int = 24
    num_experts: int = 32
    experts_per_token: int = 4
    vocab_size: int = 201088
    hidden_size: int = 2880
    intermediate_size: int = 2880
    swiglu_limit: float = 7.0
    head_dim: int = 64
    num_attention_heads: int = 64
    num_key_value_heads: int = 8
    sliding_window: int = 128
    initial_context_length: int = 4096
    rope_theta: float = 150000.0
    rope_scaling_factor: float = 32.0
    rope_ntk_alpha: float = 1.0
    rope_ntk_beta: float = 32.0

class LoadExperts(torch.nn.Module):

    def __init__(self, config, device: torch.device, model_path: str, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.alpha = 1.702
        self.limit = 7.0
        # Customized parameters
        self.device = device
        self.model_path = model_path
        self.layer_idx = layer_idx
        self.debug = False
        self.cnt = 1

    # Log debug message
    def log(self, message):
        if self.debug:
            print(f"[LoadExperts] {message}")

    def forward(self, hidden_states: torch.Tensor, router_indices=None, routing_weights=None) -> torch.Tensor:

        batch_size = hidden_states.shape[0]
        hidden_states = hidden_states.reshape(-1, self.hidden_size)  # (num_tokens, hidden_size)
        num_experts = routing_weights.shape[1]
        num_tokens = hidden_states.shape[0]
        self.log(f"Router Indices Shape: {router_indices.shape}")
        self.log(f"Routing Weights Shape: {routing_weights.shape}")
        self.log(f"hidden_states Shape: {hidden_states.shape}")

        if num_tokens == 1:
            # Only load top-k experts weight when num_tokens == 1
            router_id = router_indices.reshape(-1)
            gate_up_proj_tensors, gate_up_proj_bias_tensors, down_proj_tensors, down_proj_bias_tensors = [], [], [], []
            for expert_id in router_id:
                expert_weights = torch.load(f"{self.model_path}/layers/{self.layer_idx}/mlp/experts/{expert_id}.pt", map_location=self.device)
                gate_up_proj_tensors.append(expert_weights['gate_up_proj'])
                gate_up_proj_bias_tensors.append(expert_weights['gate_up_proj_bias'])
                down_proj_tensors.append(expert_weights['down_proj'])
                down_proj_bias_tensors.append(expert_weights['down_proj_bias'])
            gate_up_proj = torch.zeros(num_experts, gate_up_proj_tensors[0].shape[0], gate_up_proj_tensors[0].shape[1], device=self.device, dtype=gate_up_proj_tensors[0].dtype)
            gate_up_proj[router_id] = torch.stack(gate_up_proj_tensors)
            gate_up_proj_bias = torch.zeros(num_experts, gate_up_proj_bias_tensors[0].shape[0], device=self.device, dtype=gate_up_proj_bias_tensors[0].dtype)
            gate_up_proj_bias[router_id] = torch.stack(gate_up_proj_bias_tensors)
            down_proj = torch.zeros(num_experts, down_proj_tensors[0].shape[0], down_proj_tensors[0].shape[1], device=self.device, dtype=down_proj_tensors[0].dtype)
            down_proj[router_id] = torch.stack(down_proj_tensors)
            down_proj_bias = torch.zeros(num_experts, down_proj_bias_tensors[0].shape[0], device=self.device, dtype=down_proj_bias_tensors[0].dtype)
            down_proj_bias[router_id] = torch.stack(down_proj_bias_tensors)
            self.log(f"Loaded Expert Weights for Router ID: {router_id}")
            self.log(f"Shape: {gate_up_proj.shape}, {gate_up_proj_bias.shape}, {down_proj.shape}, {down_proj_bias.shape}")
            print_gpu_usage("After loading specific expert weights")
        else:
            # Load all experts weights when num_tokens > 1 because the union can be close to all experts
            experts_all = torch.load(f"{self.model_path}/layers/{self.layer_idx}/mlp/experts.pt", map_location=self.device)
            gate_up_proj, gate_up_proj_bias, down_proj, down_proj_bias = experts_all['gate_up_proj'], experts_all['gate_up_proj_bias'], experts_all['down_proj'], experts_all['down_proj_bias']
            self.log(f"Loaded Expert Weights for All Experts")
            self.log(f"Shape: {gate_up_proj.shape}, {gate_up_proj_bias.shape}, {down_proj.shape}, {down_proj_bias.shape}")
            print_gpu_usage("After loading all expert weights")

        
        hidden_states = hidden_states.to(gate_up_proj.dtype)
        hidden_states = hidden_states.repeat(num_experts, 1)
        hidden_states = hidden_states.view(num_experts, -1, self.hidden_size)
        gate_up = torch.bmm(hidden_states, gate_up_proj) + gate_up_proj_bias[..., None, :]
        gate, up = gate_up[..., ::2], gate_up[..., 1::2]
        gate = gate.clamp(min=None, max=self.limit)
        up = up.clamp(min=-self.limit, max=self.limit)
        glu = gate * torch.sigmoid(gate * self.alpha)
        next_states = torch.bmm(((up + 1) * glu), down_proj)
        next_states = next_states + down_proj_bias[..., None, :]
        next_states = next_states.view(num_experts, batch_size, -1, self.hidden_size)
        next_states = next_states * routing_weights.transpose(0, 1).view(num_experts, batch_size, -1)[..., None]
        next_states = next_states.sum(dim=0)
        print(f"================ Layer {self.layer_idx} {self.cnt}-th token inference finished ===================")
        self.cnt += 1

        return next_states