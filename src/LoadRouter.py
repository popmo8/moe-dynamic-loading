import json
import math
import os
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

def print_gpu_usage(tag=""):
    device = torch.device("cuda:1")
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

class LoadRouter(nn.Module):
    def __init__(self, config, model_path: str, layer_idx: int, top_k: int, experts_to_select: int):
        super().__init__()
        self.top_k = top_k
        self.random = experts_to_select
        self.num_experts = config.num_experts
        self.hidden_dim = config.hidden_size
        self.model_path = model_path
        self.layer_idx = layer_idx

        router_params = torch.load(f"{self.model_path}/layers/{self.layer_idx}/mlp/router.pt")
        self.weight = nn.Parameter(router_params['weight'])
        self.bias = nn.Parameter(router_params['bias'])
        print_gpu_usage("After loading all expert weights")
        
    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        router_logits = F.linear(hidden_states, self.weight, self.bias)
        router_top_values, router_top_indices = torch.topk(router_logits, self.top_k, dim=-1)

        batch_size = router_top_indices.size(0)
        final_indices_list = []
        final_values_list = []

        for i in range(batch_size):
            random_positions = torch.randperm(router_top_indices.size(-1), device=router_top_indices.device)[:self.random]

            selected_indices = router_top_indices[i, random_positions]
            selected_values = router_top_values[i, random_positions]
            sorted_values, sorted_original_indices = torch.sort(selected_values, descending=True)
            sorted_indices = selected_indices[sorted_original_indices]

            final_indices_list.append(sorted_indices)
            final_values_list.append(sorted_values)

        final_top_indices = torch.stack(final_indices_list)
        final_top_values = torch.stack(final_values_list)

        #final_top_values, final_top_indices = random_selection(router_top_indices, router_top_values, self.experts_to_select)
        final_top_values = F.softmax(final_top_values, dim=-1, dtype=torch.float32).to(hidden_states.dtype)
        router_scores = torch.zeros_like(router_logits, dtype=hidden_states.dtype)
        router_scores.scatter_(1, final_top_indices, final_top_values)

        return router_scores, final_top_indices