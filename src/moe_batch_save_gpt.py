import torch
import os
import copy
from transformers import AutoModelForCausalLM

# Hyper Parameters for saving model module weights
MODEL_ID = "openai/gpt-oss-20b"
MODE = "normal"    # normal: 正常模式, split: 進一步將單一experts檔案拆分成多個檔案
NUM_EXPERTS = 32
MODEL_PATH = "/work/u8934334/gpt-oss-20b-split"

# 載入模型
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)

# 開始儲存模型
os.makedirs(MODEL_PATH, exist_ok=True)
for name, module in model.named_modules():
    if len(list(module.children())) == 0:  # 只儲存最底層（leaf）模組
        print(f"Processing module: {name} -> {type(module)}")
        module_copy = copy.deepcopy(module).cpu()
        safe_dir = os.path.join(MODEL_PATH, *name.split('.')[1:-1])     # 使用 / 連接模組名稱
        safe_name = name.split('.')[-1]  # 只保留最後一個名稱部分

        os.makedirs(safe_dir, exist_ok=True)
        torch.save(module_copy.state_dict(), f"{safe_dir}/{safe_name}.pt")
        print(f"Saved {name} to {safe_dir}/{safe_name}.pt")

        if MODE == "split" and safe_name == "experts":
            safe_dir = safe_dir + "/experts"
            os.makedirs(safe_dir, exist_ok=True)
            for expert_idx in range(NUM_EXPERTS):
                expert_weights = {}
                for k, v in module_copy.state_dict().items():
                    expert_weights[k] = v[expert_idx]
                torch.save(expert_weights, f"{safe_dir}/{expert_idx}.pt")
                print(f"Saved {name} expert {expert_idx} to {safe_dir}/{expert_idx}.pt")
