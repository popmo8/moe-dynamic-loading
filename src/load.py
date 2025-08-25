from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch
import os
from LoadExperts import ModelConfig, LoadExperts
from LoadRouter import ModelConfig, LoadRouter
import time

MODEL_PATH = "/work/morrisliu07/gpt-oss-20b-split/"
GPU_DEVICE = "cuda:1"
TOPK = 7
RANDOM = 4

def print_gpu_usage(tag=""):
    device = torch.device(GPU_DEVICE)
    allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)  # GB
    reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)    # GB
    print(f"[{tag}] GPU memory allocated: {allocated:.3f} GB, reserved: {reserved:.3f} GB")

MODEL_NAME = "openai/gpt-oss-20b"
config = AutoConfig.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_config(config)
print_gpu_usage("After initialize model skeleton")
print(f"Model Initial Structure:\n{model}")

for name, module in model.named_modules():
    if len(list(module.children())) == 0 and 'mlp.experts' not in name and 'mlp.router' not in name:  # only leaf modules and non-expert layers
        safe_dir = os.path.join(MODEL_PATH, *name.split('.')[1:-1])  # 使用 / 連接模組名稱
        safe_name = name.split('.')[-1]  # 只保留最後一個名稱部分
        state_dict = torch.load(f"{safe_dir}/{safe_name}.pt", map_location=GPU_DEVICE)
        module.load_state_dict(state_dict)
        print(f"Loaded {name} into the model.")
print_gpu_usage("After loading non-mlp layers")

# Replace the MLP layers with customized router layers
for layer_idx, layer in enumerate(model.model.layers):
    layer.mlp.router = LoadRouter(
        config=ModelConfig(),
        model_path=MODEL_PATH,
        layer_idx=layer_idx,
        top_k=TOPK,                 
        experts_to_select=RANDOM
    )
model.to(GPU_DEVICE)
print_gpu_usage("After replacing router layers and moving model to GPU")
print(f"Model Structure After Replacement:\n{model}")
        
# Replace the MLP layers with customized MLP layers
for layer_idx, layer in enumerate(model.model.layers):
    layer.mlp.experts = LoadExperts(config=ModelConfig(), device=GPU_DEVICE, model_path=MODEL_PATH, layer_idx=layer_idx)
model.to(GPU_DEVICE)
print_gpu_usage("After replacing experts layers and moving model to GPU")
print(f"Model Structure After Replacement:\n{model}")

# Model inference
model.eval()
messages = [
    {"role": "user", "content": "Repeat after me: \"My name is Morris.\""}
]

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True  # ensures assistant response marker is added
)

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
start_time = time.time()
outputs = model.generate(**inputs, max_new_tokens=150)
print(f"Generation Time: {(time.time() - start_time) / 60:.2f} minutes")
print("Model Output:", tokenizer.decode(outputs[0], skip_special_tokens=True))

