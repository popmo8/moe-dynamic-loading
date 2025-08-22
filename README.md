# moe-dynamic-loading
## File Function Description
- `moe_batch_save_gpt.py`: Save the 'openai/gpt-oss-20b' model weights onto the disk
- `load.py`: Inference by loading non-experts layers weights onto GPU and then replacing the experts layers with customized layer "LoadExperts"
- `LoadExperts.py`: Customized layer that can dynamic load the required experts weight when inferencing

## Package Requirement
Please refer to `requirements.txt`
