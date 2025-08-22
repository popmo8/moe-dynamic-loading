# moe-dynamic-loading
## File Function Description
- `src/moe_batch_save_gpt.py`: Save the 'openai/gpt-oss-20b' model weights onto the disk
- `src/load.py`: Inference by loading non-experts layers weights onto GPU and then replacing the experts layers with customized layer "LoadExperts"
- `src/LoadExperts.py`: Customized layer that can dynamic load the required experts weight when inferencing

## Package Requirement
Please refer to `requirements.txt`

## Example Output
- `output/load.log`: Output for running `load.py`
