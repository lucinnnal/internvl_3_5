#!/bin/bash

# =====================================================================================
# Configurations
# =====================================================================================
# - Set the GPU devices to use. This value can be overridden by setting the environment
#   variable when running the script (e.g., `CUDA_VISIBLE_DEVICES=1 ./caption.sh`).
# - Multiple GPUs can be specified by separating them with commas (e.g., "0,1").
# =====================================================================================
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-5, 6} 

echo "Running captioning with CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

# Run the python script, passing all command-line arguments to it.
#
# Example usage:
# ./caption.sh \
#   --input-json-path "example_video_paths.json" \
#   --output-json-path "results.json" \
#   --model-name "OpenGVLab/InternVL3_5-8B"
#
python internvl_3_5_captioning.py \
   --model-name "OpenGVLab/InternVL3_5-38B" \
   --input-json-path video_eval_paths/L5_video_paths.json \
   --output-json-path L5_internvl_output_captions.json \
   --prompts-json /home/kipyokim/internvl3.5/prompts.json \
   --use-sys-prompt False \
   --num_segments 32 \

