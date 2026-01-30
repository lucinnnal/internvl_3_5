## Setting

### 1. Conda Env
```bash
conda create -n internvl_captioning python=3.10
conda activate internvl_captioning
```

### 2. pip upgrade
```bash
pip install --upgrade pip
```

### 3. Torch (CUDA 11.8)
```bash
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu118
```

> 🔗 PyTorch previous version download guide: https://pytorch.org/get-started/previous-versions/

### 4. Verification
```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available()); print(torch.version.cuda)"
```

### 5. Other Required Packages
```bash
pip install -r requirements.txt
```

## Drive Mini Sample
[Download Drive Mini Sample](https://drive.google.com/drive/folders/1ZZfkhpWVY-U36Y5e62geOWX-euE2JpJx?usp=drive_link)

Move downloaded data to original folder data/

## Run Captioning
```bash
bash scripts/caption.sh
```

## Bash details
caption.sh..

- `CUDA_VISIBLE_DEVICES`를 설정해 **사용할 GPU 번호**를 지정.
  - 기본값: `0,1,2,3,4`
  - `--model-name`: 사용할 체크포인트(Hugging Face repo)
  - `--input-json-path`: 비디오 경로가 들어있는 JSON 파일
  - `--output-json-path`: 결과 캡션을 저장할 JSON 파일
  - `--use-sys-prompt`, `--sys-prompt`: 시스템 프롬프트 사용 여부 및 내용
  - `--question-suffix`: 메인 query

```bash
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
   --model-name "OpenGVLab/InternVL3_5-1B" \
   --input-json-path video_eval_paths/L5_video_paths.json \
   --output-json-path L5_internvl_output_captions.json \
   --prompts-json /home/kipyokim/internvl3.5/prompts.json \
   --use-sys-prompt False \
   --num_segments 16 \
```
