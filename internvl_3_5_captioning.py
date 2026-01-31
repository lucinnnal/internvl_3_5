import math
import json
import random
import torch
import argparse
import numpy as np
import torchvision.transforms as T
from PIL import Image
from decord import VideoReader, cpu
from transformers import AutoTokenizer, AutoModel
from torchvision.transforms.functional import InterpolationMode

# Utils
# Each frame to tensor
def build_transform(input_size, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    MEAN, STD = mean, std
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

# Single frame(image) to 448*448 tiles + Thumbnail
def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

# Single image preprocess and dynamic porcessing
def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

# Get frame indices to extract (divide video into segments and extract frame in each segment)
def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array([
        int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
        for idx in range(num_segments)
    ])
    return frame_indices

# Video Load + Tensorfy the frames
def load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=32, fps=1):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    video_fps = float(vr.get_avg_fps())

    """
    duration = max_frame / video_fps
    num_segments = max(1, int(duration * fps))
    """

    pixel_values_list, num_patches_list = [], []
    transform = build_transform(input_size=input_size)
    frame_indices = get_index(bound, video_fps, max_frame, first_idx=0, num_segments=num_segments)
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
        img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num) # max_num=1, does not split into tiles for single image
        pixel_values = [transform(tile) for tile in img]
        pixel_values = torch.stack(pixel_values)
        num_patches_list.append(pixel_values.shape[0]) # how many tiles for each frame?
        pixel_values_list.append(pixel_values)
    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list

def main(args):
    
    # Seed
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    # Model Load
    model = AutoModel.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        use_flash_attn=True,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        device_map="auto").eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True, use_fast=False)
    generation_config = dict(max_new_tokens=1024, do_sample=False)

    # System prompt (if using)
    if args.use_sys_prompt:
        R1_SYSTEM_PROMPT = args.sys_prompt.strip()
        model.system_message = R1_SYSTEM_PROMPT
    
    # Get prompt candidates for captioning
    if args.prompts_json:
        with open(args.prompts_json, 'r') as f:
            prompts = json.load(f)["prompts"]
    else:
        prompts = [args.question_suffix]
    
    # Get video paths
    with open(args.input_json_path, 'r') as f:
        video_paths = json.load(f)["video_paths"]
    
    # Captioning
    results = []
    for video_path in video_paths:
        try:
            pixel_values, num_patches_list = load_video(video_path, max_num=1, num_segments=args.num_segments, fps=args.fps)
            pixel_values = pixel_values.to(torch.bfloat16).cuda()
            video_prefix = ''.join([f'Frame{i + 1}: <image>\n' for i in range(len(num_patches_list))])
            
            # Store the selected base prompt
            selected_prompt_text = random.choice(prompts)
            full_question = video_prefix + selected_prompt_text

            response, history = model.chat(tokenizer, pixel_values, full_question, generation_config,
                                           num_patches_list=num_patches_list, history=None, return_history=True)
            
            print(f"Processed: {video_path}")
            results.append({
                'video_path': video_path,
                'model_name': args.model_name,
                'prompt': selected_prompt_text,
                'response': response
            })
        
        # Error Logging
        except Exception as e:
            print(f"Error processing {video_path}: {e}")
            results.append({
                'video_path': video_path,
                'model_name': args.model_name,
                'prompt': selected_prompt_text,
                'response': f"Error: {e}"
            })

    with open(args.output_json_path, 'w') as f:
        json.dump({'results': results}, f, indent=4)
    print(f"Results saved to {args.output_json_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="OpenGVLab/InternVL3_5-4B", help="The name of the model to use.")
    parser.add_argument("--input-json-path", type=str, required=True, help="Path to the input JSON file containing video paths.")
    parser.add_argument("--output-json-path", type=str, required=True, help="Path to the output JSON file to save the results.")
    parser.add_argument("--prompts-json", type=str, help="Path to the JSON file containing prompts.")
    parser.add_argument("--use-sys-prompt", type=bool, default=False, help="Use system prompt?")
    parser.add_argument("--sys-prompt", type=str, help="System prompt", default = "You are an AI assistant that rigorously follows this response protocol: 1. First, conduct a detailed analysis of the question. Consider different angles, potential solutions, and reason through the problem step-by-step. Enclose this entire thinking process within <think> and </think> tags. 2. After the thinking section, provide a clear, concise, and direct answer to the user's question. Separate the answer from the think section with a newline. Ensure that the thinking process is thorough but remains focused on the query. The final answer should be standalone and not reference the thinking section.")
    parser.add_argument("--question-suffix", type=str, help="Suffix to append to the question.", default = 'Describe this video in detail.')
    parser.add_argument("--num_segments", type=int, help="segments number", default = 16)
    parser.add_argument("--fps", type=int, help="frame sampling rate", default = 1)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    args = parser.parse_args()
    main(args)

