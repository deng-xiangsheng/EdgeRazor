#!/bin/bash
# =============================================================================
# distill_qwen2.5-omni.sh
# Distill Qwen2.5-Omni-7B training data from TGIF dataset.
#
# Usage:
#   bash distill_qwen2.5-omni.sh              # process all samples
#   bash distill_qwen2.5-omni.sh --sample 1   # process only 1 sample
#   bash distill_qwen2.5-omni.sh --sample 10  # process 10 samples
#   bash distill_qwen2.5-omni.sh --data-dir /path/to/datasets  # specify output directory
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export DATA_DIR="$SCRIPT_DIR/data"
export VIDEO_DIR="$SCRIPT_DIR/videos"
OUTPUT_DIR="/path/to/"
export OUTPUT_FILE="$OUTPUT_DIR/video_distilled_tgif_sub10k.jsonl"

mkdir -p "$DATA_DIR"
mkdir -p "$VIDEO_DIR"

# -----------------------------------------------------------------------------
# Parse arguments
# -----------------------------------------------------------------------------
SAMPLE_LIMIT=0
export SAMPLE_LIMIT

while [[ $# -gt 0 ]]; do
    case "$1" in
        --data-dir)
            if [[ -z "$2" ]]; then
                echo "Error: --data-dir requires a directory path."
                exit 1
            fi
            OUTPUT_DIR="$2"
            export OUTPUT_FILE="$OUTPUT_DIR/video_distilled_tgif_sub10k.jsonl"
            shift 2
            ;;
        --sample)
            SAMPLE_LIMIT="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: bash distill_qwen2.5-omni.sh [--sample N] [--data-dir <path>]"
            exit 1
            ;;
    esac
done

mkdir -p "$OUTPUT_DIR"

echo "============================================"
echo " Qwen2.5-Omni-7B TGIF Distillation"
echo " Data dir:   $DATA_DIR"
echo " Video dir:  $VIDEO_DIR"
echo " Output:     $OUTPUT_FILE"
echo " Sample limit: ${SAMPLE_LIMIT:-all}"
echo "============================================"

# -----------------------------------------------------------------------------
# Check / install Python dependencies
# -----------------------------------------------------------------------------
echo ""
echo "Checking Python dependencies..."
pip install -q datasets tqdm requests huggingface_hub torch transformers accelerate soundfile qwen-omni-utils 2>&1 | tail -1
echo "  Dependencies ready."

# Check ffmpeg
if ! command -v ffmpeg &> /dev/null; then
    echo "Error: ffmpeg is required but not found."
    exit 1
fi

# -----------------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------------

python - <<PYEOF
import json, os, subprocess, requests
import torch
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info
from datasets import load_dataset
from tqdm import tqdm

os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '300'
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'

data_dir = os.environ.get("DATA_DIR", ".")
video_dir = os.environ.get("VIDEO_DIR", ".")
output_file = os.environ.get("OUTPUT_FILE", ".")
sample_limit = int(os.environ.get("SAMPLE_LIMIT", "0"))

QUESTION = (
    "Describe what happens in this video from beginning to end. "
    "Include the opening scene, main events, and how it concludes. "
    "Write a single factual paragraph. Do not ask any questions."
)

def download_gif(url, save_path):
    try:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        with open(save_path, "wb") as f:
            f.write(resp.content)
        return True
    except Exception as e:
        print(f"  Download failed: {e}")
        return False

def convert_to_mp4(gif_path, mp4_path):
    cmd = [
        "ffmpeg", "-y",
        "-fflags", "+genpts+igndts",
        "-r", "30",
        "-i", str(gif_path),
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-r", "30",
        "-an",
        str(mp4_path),
    ]
    result = subprocess.run(cmd, capture_output=True)
    return result.returncode == 0

# ── Load Qwen2.5-Omni-7B model ──────────────────────────────────────────
print("Loading Qwen/Qwen2.5-Omni-7B model ...")
model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-Omni-7B",
    torch_dtype="auto",
    device_map="auto",
    # attn_implementation="flash_attention_2", # if flash attention is available
)
processor = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")
print("  Model loaded.")

# ── Load TGIF dataset ───────────────────────────────────────────────────
print("Loading navintiwari/TGIF dataset ...")
ds = load_dataset("navintiwari/TGIF", split="train")
print(f"  Total samples: {len(ds)}")

count = 0
with open(output_file, "w", encoding="utf-8") as out_f:
    iterator = enumerate(ds)
    if sample_limit > 0:
        iterator = list(iterator)[:sample_limit]

    for idx, sample in tqdm(iterator, desc="Distilling", total=sample_limit if sample_limit > 0 else len(ds)):
        url = sample.get("url", "")
        gif_name = os.path.basename(sample.get("path", ""))

        if not url or not gif_name:
            continue

        mp4_name = gif_name.rsplit(".", 1)[0] + ".mp4"
        gif_path = os.path.join(data_dir, gif_name)
        mp4_path = os.path.join(video_dir, mp4_name)

        # Download GIF if not present
        if not os.path.exists(gif_path):
            if not download_gif(url, gif_path):
                continue

        # Convert to MP4 if not present
        if not os.path.exists(mp4_path):
            if not convert_to_mp4(gif_path, mp4_path):
                print(f"  Convert failed: {gif_name}")
                continue

        abs_mp4_path = os.path.abspath(mp4_path)

        # ── Distill with Qwen2.5-Omni-7B ──────────────────────────────────
        USE_AUDIO_IN_VIDEO = False

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": abs_mp4_path},
                    {"type": "text", "text": QUESTION},
                ],
            },
        ]

        text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)
        inputs = processor(
            text=text, audio=audios, images=images, videos=videos,
            return_tensors="pt", padding=True, use_audio_in_video=USE_AUDIO_IN_VIDEO,
        )
        inputs = inputs.to(model.device).to(model.dtype)

        with torch.no_grad():
            text_ids, audio = model.generate(**inputs, use_audio_in_video=USE_AUDIO_IN_VIDEO, max_new_tokens=512)

        text_ids = text_ids[:, inputs.input_ids.shape[1]:]
        distilled_answer = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()

        if not distilled_answer:
            continue

        # ── Write output ──────────────────────────────────────────────────
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": abs_mp4_path},
                    {"type": "text", "text": QUESTION},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": distilled_answer},
                ],
            },
        ]

        out_f.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
        out_f.flush()
        count += 1

size_mb = os.path.getsize(output_file) / 1024 / 1024
print(f"  Saved {count} samples → {output_file} ({size_mb:.1f} MB)")
PYEOF

echo ""
echo "Done: video_distilled_tgif_sub10k.jsonl"
