#!/bin/bash
# =============================================================================
# data_prepare.sh
# Generate training datasets for EdgeRazor-LLMs.
#
# Usage:
#   bash data_prepare.sh --all                          # download all datasets
#   bash data_prepare.sh --data ii_7M_instruct.jsonl    # download a single dataset
#   bash data_prepare.sh --data-dir /path/to/datasets    # specify output directory
#
# Valid dataset names:
#   ii_7M_instruct.jsonl        BAAI/Infinity-Instruct (7M)            ~7.45M
#   ii_gen_1.4M_instruct.jsonl  BAAI/Infinity-Instruct (Gen)           ~1.4M
#   tulu_0.6M_instruct.jsonl    allenai/tulu-v3.1-mix-preview-4096-OLMoE  ~0.61M
#   am_1.4M_instruct.jsonl      a-m-team/AM-DeepSeek-R1-Distilled-1.4M ~1.4M
#   task_0.2M_instruct.jsonl    Mixed Downstream Dataset               ~0.2M
#   ii_1.5M_base.jsonl          BAAI/Infinity-Instruct (7M_core)       ~1.48M
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_DIR="/path/to/"
export OUTPUT_DIR

# -----------------------------------------------------------------------------
# Valid dataset names
# -----------------------------------------------------------------------------
VALID_DATASETS=(
    ii_7M_instruct.jsonl
    ii_gen_1.4M_instruct.jsonl
    tulu_0.6M_instruct.jsonl
    am_1.4M_instruct.jsonl
    am_0.9M_sample_1k.jsonl
    task_0.2M_instruct.jsonl
    ii_1.5M_base.jsonl
)

# -----------------------------------------------------------------------------
# Parse arguments
# -----------------------------------------------------------------------------
DO_ALL=false
SELECTED_DATASETS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --all)
            DO_ALL=true
            shift
            ;;
        --data-dir)
            if [[ -z "$2" ]]; then
                echo "Error: --data-dir requires a directory path."
                exit 1
            fi
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --data)
            if [[ -z "$2" ]]; then
                echo "Error: --data requires a dataset name."
                echo "Valid options: ${VALID_DATASETS[*]}"
                exit 1
            fi
            SELECTED_DATASETS+=("$2")
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: bash data_prepare.sh [--all] [--data <dataset_name>] [--data-dir <path>]"
            echo "Valid dataset names: ${VALID_DATASETS[*]}"
            exit 1
            ;;
    esac
done

# Default to --all if no dataset specified
if [[ "$DO_ALL" == false && ${#SELECTED_DATASETS[@]} -eq 0 ]]; then
    DO_ALL=true
fi

# Validate selected datasets
for ds in "${SELECTED_DATASETS[@]}"; do
    found=false
    for valid in "${VALID_DATASETS[@]}"; do
        if [[ "$ds" == "$valid" ]]; then
            found=true
            break
        fi
    done
    if [[ "$found" == false ]]; then
        echo "Error: Invalid dataset '$ds'."
        echo "Valid dataset names: ${VALID_DATASETS[*]}"
        exit 1
    fi
done

# -----------------------------------------------------------------------------
# Build the list of datasets to generate
# -----------------------------------------------------------------------------
if [[ "$DO_ALL" == true ]]; then
    TO_GENERATE=("${VALID_DATASETS[@]}")
else
    TO_GENERATE=("${SELECTED_DATASETS[@]}")
fi

mkdir -p "$OUTPUT_DIR"

echo "============================================"
echo " EdgeRazor-LLMs Dataset Preparation"
echo " Output directory: $OUTPUT_DIR"
echo " Datasets to generate: ${TO_GENERATE[*]}"
echo "============================================"

# -----------------------------------------------------------------------------
# Helper: check if a dataset is in the to-generate list
# -----------------------------------------------------------------------------
should_generate() {
    local target="$1"
    for ds in "${TO_GENERATE[@]}"; do
        if [[ "$ds" == "$target" ]]; then
            return 0
        fi
    done
    return 1
}

# -----------------------------------------------------------------------------
# Check / install Python dependencies
# -----------------------------------------------------------------------------
echo ""
echo "Checking Python dependencies..."
pip install -q datasets jinja2 tqdm zstandard huggingface_hub 2>&1 | tail -1
echo "  Dependencies ready."

# =============================================================================
# Dataset generators (each is a bash function)
# =============================================================================

generate_ii_7M_instruct() {
    echo ""
    echo "--- Generating ii_7M_instruct.jsonl (BAAI/Infinity-Instruct, 7M, ~7.45M) ---"

    python - <<'PYEOF'
import json, os
from datasets import load_dataset
from tqdm import tqdm

output_dir = os.environ.get("OUTPUT_DIR", ".")
role_map = {"human": "user", "gpt": "assistant"}

def convert(conv):
    return [{"role": role_map.get(t["from"], t["from"]), "content": t["value"]} for t in conv]

output_path = os.path.join(output_dir, "ii_7M_instruct.jsonl")
print("  Loading BAAI/Infinity-Instruct (7M) ...")
ds = load_dataset("BAAI/Infinity-Instruct", "7M", split="train")
count = 0
with open(output_path, "w", encoding="utf-8") as f:
    for ex in tqdm(ds, desc="  Converting 7M"):
        convs = ex.get("conversations")
        if not convs or not isinstance(convs, list):
            continue
        msgs = convert(convs)
        f.write(json.dumps({"messages": msgs}, ensure_ascii=False) + "\n")
        count += 1
size_mb = os.path.getsize(output_path) / 1024 / 1024
print(f"  Saved {count} samples → {output_path} ({size_mb:.1f} MB)")
PYEOF

    echo "  Done: ii_7M_instruct.jsonl"
}

generate_ii_gen_1_4M_instruct() {
    echo ""
    echo "--- Generating ii_gen_1.4M_instruct.jsonl (BAAI/Infinity-Instruct, Gen, ~1.4M) ---"

    python - <<'PYEOF'
import json, os
from datasets import load_dataset
from tqdm import tqdm

output_dir = os.environ.get("OUTPUT_DIR", ".")
role_map = {"human": "user", "gpt": "assistant"}

def convert(conv):
    return [{"role": role_map.get(t["from"], t["from"]), "content": t["value"]} for t in conv]

output_path = os.path.join(output_dir, "ii_gen_1.4M_instruct.jsonl")
print("  Loading BAAI/Infinity-Instruct (Gen) ...")
ds = load_dataset("BAAI/Infinity-Instruct", "Gen", split="train")
count = 0
with open(output_path, "w", encoding="utf-8") as f:
    for ex in tqdm(ds, desc="  Converting Gen"):
        convs = ex.get("conversations")
        if not convs or not isinstance(convs, list):
            continue
        msgs = convert(convs)
        f.write(json.dumps({"messages": msgs}, ensure_ascii=False) + "\n")
        count += 1
size_mb = os.path.getsize(output_path) / 1024 / 1024
print(f"  Saved {count} samples → {output_path} ({size_mb:.1f} MB)")
PYEOF

    echo "  Done: ii_gen_1.4M_instruct.jsonl"
}

generate_tulu_0_6M_instruct() {
    echo ""
    echo "--- Generating tulu_0.6M_instruct.jsonl (allenai/tulu-v3.1-mix-preview-4096-OLMoE, ~0.61M) ---"

    python - <<'PYEOF'
import json, os
from datasets import load_dataset
from tqdm import tqdm

output_dir = os.environ.get("OUTPUT_DIR", ".")
output_path = os.path.join(output_dir, "tulu_0.6M_instruct.jsonl")

print("  Loading allenai/tulu-v3.1-mix-preview-4096-OLMoE ...")
ds = load_dataset("allenai/tulu-v3.1-mix-preview-4096-OLMoE", split="train")

count = 0
with open(output_path, "w", encoding="utf-8") as f:
    for ex in tqdm(ds, desc="  Converting tulu"):
        msgs = None
        if "messages" in ex and isinstance(ex["messages"], list):
            msgs = ex["messages"]
        if msgs is None:
            continue
        f.write(json.dumps({"messages": msgs}, ensure_ascii=False) + "\n")
        count += 1

size_mb = os.path.getsize(output_path) / 1024 / 1024
print(f"  Saved {count} samples → {output_path} ({size_mb:.1f} MB)")
PYEOF

    echo "  Done: tulu_0.6M_instruct.jsonl"
}

generate_am_1_4M_instruct() {
    echo ""
    echo "--- Generating am_1.4M_instruct.jsonl (AM-DeepSeek-R1-Distilled-1.4M, ~1.4M) ---"

    python - <<'PYEOF'
import json, os
from huggingface_hub import hf_hub_download
from tqdm import tqdm

output_dir = os.environ.get("OUTPUT_DIR", ".")
output_path = os.path.join(output_dir, "am_1.4M_instruct.jsonl")

REPO = "a-m-team/AM-DeepSeek-R1-Distilled-1.4M"
FILES = ["am_0.5M.jsonl", "am_0.9M.jsonl"]

def clean_messages(msgs):
    """Extract user/system content and assistant answer_content, discarding think."""
    if not msgs or not isinstance(msgs, list):
        return None
    clean = []
    for msg in msgs:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        info = msg.get("info") if isinstance(msg, dict) else None
        if role == "assistant":
            content = (info.get("answer_content") or "") if isinstance(info, dict) else ""
        else:
            content = msg.get("content", "")
        clean.append({"role": role, "content": content})
    return clean

count = 0
skipped = 0
with open(output_path, "w", encoding="utf-8") as f:
    for filename in FILES:
        print(f"  Downloading {filename} ...")
        local_path = hf_hub_download(REPO, filename, repo_type="dataset")
        with open(local_path, "r", encoding="utf-8") as fh:
            for line in tqdm(fh, desc=f"  Processing {filename}"):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    skipped += 1
                    continue
                msgs = clean_messages(row.get("messages"))
                if not msgs:
                    skipped += 1
                    continue
                f.write(json.dumps({"messages": msgs}, ensure_ascii=False) + "\n")
                count += 1

size_mb = os.path.getsize(output_path) / 1024 / 1024
print(f"  Saved {count} samples (skipped {skipped}) → {output_path} ({size_mb:.1f} MB)")
PYEOF

    echo "  Done: am_1.4M_instruct.jsonl"
}

generate_am_0_9M_sample_1k() {
    echo ""
    echo "--- Generating am_0.9M_sample_1k.jsonl (AM sample, ~1k) ---"

    python - <<'PYEOF'
import json, os
from huggingface_hub import hf_hub_download
from tqdm import tqdm

output_dir = os.environ.get("OUTPUT_DIR", ".")
output_path = os.path.join(output_dir, "am_0.9M_sample_1k.jsonl")

REPO = "a-m-team/AM-DeepSeek-R1-Distilled-1.4M"
FILENAME = "am_0.9M_sample_1k.jsonl"

def clean_messages(msgs):
    if not msgs or not isinstance(msgs, list):
        return None
    clean = []
    for msg in msgs:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        info = msg.get("info") if isinstance(msg, dict) else None
        if role == "assistant":
            content = (info.get("answer_content") or "") if isinstance(info, dict) else ""
        else:
            content = msg.get("content", "")
        clean.append({"role": role, "content": content})
    return clean

print(f"  Downloading {FILENAME} ...")
local_path = hf_hub_download(REPO, FILENAME, repo_type="dataset")

count = 0
skipped = 0
with open(output_path, "w", encoding="utf-8") as f:
    with open(local_path, "r", encoding="utf-8") as fh:
        for line in tqdm(fh, desc=f"  Processing {FILENAME}"):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            msgs = clean_messages(row.get("messages"))
            if not msgs:
                skipped += 1
                continue
            f.write(json.dumps({"messages": msgs}, ensure_ascii=False) + "\n")
            count += 1

size_mb = os.path.getsize(output_path) / 1024 / 1024
print(f"  Saved {count} samples (skipped {skipped}) → {output_path} ({size_mb:.1f} MB)")
PYEOF

    echo "  Done: am_0.9M_sample_1k.jsonl"
}

generate_task_0_2M_instruct() {
    echo ""
    echo "--- Generating task_0.2M_instruct.jsonl (Mixed Downstream Dataset, ~0.2M) ---"

    python - <<'PYEOF'
import json, os, re, random
from pathlib import Path
from datasets import load_dataset
from jinja2 import Template
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Embedded utility functions
# ---------------------------------------------------------------------------

def _hellaswag_preprocess(text):
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    text = text.replace("  ", " ")
    return text

def hellaswag_process_docs(dataset):
    def _proc(doc):
        ctx = doc["ctx_a"] + " " + doc["ctx_b"].capitalize()
        return {
            "query": _hellaswag_preprocess(doc["activity_label"] + ": " + ctx),
            "choices": [_hellaswag_preprocess(e) for e in doc["endings"]],
            "gold": int(doc["label"]),
        }
    return dataset.map(_proc)

def winogrande_doc_to_text(doc):
    if not doc.get("answer") or doc["answer"] == "":
        return None
    idx = doc["sentence"].index("_")
    answer_to_num = {"1": 0, "2": 1}
    target_idx = answer_to_num[doc["answer"]]
    option = doc["option1"] if target_idx == 0 else doc["option2"]
    return doc["sentence"][:idx].rstrip() + " " + option

def winogrande_doc_to_target(doc):
    if not doc.get("answer") or doc["answer"] == "":
        return None
    return {"1": 0, "2": 1}[doc["answer"]]

def winogrande_doc_to_choice(doc):
    idx = doc["sentence"].index("_")
    remaining = doc["sentence"][idx+1:].lstrip()
    return [remaining, remaining]

def _ethics_util_preproc(doc):
    rnd = random.Random(doc["activity"])
    scenarios = [doc["activity"], doc["baseline"]]
    ordering = [0, 1]
    rnd.shuffle(ordering)
    return {
        "scenarios": [scenarios[ordering[0]], scenarios[ordering[1]]],
        "label": int(ordering.index(0) == 0),
    }

def ethics_util_doc_to_text(doc):
    d = _ethics_util_preproc(doc)
    return f"Scenario 1: {d['scenarios'][0]}\nScenario 2: {d['scenarios'][1]}\nQuestion: Is Scenario 1 preferable?\nAnswer:"

def ethics_util_doc_to_target(doc):
    return _ethics_util_preproc(doc)["label"]

DATASET_CONFIGS = {
    "hendrycks_ethics_cm": {
        "dataset_path": "EleutherAI/hendrycks_ethics", "dataset_name": "commonsense",
        "training_split": "train",
        "doc_to_text": "{{input}}\nQuestion: Is this wrong?\nAnswer:",
        "doc_to_target": "label", "doc_to_choice": ["no", "yes"],
    },
    "hendrycks_ethics_deontology": {
        "dataset_path": "EleutherAI/hendrycks_ethics", "dataset_name": "deontology",
        "training_split": "train",
        "doc_to_text": 'Question: Would most people believe this reasonable or unreasonable to say? "{{scenario}} {{excuse.rstrip()}}"\nAnswer:',
        "doc_to_target": "label", "doc_to_choice": ["unreasonable", "reasonable"],
    },
    "hendrycks_ethics_justice": {
        "dataset_path": "EleutherAI/hendrycks_ethics", "dataset_name": "justice",
        "training_split": "train",
        "doc_to_text": 'Question: Would most people believe this reasonable or unreasonable to say? "{{scenario}}"\nAnswer:',
        "doc_to_target": "label", "doc_to_choice": ["unreasonable", "reasonable"],
    },
    "hendrycks_ethics_utilitarianism": {
        "dataset_path": "EleutherAI/hendrycks_ethics", "dataset_name": "utilitarianism",
        "training_split": "train",
        "doc_to_text": ethics_util_doc_to_text,
        "doc_to_target": ethics_util_doc_to_target, "doc_to_choice": ["no", "yes"],
    },
    "hendrycks_ethics_virtue": {
        "dataset_path": "EleutherAI/hendrycks_ethics", "dataset_name": "virtue",
        "training_split": "train",
        "doc_to_text": 'Sentence: {{scenario}}\nQuestion: Does the character in this sentence exhibit the trait "{{trait}}"?\nAnswer:',
        "doc_to_target": "label", "doc_to_choice": ["no", "yes"],
    },
    "arc_e": {
        "dataset_path": "allenai/ai2_arc", "dataset_name": "ARC-Easy",
        "training_split": "train",
        "doc_to_text": "Question: {{question}}\nAnswer:",
        "doc_to_target": "{{choices.label.index(answerKey)}}",
        "doc_to_choice": "{{choices.text}}",
    },
    "arc_c": {
        "dataset_path": "allenai/ai2_arc", "dataset_name": "ARC-Challenge",
        "training_split": "train",
        "doc_to_text": "Question: {{question}}\nAnswer:",
        "doc_to_target": "{{choices.label.index(answerKey)}}",
        "doc_to_choice": "{{choices.text}}",
    },
    "hellaswag": {
        "dataset_path": "Rowan/hellaswag", "dataset_name": None,
        "training_split": "train",
        "process_docs": hellaswag_process_docs,
        "doc_to_text": "{{query}}", "doc_to_target": "{{gold}}", "doc_to_choice": "{{choices}}",
    },
    "boolq": {
        "dataset_path": "super_glue", "dataset_name": "boolq",
        "training_split": "train",
        "doc_to_text": "{{passage}}\nQuestion: {{question}}?\nAnswer:",
        "doc_to_target": "{{label}}", "doc_to_choice": ["no", "yes"],
    },
    "piqa": {
        "dataset_path": "baber/piqa", "dataset_name": None,
        "training_split": "train",
        "doc_to_text": "Question: {{goal}}\nAnswer:",
        "doc_to_target": "{{label}}", "doc_to_choice": "{{[sol1, sol2]}}",
    },
    "winogrande": {
        "dataset_path": "winogrande", "dataset_name": "winogrande_xl",
        "training_split": "train",
        "doc_to_text": winogrande_doc_to_text,
        "doc_to_target": winogrande_doc_to_target,
        "doc_to_choice": winogrande_doc_to_choice,
    },
    "social_iqa": {
        "dataset_path": "social_i_qa", "dataset_name": None,
        "training_split": "train",
        "doc_to_text": "Q: {{context}} {{question}}\nA:",
        "target_delimiter": " ",
        "doc_to_choice": "{{[answerA, answerB, answerC]}}",
        "doc_to_target": "{{ (label|int) - 1 }}",
    },
    "openbookqa": {
        "dataset_path": "openbookqa", "dataset_name": "main",
        "training_split": "train",
        "doc_to_text": "{{question_stem}}",
        "doc_to_target": "{{choices.label.index(answerKey.lstrip())}}",
        "doc_to_choice": "{{choices.text}}",
    },
}

def render_template(template_str, doc):
    if callable(template_str):
        return template_str(doc)
    if isinstance(template_str, list):
        return template_str
    try:
        return Template(template_str).render(**doc)
    except Exception:
        if "{{" in template_str and "}}" in template_str:
            expr = template_str.split("{{")[1].split("}}")[0].strip()
            try:
                return eval(expr, {"__builtins__": {}}, doc)
            except Exception:
                return template_str
        return template_str

def process_document(doc, config):
    question_text = render_template(config["doc_to_text"], doc)
    if question_text is None:
        return None

    if callable(config["doc_to_choice"]):
        choices = config["doc_to_choice"](doc)
    else:
        choices_result = render_template(config["doc_to_choice"], doc)
        if isinstance(choices_result, str) and choices_result.startswith("["):
            try:
                choices = eval(choices_result)
            except Exception:
                choices = choices_result
        else:
            choices = choices_result

    if callable(config["doc_to_target"]):
        target_idx = config["doc_to_target"](doc)
    else:
        dt = config["doc_to_target"]
        if isinstance(dt, str) and "{{" not in dt and dt in doc:
            target_idx = doc[dt]
        else:
            target_result = render_template(dt, doc)
            if isinstance(target_result, int):
                target_idx = target_result
            elif isinstance(target_result, str) and target_result.isdigit():
                target_idx = int(target_result)
            else:
                try:
                    target_idx = eval(target_result) if isinstance(target_result, str) else target_result
                except Exception:
                    target_idx = target_result

    if target_idx is None or target_idx == "":
        return None

    if isinstance(choices, list) and len(choices) == 2 and set(choices) == {"no", "yes"}:
        try:
            answer_text = choices[int(target_idx)]
        except Exception:
            answer_text = str(target_idx)
    elif isinstance(choices, list) and isinstance(target_idx, int) and 0 <= target_idx < len(choices):
        answer_text = choices[target_idx]
    elif isinstance(choices, list):
        try:
            answer_text = choices[int(target_idx)]
        except Exception:
            answer_text = str(choices)
    else:
        answer_text = str(choices)

    return {
        "messages": [
            {"role": "user", "content": question_text},
            {"role": "assistant", "content": " " + answer_text},
        ]
    }

# ---------------------------------------------------------------------------
# Main: generate per-task training JSONLs, then combine
# ---------------------------------------------------------------------------

output_dir = os.environ.get("OUTPUT_DIR", ".")
processed_dir = os.path.join(output_dir, "processed")
Path(processed_dir).mkdir(parents=True, exist_ok=True)

for task_name, config in DATASET_CONFIGS.items():
    print(f"  Processing {task_name} ...")
    if config["dataset_name"]:
        dataset = load_dataset(config["dataset_path"], config["dataset_name"])
    else:
        dataset = load_dataset(config["dataset_path"])

    process_docs_fn = config.get("process_docs")
    split_name = config.get("training_split")
    if split_name and split_name in dataset:
        output_filename = f"{task_name}_training.jsonl"
        output_file = os.path.join(processed_dir, output_filename)

        split_data = dataset[split_name]
        if task_name == "hellaswag":
            split_data = split_data.filter(lambda x: str(x.get("label", "")).isdigit())
        if process_docs_fn:
            split_data = process_docs_fn(split_data)

        processed_count = 0
        with open(output_file, "w", encoding="utf-8") as f:
            for doc in split_data:
                try:
                    processed = process_document(doc, config)
                    if processed is None:
                        continue
                    f.write(json.dumps(processed, ensure_ascii=False) + "\n")
                    processed_count += 1
                except Exception:
                    continue
        print(f"    {output_filename}: {processed_count} samples")

# Combine all *_training.jsonl → task_0.2M_instruct.jsonl
print("  Combining training splits → task_0.2M_instruct.jsonl ...")

combined_path = os.path.join(output_dir, "task_0.2M_instruct.jsonl")
training_files = sorted(Path(processed_dir).glob("*_training.jsonl"))

total = 0
stats = {}
with open(combined_path, "w", encoding="utf-8") as outf:
    for tf in training_files:
        ds_name = tf.stem
        if ds_name.endswith("_training"):
            ds_name = ds_name[:-len("_training")]
        cnt = 0
        with open(tf, "r", encoding="utf-8") as inf:
            for line in inf:
                line = line.strip()
                if line:
                    try:
                        json.loads(line)
                        outf.write(line + "\n")
                        cnt += 1
                    except json.JSONDecodeError:
                        pass
        stats[ds_name] = cnt
        total += cnt

size_mb = os.path.getsize(combined_path) / 1024 / 1024
print(f"  Combined {total} training samples → {combined_path} ({size_mb:.1f} MB)")
print("  Per-dataset breakdown:")
for ds, cnt in sorted(stats.items()):
    pct = (cnt / total * 100) if total > 0 else 0
    print(f"    {ds}: {cnt} samples ({pct:.2f}%)")
PYEOF

    echo "  Done: task_0.2M_instruct.jsonl"
}

generate_ii_1_5M_base() {
    echo ""
    echo "--- Generating ii_1.5M_base.jsonl (BAAI/Infinity-Instruct, 7M_core, ~1.48M) ---"

    python - <<'PYEOF'
import json, os
from datasets import load_dataset
from tqdm import tqdm

output_dir = os.environ.get("OUTPUT_DIR", ".")
role_map = {"human": "user", "gpt": "assistant"}

def convert_base(conv):
    result = []
    for t in conv:
        role = role_map.get(t["from"], t["from"])
        content = t["value"]
        if role == "user":
            content = content + "\nAnswer:"
        elif role == "assistant":
            content = " " + content
        result.append({"role": role, "content": content})
    return result

output_path = os.path.join(output_dir, "ii_1.5M_base.jsonl")
print("  Loading BAAI/Infinity-Instruct (7M_core) ...")
ds = load_dataset("BAAI/Infinity-Instruct", "7M_core", split="train")
count = 0
with open(output_path, "w", encoding="utf-8") as f:
    for ex in tqdm(ds, desc="  Converting 7M_core"):
        convs = ex.get("conversations")
        if not convs or not isinstance(convs, list):
            continue
        msgs = convert_base(convs)
        f.write(json.dumps({"messages": msgs}, ensure_ascii=False) + "\n")
        count += 1
size_mb = os.path.getsize(output_path) / 1024 / 1024
print(f"  Saved {count} samples → {output_path} ({size_mb:.1f} MB)")
PYEOF

    echo "  Done: ii_1.5M_base.jsonl"
}

# =============================================================================
# Execute
# =============================================================================

if should_generate "ii_7M_instruct.jsonl"; then
    generate_ii_7M_instruct
fi

if should_generate "ii_gen_1.4M_instruct.jsonl"; then
    generate_ii_gen_1_4M_instruct
fi

if should_generate "tulu_0.6M_instruct.jsonl"; then
    generate_tulu_0_6M_instruct
fi

if should_generate "am_1.4M_instruct.jsonl"; then
    generate_am_1_4M_instruct
fi

if should_generate "am_0.9M_sample_1k.jsonl"; then
    generate_am_0_9M_sample_1k
fi

if should_generate "task_0.2M_instruct.jsonl"; then
    generate_task_0_2M_instruct
fi

if should_generate "ii_1.5M_base.jsonl"; then
    generate_ii_1_5M_base
fi

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "Summary of generated files:"
echo "============================================"
for f in "${VALID_DATASETS[@]}"; do
    fp="$OUTPUT_DIR/$f"
    if [ -f "$fp" ]; then
        lines=$(wc -l < "$fp" | tr -d ' ')
        size=$(du -h "$fp" | cut -f1)
        printf "  %-35s %8s samples  %6s\n" "$f" "$lines" "$size"
    else
        printf "  %-35s %s\n" "$f" "(not generated)"
    fi
done
echo "============================================"
echo "Done."
