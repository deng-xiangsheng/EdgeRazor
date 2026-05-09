
import argparse
import torch
import os
from transformers import Qwen2_5OmniForConditionalGeneration
from transformers import Qwen2_5OmniThinkerForConditionalGeneration

def parse_args():
	parser = argparse.ArgumentParser(description="Combine total model and thinker weights.")
	parser.add_argument('--input_model_path', type=str, required=True, help='Path to thinker weights (checkpoint)')
	parser.add_argument('--output_model_path', type=str, required=True, help='Path to save combined model')
	parser.add_argument('--total_model_path', type=str, default=None, help='Path to total model (default: use Qwen2.5-Omni-7B from HF)')
	parser.add_argument('--device', type=str, default='cpu', help='Device to load model')
	return parser.parse_args()

def main():
	args = parse_args()

	# 1. Load total_model
	if args.total_model_path:
		print(f"Loading total model from: {args.total_model_path}")
		model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
			args.total_model_path,
			torch_dtype=torch.bfloat16,
			device_map=None
		)
	else:
		print("Loading total model from HuggingFace: Qwen/Qwen2.5-Omni-7B")
		model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
			"Qwen/Qwen2.5-Omni-7B",
			torch_dtype=torch.bfloat16,
			device_map=None
		)


	# 2. Load thinker model (supports directory or single weight file)
	print(f"Loading thinker model from: {args.input_model_path}")
	if os.path.isdir(args.input_model_path):
		thinker = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
			args.input_model_path,
			torch_dtype=torch.bfloat16,
			device_map=None
		)
		thinker_state = thinker.state_dict()
	elif args.input_model_path.endswith('.bin'):
		thinker_state = torch.load(args.input_model_path, map_location='cpu')
	else:
		raise ValueError('input_model_path must be a thinker model directory or pytorch_model.bin file')

	# 3. Overwrite total_model weights
	print("Merging weights...")
	model.load_state_dict(thinker_state, strict=False)

	# 4. Save combined model
	print(f"Saving combined model to: {args.output_model_path}")
	os.makedirs(args.output_model_path, exist_ok=True)
	model.save_pretrained(args.output_model_path)
	print("Done.")

if __name__ == "__main__":
	main()
    # python partial_combine.py --device cuda:0 --input_model_path /path/to/checkpoint --output_model_path /path/to/final_model
