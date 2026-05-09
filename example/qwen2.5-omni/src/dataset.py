# ruff: noqa: F401

import json
import logging
import random
from typing import Any

import torch
from torch.utils.data import Dataset
from transformers import Qwen2_5OmniProcessor

try:
    from qwen_omni_utils import process_mm_info
    QWEN_OMNI_UTILS_AVAILABLE = True
except ImportError:
    QWEN_OMNI_UTILS_AVAILABLE = False
    process_mm_info = None

try:
    import matplotlib.pyplot as plt
    import numpy as np
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100

# System prompt for objective video descriptions
SYSTEM_PROMPT = (
    "You are a video captioning system that outputs factual descriptions only. "
    "Rules: "
    "1. Describe only what is visually present: objects, actions, movements, scenes. "
    "2. Use objective language without subjective opinions. "
    "3. NEVER end with questions or ask for user feedback. "
    "4. End the description with the final scene, not with commentary."
)

# Prompt for video descriptions
PROMPT = (
    "Describe what happens in this video from beginning to end. "
    "Include the opening scene, main events, and how it concludes. "
    "Write a single factual paragraph. Do not ask any questions."
)


class MultimodalDataset(Dataset):
    """
    Qwen2.5-Omni multimodal dataset (Video + Text)

    Key relationship:
    - pixel_values_videos.shape[0] == Σ(T·H·W) from video_grid_thw
    - No direct proportional relationship with <|VIDEO|> token count
    """
    
    # Maximum retry attempts on sampling failure
    MAX_RETRY_ATTEMPTS = 3
    
    def __init__(
        self,
        dataset_path: str | list[str],
        processor: Qwen2_5OmniProcessor,
        max_seq_len: int = 4096,
        add_system_prompt: bool = False,
        limit_num_samples: int | None = None,
        use_audio_in_video: bool = False,
        validate: bool = True,
    ):
        """
        Args:
            dataset_path: JSONL/JSON file path or list of paths
            processor: Qwen2_5OmniProcessor instance
            max_seq_len: Maximum sequence length
            add_system_prompt: Whether to add system prompt
            limit_num_samples: Limit number of samples (for debugging)
            use_audio_in_video: Whether to use audio in video
            validate: Whether to validate video patches and grid consistency
        """
        super().__init__()
        
        # Check critical dependencies
        if not QWEN_OMNI_UTILS_AVAILABLE:
            raise ImportError(
                "qwen_omni_utils is required for video processing. "
                "Please install it from the Qwen2.5-Omni repository."
            )
        
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.max_seq_len = max_seq_len
        self.add_system_prompt = add_system_prompt
        self.use_audio_in_video = use_audio_in_video
        self.validate = validate
        
        # Token IDs
        self.eos_token_id = self.tokenizer.eos_token_id
        self.pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        
        logger.info(f"Token IDs - EOS: {self.eos_token_id}, PAD: {self.pad_token_id}")

        # Load data
        self.data = self._load_data(dataset_path)
        logger.info(f"Loaded {len(self.data)} samples")
        
        # Random sampling
        if limit_num_samples and 0 < limit_num_samples < len(self.data):
            random.seed(3407)
            self.data = random.sample(self.data, limit_num_samples)
            logger.info(f"Randomly sampled {limit_num_samples} samples")

    def _load_data(self, dataset_path: str | list[str]) -> list[dict]:
        """Load JSONL/JSON data"""
        data = []
        paths = [dataset_path] if isinstance(dataset_path, str) else dataset_path
        
        for path in paths:
            logger.info(f"Loading: {path}")
            with open(path, 'r', encoding='utf-8') as f:
                if path.endswith('.jsonl'):
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if line:
                            try:
                                data.append(json.loads(line))
                            except json.JSONDecodeError as e:
                                logger.warning(f"Skipping invalid JSON at {path}:{line_num}: {e}")
                else:
                    items = json.load(f)
                    data.extend(items if isinstance(items, list) else [items])
        
        return data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """
        Get a single sample with retry on failure
        """
        last_error = None
        
        for attempt in range(self.MAX_RETRY_ATTEMPTS):
            try:
                # First attempt uses the original index, subsequent attempts pick randomly
                current_index = index if attempt == 0 else random.randint(0, len(self.data) - 1)
                return self._process_sample(current_index)
            except Exception as e:
                last_error = e
                if attempt < self.MAX_RETRY_ATTEMPTS - 1:
                    logger.warning(
                        f"[{index}] Attempt {attempt + 1} failed: {e}. "
                        f"Retrying with different sample..."
                    )
        
        # All retries failed
        raise RuntimeError(
            f"Failed to process sample after {self.MAX_RETRY_ATTEMPTS} attempts. "
            f"Last error: {last_error}"
        )

    def _process_sample(self, index: int) -> dict[str, Any]:
        """Process a single sample"""
        item = self.data[index]
        messages = item['messages']
        
        # Build conversation
        if self.add_system_prompt:
            conversation = [{
                "role": "system",
                "content": [{"type": "text", "text": SYSTEM_PROMPT}]
            }] + messages
        else:
            conversation = messages

        # Generate full text
        full_text = self.processor.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=False
        )
        
        # Process multimodal content
        audios, images, videos = process_mm_info(
            conversation,
            use_audio_in_video=self.use_audio_in_video
        )
        
        # Processor processing
        inputs = self.processor(
            text=full_text,
            audio=audios or None,
            images=images or None,
            videos=videos or None,
            return_tensors="pt",
            padding=False,
            use_audio_in_video=self.use_audio_in_video
        )
        
        # Extract tensors, remove batch dimension
        input_ids = inputs['input_ids'].squeeze(0)  # [seq_len]
        attention_mask = inputs['attention_mask'].squeeze(0)
        
        # Validate video patches and grid consistency
        if self.validate and 'pixel_values_videos' in inputs:
            self._validate_video_patches(inputs, index)
        
        # Build labels
        labels = self._build_labels(input_ids, conversation)
        
        # Truncate (reserve one position for EOS)
        max_len = self.max_seq_len - 1
        if len(input_ids) > max_len:
            input_ids = input_ids[:max_len]
            attention_mask = attention_mask[:max_len]
            labels = labels[:max_len]

        # Add EOS
        input_ids = torch.cat([input_ids, torch.tensor([self.eos_token_id])])
        attention_mask = torch.cat([attention_mask, torch.tensor([1])])
        labels = torch.cat([labels, torch.tensor([self.eos_token_id])])

        # Build return dictionary
        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        
        # Video features (no batch dimension)
        if 'pixel_values_videos' in inputs:
            result['pixel_values_videos'] = inputs['pixel_values_videos']
            result['video_grid_thw'] = inputs['video_grid_thw']
        if 'video_second_per_grid' in inputs:
            result['video_second_per_grid'] = inputs['video_second_per_grid']
        
        # Image features
        if 'pixel_values' in inputs:
            result['pixel_values'] = inputs['pixel_values']
            result['image_grid_thw'] = inputs['image_grid_thw']
        
        # Audio features
        if 'input_features' in inputs:
            result['input_features'] = inputs['input_features'].squeeze(0)
            result['feature_attention_mask'] = inputs['feature_attention_mask'].squeeze(0)

        return result

    def _validate_video_patches(self, inputs: dict, index: int) -> None:
        """
        Validate that video patches count matches video_grid_thw

        Relationship: pixel_values_videos.shape[0] == Σ(T·H·W)
        """
        pixel_values_videos = inputs['pixel_values_videos']
        video_grid_thw = inputs['video_grid_thw']
        
        num_patches = pixel_values_videos.shape[0]
        expected_patches = sum(
            t * h * w for t, h, w in video_grid_thw.tolist()
        )
        
        if num_patches != expected_patches:
            raise ValueError(
                f"[{index}] Video patches mismatch! "
                f"pixel_values_videos.shape[0]={num_patches}, "
                f"Σ(T·H·W)={expected_patches}, "
                f"video_grid_thw={video_grid_thw.tolist()}"
            )

    def _build_labels(
        self,
        input_ids: torch.Tensor,
        conversation: list[dict],
    ) -> torch.Tensor:
        """
        Build labels, computing loss only for assistant responses

        Args:
            input_ids: Complete input_ids tensor
            conversation: Conversation list

        Returns:
            labels tensor, non-assistant parts filled with IGNORE_INDEX
        """
        seq_len = len(input_ids)
        labels = torch.full((seq_len,), IGNORE_INDEX, dtype=torch.long)
        input_ids_list = input_ids.tolist()
        
        temp_messages = []
        for msg in conversation:
            temp_messages.append(msg)
            
            if msg['role'] != 'assistant':
                continue
            
            # Extract assistant text
            content = msg['content']
            if isinstance(content, str):
                assistant_text = content
            elif isinstance(content, list):
                assistant_text = next(
                    (item['text'] for item in content 
                     if isinstance(item, dict) and item.get('type') == 'text'),
                    ''
                )
            else:
                continue
            
            if not assistant_text:
                continue
            
            # Calculate position: current full conversation vs conversation without current assistant
            current_text = self.processor.apply_chat_template(
                temp_messages, tokenize=False, add_generation_prompt=False
            )
            prev_text = self.processor.apply_chat_template(
                temp_messages[:-1], tokenize=False, add_generation_prompt=True
            )
            
            current_ids = self.tokenizer.encode(current_text, add_special_tokens=False)
            prev_ids = self.tokenizer.encode(prev_text, add_special_tokens=False)
            
            start_pos = len(prev_ids)
            end_pos = min(len(current_ids), seq_len)
            
            # Set labels
            for pos in range(start_pos, end_pos):
                labels[pos] = input_ids_list[pos]
        
        return labels

    def get_collate_config(self) -> dict[str, Any]:
        """Return configuration needed by the collate function"""
        return {
            'pad_token_id': self.pad_token_id,
        }
    
    def get_token_stats(self, plot_distribution=False, output_path="./sequence_length_distribution.png"):
        """
        Compute dataset statistics in a single pass (for multimodal data):
        - video_seq_len: Video token length
        - prompt_seq_len: Prompt (system + question) token length
        - input_seq_len: Total input token length
        - output_seq_len: Generated output token length (assistant response)
        - max_seq_len: Total token length (input + output)

        Args:
            plot_distribution: Whether to plot sequence length distribution
            output_path: Output path for the distribution plot

        Returns:
            dict: Dictionary containing various statistics
                - total_samples: Total number of samples
                - stats: List of statistics for each seq_len type
                - summary: Summary statistics
        """
        stats_list = []  # Store seq_len_info for each sample

        for i in range(len(self.data)):
            item = self.data[i]
            messages = item['messages']

            # Build full conversation
            if self.add_system_prompt:
                conversation = [{
                    "role": "system",
                    "content": [{"type": "text", "text": SYSTEM_PROMPT}]
                }] + messages
            else:
                conversation = messages

            # Extract text length of assistant response
            output_text = ""
            for msg in messages:
                if msg['role'] == 'assistant':
                    if isinstance(msg['content'], list):
                        for content_item in msg['content']:
                            if isinstance(content_item, dict) and content_item.get('type') == 'text':
                                output_text = content_item.get('text', '')
                                break
                    elif isinstance(msg['content'], str):
                        output_text = msg['content']
                    break

            # Calculate output_seq_len (token length of assistant response)
            output_seq_len = len(self.tokenizer.encode(output_text, add_special_tokens=False)) if output_text else 0

            # Generate full text sequence
            full_text = self.processor.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=False
            )
            
            # Calculate prompt_seq_len (text-only part, excluding video)
            prompt_only_ids = self.tokenizer.encode(full_text, add_special_tokens=False)
            prompt_seq_len = len(prompt_only_ids) - output_seq_len  # Total text - output = prompt

            # Process multimodal info to get full input_ids length
            try:
                audios, images, videos = process_mm_info(
                    conversation,
                    use_audio_in_video=self.use_audio_in_video
                )
                
                inputs = self.processor(
                    text=full_text,
                    audio=audios,
                    images=images,
                    videos=videos,
                    return_tensors="pt",
                    padding=False,
                    use_audio_in_video=self.use_audio_in_video
                )
                
                input_seq_len = inputs['input_ids'].shape[1]
            except Exception:
                # If processing fails, use text length as estimate
                input_seq_len = len(prompt_only_ids)
            
            # Calculate video_seq_len (multimodal tokens - text tokens)
            video_seq_len = input_seq_len - len(prompt_only_ids)
            if video_seq_len < 0:
                video_seq_len = 0
            
            # max_seq_len = input_seq_len (already includes output)
            max_seq_len = input_seq_len + 1  # +1 for EOS token
            
            seq_len_info = {
                'video_seq_len': video_seq_len,
                'prompt_seq_len': prompt_seq_len,
                'input_seq_len': input_seq_len,
                'output_seq_len': output_seq_len,
                'max_seq_len': max_seq_len,
            }
            stats_list.append(seq_len_info)

        # Plot distribution
        if plot_distribution:
            self._plot_dist_seqlen(stats_list, output_path)

        # Compute summary statistics
        if MATPLOTLIB_AVAILABLE:
            import numpy as np
            video_lens = np.array([s['video_seq_len'] for s in stats_list])
            prompt_lens = np.array([s['prompt_seq_len'] for s in stats_list])
            input_lens = np.array([s['input_seq_len'] for s in stats_list])
            output_lens = np.array([s['output_seq_len'] for s in stats_list])
            max_lens = np.array([s['max_seq_len'] for s in stats_list])
            
            summary = {
                'total_samples': len(stats_list),
                'video_seq_len': {'mean': np.mean(video_lens), 'max': np.max(video_lens), 'min': np.min(video_lens)},
                'prompt_seq_len': {'mean': np.mean(prompt_lens), 'max': np.max(prompt_lens), 'min': np.min(prompt_lens)},
                'input_seq_len': {'mean': np.mean(input_lens), 'max': np.max(input_lens), 'min': np.min(input_lens)},
                'output_seq_len': {'mean': np.mean(output_lens), 'max': np.max(output_lens), 'min': np.min(output_lens)},
                'max_seq_len': {'mean': np.mean(max_lens), 'max': np.max(max_lens), 'min': np.min(max_lens)},
                'truncated_samples': np.sum(max_lens > self.max_seq_len),
            }
        else:
            summary = {'total_samples': len(stats_list)}

        return {
            'total_samples': len(stats_list),
            'stats': stats_list,
            'summary': summary
        }

    def _plot_dist_seqlen(self, stats_list: list, output_path: str):
        """
        Plot sequence length distribution (1x5 layout)

        Args:
            stats_list: List containing seq_len_info
            output_path: Output image path
        """
        if not MATPLOTLIB_AVAILABLE:
            print("Warning: matplotlib is not installed, cannot plot distribution. Please run: pip install matplotlib")
            return

        import matplotlib.pyplot as plt
        import numpy as np

        # Extract various length data
        video_lens = np.array([s['video_seq_len'] for s in stats_list])
        prompt_lens = np.array([s['prompt_seq_len'] for s in stats_list])
        input_lens = np.array([s['input_seq_len'] for s in stats_list])
        output_lens = np.array([s['output_seq_len'] for s in stats_list])
        max_lens = np.array([s['max_seq_len'] for s in stats_list])

        # Create 1x5 chart
        fig, axes = plt.subplots(1, 5, figsize=(25, 5))
        fig.suptitle('Sequence Length Distribution Analysis (Multimodal Dataset)', fontsize=16, fontweight='bold')

        # Define plot data
        plot_data = [
            ('Video Seq Len', video_lens, 'steelblue'),
            ('Prompt Seq Len', prompt_lens, 'forestgreen'),
            ('Input Seq Len', input_lens, 'darkorange'),
            ('Output Seq Len', output_lens, 'crimson'),
            ('Max Seq Len', max_lens, 'purple'),
        ]

        for ax, (title, data, color) in zip(axes, plot_data):
            # Plot histogram
            ax.hist(data, bins=50, color=color, edgecolor='black', alpha=0.7)
            
            # Add statistical lines
            ax.axvline(np.mean(data), color='red', linestyle='--', linewidth=2, 
                      label=f'Mean: {np.mean(data):.0f}')
            ax.axvline(np.median(data), color='green', linestyle=':', linewidth=2, 
                      label=f'Median: {np.median(data):.0f}')
            
            # For max_seq_len, add truncation line
            if title == 'Max Seq Len':
                ax.axvline(self.max_seq_len, color='black', linestyle='-', linewidth=2,
                          label=f'Truncate: {self.max_seq_len}')
            
            ax.set_xlabel('Token Count')
            ax.set_ylabel('Frequency')
            ax.set_title(f'{title}\n(Max: {np.max(data):.0f}, Min: {np.min(data):.0f})')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        # Print statistics summary
        print(f"\n{'='*80}")
        print("SEQUENCE LENGTH STATISTICS")
        print(f"{'='*80}")
        print(f"Total Samples: {len(stats_list):,}")
        print(f"\n{'Metric':<20} {'Mean':>10} {'Median':>10} {'Min':>10} {'Max':>10} {'Std':>10}")
        print("-" * 70)
        
        for title, data, _ in plot_data:
            print(f"{title:<20} {np.mean(data):>10.1f} {np.median(data):>10.1f} "
                  f"{np.min(data):>10.0f} {np.max(data):>10.0f} {np.std(data):>10.1f}")
        
        # Truncation statistics
        truncated = np.sum(max_lens > self.max_seq_len)
        print(f"\nTruncation Analysis (max_seq_len={self.max_seq_len}):")
        print(f"  Samples > max_seq_len: {truncated:,} ({truncated/len(stats_list)*100:.2f}%)")
        print(f"  Samples ≤ max_seq_len: {len(stats_list)-truncated:,} ({(len(stats_list)-truncated)/len(stats_list)*100:.2f}%)")
        print(f"{'='*80}")
        
        print(f"\n✓ Sequence length distribution plot saved to: {output_path}")
