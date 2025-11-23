"""
Generate performance tables from test results JSON files.

This script reads test_results.json from runs1 and runs2 directories,
and generates markdown tables with training time information.
"""
import json
from pathlib import Path


def load_test_results(json_path):
    """Load test results from JSON file."""
    if not json_path.exists():
        print(f"Error: {json_path} not found")
        return None
    
    with open(json_path) as f:
        return json.load(f)


def load_training_times(logdir):
    """Load training times from TensorBoard event files or checkpoint metadata."""
    training_times = {}
    
    checkpoint_dir = logdir / "checkpoints"
    if not checkpoint_dir.exists():
        return training_times
    
    # Try to load from each model's checkpoint
    for model_dir in checkpoint_dir.iterdir():
        if not model_dir.is_dir():
            continue
        
        config_name = model_dir.name
        checkpoint_path = model_dir / "best_model.pth"
        
        if checkpoint_path.exists():
            try:
                import torch
                checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
                
                # Try to get training time if available
                # This would need to be saved during training
                # For now, we'll just mark it as available
                training_times[config_name] = {
                    'epoch': checkpoint.get('epoch', 'N/A'),
                    'time': 'N/A'  # Would need to be saved during training
                }
            except Exception as e:
                print(f"Warning: Could not load {checkpoint_path}: {e}")
    
    return training_times


def format_params(params):
    """Format parameter count."""
    if params >= 1_000_000:
        return f"~{params // 1_000_000}M"
    elif params >= 1_000:
        return f"~{params // 1_000}K"
    else:
        return str(params)


def generate_table(results, logdir_name, training_times=None):
    """Generate markdown table from results."""
    if not results:
        print(f"No results found for {logdir_name}")
        return
    
    print(f"\n{'='*80}")
    print(f"{logdir_name.upper()} - PERFORMANCE RESULTS")
    print(f"{'='*80}\n")
    
    # Sort results by configuration
    results_sorted = sorted(results, key=lambda x: (
        0 if x['w_bits'] == 16 else 1,
        x['w_bits'] if isinstance(x['w_bits'], (int, float)) else 99,
        x['a_bits'],
        not x['distill']
    ))
    
    # Table header
    print("| Model    | Step | Training  | Distill      | Params | W      | A    | Prop   | Comp  | Accuracy | Recall | Precision | F1     |")
    print("| :------- | ---- | :-------- | ------------ | :----- | :----- | :--- | :----- | :---- | :------- | :----- | :-------- | :----- |")
    
    # Table rows
    for r in results_sorted:
        w_str = str(r['w_bits'])
        a_str = str(r['a_bits'])
        params_str = format_params(r['params'])
        prop_str = f"{r['proportion']*100:.2f}%" if r['proportion'] > 0 else "0%"
        comp_str = f"{r['compression']:.2f}x"
        distill_str = "$\\checkmark$" if r['distill'] else "$\\times$"
        
        # Get training time if available
        config_name = r['config_name']
        if training_times and config_name in training_times:
            time_str = training_times[config_name]['time']
        else:
            time_str = "..... min"
        
        # Format row
        row = (
            f"| ViT-S/16 | {r['epoch']:<4} | {time_str:<9} | {distill_str:<12} | "
            f"{params_str:<6} | {w_str:<6} | {a_str:<4} | {prop_str:<6} | {comp_str:<5} | "
            f"{r['accuracy']*100:>7.2f}% | {r['recall']*100:>6.2f}% | "
            f"{r['precision']*100:>8.2f}% | {r['f1']*100:>6.2f}% |"
        )
        print(row)
    
    print()


def main():
    """Main function to generate tables."""
    # Get script directory
    script_dir = Path(__file__).parent
    
    # Process runs1
    runs1_json = script_dir / "runs1" / "test_results.json"
    if runs1_json.exists():
        results1 = load_test_results(runs1_json)
        training_times1 = load_training_times(script_dir / "runs1")
        generate_table(results1, "runs1", training_times1)
    else:
        print(f"Warning: {runs1_json} not found. Run test script first:")
        print("  python -m src.test --logdir ./runs1")
        print()
    
    # Process runs2
    runs2_json = script_dir / "runs2" / "test_results.json"
    if runs2_json.exists():
        results2 = load_test_results(runs2_json)
        training_times2 = load_training_times(script_dir / "runs2")
        generate_table(results2, "runs2", training_times2)
    else:
        print(f"Warning: {runs2_json} not found. Run test script first:")
        print("  python -m src.test --logdir ./runs2")
        print()


if __name__ == "__main__":
    main()
