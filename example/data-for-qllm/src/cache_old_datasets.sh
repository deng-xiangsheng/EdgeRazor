#!/bin/bash

echo "Installing datasets==2.17.0 for dataset download..."
pip install datasets==2.17.0

python - <<'EOF'
from datasets import load_dataset, get_dataset_config_names
import sys

def download_all_configs(dataset_name, dataset_path):
    """Download all configs for a given dataset and print summary info."""
    print("=" * 70)
    print(f"Processing: {dataset_name}")
    print("=" * 70)
    
    try:
        configs = get_dataset_config_names(dataset_path)
        print(f"\nFound {len(configs)} config(s): {configs}")
        
        for config in configs:
            try:
                print(f"\n  → Downloading config: {config}")
                dataset = load_dataset(dataset_path, config)
                splits = list(dataset.keys())
                print(f"    ✓ Config '{config}' downloaded successfully")
                print(f"    Splits: {splits}")
                for split in splits:
                    num_examples = len(dataset[split])
                    print(f"      - {split}: {num_examples} examples")
            except Exception as e:
                print(f"    ✗ Error downloading config '{config}': {e}")
        
        return True
    except Exception as e:
        print(f"\nNo configs found, trying direct download...")
        try:
            dataset = load_dataset(dataset_path)
            splits = list(dataset.keys())
            print(f"  ✓ Dataset downloaded successfully")
            print(f"  Splits: {splits}")
            for split in splits:
                num_examples = len(dataset[split])
                print(f"    - {split}: {num_examples} examples")
            return True
        except Exception as e2:
            print(f"  ✗ Error: {e2}")
            return False

success1 = download_all_configs("EleutherAI/hendrycks_ethics", "EleutherAI/hendrycks_ethics")

success2 = download_all_configs("social_i_qa", "social_i_qa")

print("\n" + "=" * 70)
print("DOWNLOAD SUMMARY")
print("=" * 70)
print(f"EleutherAI/hendrycks_ethics: {'✓ SUCCESS' if success1 else '✗ FAILED'}")
print(f"social_i_qa: {'✓ SUCCESS' if success2 else '✗ FAILED'}")
print("=" * 70)

if success1 and success2:
    print("\n🎉 All datasets downloaded successfully!")
    sys.exit(0)
else:
    print("\n⚠️  Some datasets failed to download")
    sys.exit(1)
EOF

exit_code=$?

echo -e "\nUpgrading datasets to latest version..."
pip install -U datasets

if [ $exit_code -eq 0 ]; then
    echo -e "\n✓ Setup complete! All datasets are cached and ready to use."
else
    echo -e "\n⚠️  Setup completed with some errors. Check the output above."
fi

exit $exit_code