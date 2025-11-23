mkdir -p runs1 runs2
bash run1.sh | tee ./runs1/$(date +%Y_%m_%d_%H_%M).log 2>&1
bash run2.sh | tee ./runs2/$(date +%Y_%m_%d_%H_%M).log 2>&1