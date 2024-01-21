#!/bin/bash

# Array of different rank values
rank_values=(3 4 5 6 7)

# Loop through each rank value
for rank in "${rank_values[@]}"
do
    echo "Running with rank=$rank"
    
    # Measure the time taken for script execution
    start_time=$(date +%s.%N)
    
    # Run the Python script with the current rank value
    python main.py fit \
        --seed_everything 1337 \
        --model.model_name vit_base_patch16_clip_224 \
        --model.use_lora True \
        --model.rank $rank \
        --model.learning_rate 0.003 \
        --model.weight_decay 0.001 \
        --data.ds_name deities-v0 \
        --data.batch_size 16 \
        --data.num_workers 8 \
        --trainer.accelerator mps \
        --trainer.devices 1 \
        --trainer.min_epochs 1 \
        --trainer.max_epochs 1 \
        --trainer.enable_progress_bar True \
        --trainer.enable_model_summary True \
        --trainer.accumulate_grad_batches 8 
    
    # Calculate and print the time taken
    end_time=$(date +%s.%N)
    elapsed_time=$(echo "$end_time - $start_time" | bc)
    echo "Time taken: $elapsed_time seconds"
    
    echo "--------------------------"
done

