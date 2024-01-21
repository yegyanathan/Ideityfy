python main.py fit --seed_everything 999 --model.model_name vit_base_patch16_clip_224 --model.use_lora True --model.rank 4 \
                   --model.learning_rate 0.003 --model.weight_decay 0.001 --model.push_to_hf True \
                   --model.commit_message "Push model to Huggingface" --model.repo_id Yegiiii/ideityfy --data.ds_name deities-v0 \
                   --data.batch_size 12 --data.num_workers 8 --trainer.accelerator mps --trainer.devices 1 \
                   --trainer.logger.class_path WandbLogger --trainer.logger.init_args.project Ideityfy \
                   --trainer.logger.init_args.log_model True --trainer.min_epochs 1 --trainer.max_epochs 5 \
                   --trainer.enable_checkpointing True --trainer.enable_progress_bar True --trainer.enable_model_summary True \
                   --trainer.accumulate_grad_batches 16 --trainer.callbacks ModelCheckpoint \
                   --trainer.callbacks.init_args.dirpath checkpoints --trainer.callbacks.init_args.filename vit_base_clip_rank4 \
                   --trainer.callbacks.init_args.monitor top1_acc --trainer.callbacks.init_args.mode max 
