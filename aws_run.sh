accelerate launch --mixed_precision bf16 train.py --config-name=train_pusht_rollout_headless_dino encoder=vit_b16_random batch_size=128 num_workers=8
