accelerate launch --mixed_precision bf16 train.py --config-name=train_pusht_rollout_headless_dino encoder=vit_b16_random batch_size=128 num_workers=8 num_epochs=30 ssl_lr=1e-5 subset_fraction=0.1
