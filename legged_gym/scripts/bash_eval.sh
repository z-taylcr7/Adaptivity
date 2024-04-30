python legged_gym/scripts/eval.py  --task=a1 --headless --rl_device=cuda:1 --sim_device=cuda:0 --load_run="dual_transformer/y=12_l=66_v=3.0_direct"
python legged_gym/scripts/eval.py  --task=a1 --headless --rl_device=cuda:1 --sim_device=cuda:0 --load_run="dual_cnn/y=12_l=66_v=3.0"
python legged_gym/scripts/eval.py  --task=a1 --headless --rl_device=cuda:1 --sim_device=cuda:0 --load_run="dual_lstm/y=12_l=66_v=3.0"
python legged_gym/scripts/eval.py  --task=a1 --headless --rl_device=cuda:1 --sim_device=cuda:0 --load_run="dual_gru/y=12_l=66_v=3.0"