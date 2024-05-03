
python legged_gym/scripts/eval.py  --task=a1 --headless --rl_device=cuda:4 --sim_device=cuda:5 --load_run="dual_mlp/y=12_l=66_v=3.0"
python legged_gym/scripts/eval.py  --task=a1 --headless --rl_device=cuda:4 --sim_device=cuda:5 --load_run="dual_transformer/y=12_l=66_v=3.0_t1"
python legged_gym/scripts/eval.py  --task=a1 --headless --rl_device=cuda:4 --sim_device=cuda:5 --load_run="dual_cnn/y=12_l=66_v=3.0_t1"
python legged_gym/scripts/eval.py  --task=a1 --headless --rl_device=cuda:4 --sim_device=cuda:5 --load_run="dual_lstm/y=12_l=66_v=3.0_t1"
python legged_gym/scripts/eval.py  --task=a1 --headless --rl_device=cuda:4 --sim_device=cuda:5 --load_run="dual_gru/y=12_l=66_v=3.0_t1"
python legged_gym/scripts/eval.py  --task=a1 --headless --rl_device=cuda:4 --sim_device=cuda:5 --load_run="dual_transformer/y=12_l=66_ter_on_thigh_calf"
python legged_gym/scripts/eval.py  --task=a1 --headless --rl_device=cuda:4 --sim_device=cuda:5 --load_run="dual_transformer/y=12_l=66_v=1.0"

