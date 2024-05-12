
python legged_gym/scripts/eval.py  --task=anymal_b --headless --rl_device=cuda:0 --sim_device=cuda:1 --load_run="rma_student"
python legged_gym/scripts/eval.py  --task=anymal_b --headless --rl_device=cuda:0 --sim_device=cuda:1 --load_run="dual_transformer/y=12_l=66_v=3.0"
python legged_gym/scripts/eval.py  --task=anymal_b --headless --rl_device=cuda:0 --sim_device=cuda:1 --load_run="dual_cnn/y=12_l=66_v=3.0"
python legged_gym/scripts/eval.py  --task=anymal_b --headless --rl_device=cuda:0 --sim_device=cuda:1 --load_run="dual_lstm/y=12_l=66_v=3.0"
python legged_gym/scripts/eval.py  --task=anymal_b --headless --rl_device=cuda:0 --sim_device=cuda:1 --load_run="dual_gru/y=12_l=66_v=3.0"

