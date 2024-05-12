# Reinforcement Learning-based adaptive control for legged robots
### Train dual_history policy
```python
python legged_gym/scripts/train_dual.py --headless --task=a1 --sim_device=cuda:0 --rl_device=cuda:0
```
### Evaluation
```python
python legged_gym/scripts/eval.py --headless --task=a1 --sim_device=cuda:0 --rl_device=cuda:0
```

### Radar Ploting
```python
python adaptivity.py <a.pt> <b.pt>
```

It would be better add some args like *--net_type=<nt>*, --hist_design=<hd>,

### Metrics Description
- Command velocity test
    - Fixed lin_vel_x
    - Fixed lin_vel_y
- Friction test
- Add Mass test
<!-- 
# Metrics Upd Log
1.0: 
10*5
```
# labels = [
# ver 1.0
#     "cmd_-1.0",  # 0-4
#     "cmd_0.2",  # 5-9 # 0.2,0.2
#     "cmd_0.4",  # 10-14 # 0.4,0.4
#     "cmd_0.8",  # 15-19 # 0.8,0.8
#     "cmd.1.6",  # 20-24
#     "cmd_2.0",  # 25-29
#     "cmd_3.0",  # 30-34
#     "fri_low",  # 35-39  # 0.1
#     "fri_high",  # 40-44  # 2.0
#     "mas",  # 45-49 # 1.0-1.5
# ]
```
2.0:
25*4
```
labels = [
    # ver 2.0
    "cmd_-1.5",
    "cmd_-1.0",
    "cmd_-0.5",
    "cmd_0.1",
    "cmd_0.5",
    "cmd_1.0",
    "cmd_1.5",
    "cmd_2.0",
    "cmd_2.5",
    "cmd_3.0",
    "cmd_3.5",
    "cmd_4.0",
    "cmd_y_-1.5",
    "cmd_y_0.0",
    "cmd_y_1.5",
    "fri_0.1",
    "fri_1.0",
    "fri_1.5",
    "fri_2.0",
    "fri_5.0",
    "mas_-1.0",
    "mas_0.5",
    "mas_1.0",
    "mas_2.5",
    "mas_5.0",
]
```
2.1:
25*40
```
labels = [
    # ver 2.1
    "cmd_-1.5",
    "cmd_-1.0",
    "cmd_-0.5",
    "cmd_0.1",
    "cmd_0.5",
    "cmd_1.0",
    "cmd_1.5",
    "cmd_2.0",
    "cmd_2.5",
    "cmd_3.0",
    "cmd_3.5",
    "cmd_4.0",
    "cmd_y_-1.5",
    "cmd_y_0.0",
    "cmd_y_1.5",
    "fri_0.1",
    "fri_1.0",
    "fri_2.0",
    "fri_5.0",
    "fri_15.0",
    "mas_-2.5",
    "mas_0.5",
    "mas_5.0",
    "mas_10.0",
    "mas_25.0",
]
``` -->