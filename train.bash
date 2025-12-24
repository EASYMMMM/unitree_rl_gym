# mly
export LD_LIBRARY_PATH=/home/zdh232/anaconda3/envs/unitree-rl/lib
export WANDB_API_KEY=95d44e5266d5325cb6a1b4dda1b8d100de903ace
export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890

# numpy==1.20.0

# basic h1 training
python legged_gym/scripts/train.py --task=h1 --headless --num_envs=4096 --max_iterations=2000
# check
python legged_gym/scripts/play.py --task=h1 

# CVAE h1_m training
python legged_gym/scripts/train.py --task=h1_m  --num_envs=4096 --max_iterations=20000 --headless --sim_device=cuda:3 --rl_device=cuda:3 --run_name=CVAE_vt
python legged_gym/scripts/train.py --task=h1_m  --num_envs=4096 --max_iterations=2000 --headless --sim_device=cuda:3 --rl_device=cuda:3 --run_name=CVAE
# h1_m check
python legged_gym/scripts/play.py --task=h1_m 


# basic g1 training
python legged_gym/scripts/train.py --task=g1 --headless --num_envs=4096 --max_iterations=2000  --sim_device=cuda:1 --rl_device=cuda:1
# check
python legged_gym/scripts/play.py --task=g1 --num_envs=24

# CVAE g1_m training
python legged_gym/scripts/train.py --task=g1_m  --num_envs=4096 --max_iterations=10000 --headless --sim_device=cuda:2 --rl_device=cuda:2 --run_name=CVAE_vt
python legged_gym/scripts/train.py --task=g1_m  --num_envs=4096 --max_iterations=2000 --headless --sim_device=cuda:3 --rl_device=cuda:3 --run_name=CVAE
python legged_gym/scripts/train.py --task=g1_m  --num_envs=4096 --max_iterations=2000 --headless  --run_name=CVAE_dynamic_random
# g1_m check
python legged_gym/scripts/legged_gym_play.py --task=g1_m --num_envs=12  --load_run=Dec17_11-11-39_CVAE_actor_single_obs --record_video

# DPCVAE g1_m training
python legged_gym/scripts/train.py --task=g1_m  --num_envs=4096 --max_iterations=2500 --headless  --run_name=CVAE_dpcvae
# DPCVAE g1_m check
python legged_gym/scripts/legged_gym_play.py --task=g1_m --num_envs=12  --load_run=Dec24_14-01-54_CVAE_dpcvae --record_video
# DPCVAE g1_m 结果可视化
python legged_gym/scripts/play_cvae_vis.py --task=g1_m --num_envs=100 --load_run=Dec24_14-01-54_CVAE_dpcvae

# Continual RL中 采集真机数据
# Collect Real World Data
python legged_gym/scripts/legged_gym_play.py --task=g1_m --num_envs=50 --collect_data  --collect_max_steps 20000 --headless
# Continual Learning Stage1
python legged_gym/scripts/train_offline_stage1.py --task=g1_m --resume --checkpoint=-1  --load_run=Sep29_14-53-15_CVAE_vt --headless