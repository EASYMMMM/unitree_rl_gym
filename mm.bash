export LD_LIBRARY_PATH=/home/zdh232/anaconda3/envs/unitree-rl/lib
export WANDB_API_KEY=95d44e5266d5325cb6a1b4dda1b8d100de903ace
export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890

# basic h1 training
python legged_gym/scripts/train.py --task=h1 --headless --num_envs=4096 --max_iterations=2000

# check
python legged_gym/scripts/play.py --task=h1 

# h1_m training
python legged_gym/scripts/train.py --task=h1_m  --num_envs=4096 --max_iterations=2000 --headless --run_name=AC

# h1_m check
python legged_gym/scripts/play.py --task=h1_m 