export LD_LIBRARY_PATH=/home/zdh232/anaconda3/envs/unitree-rl/lib
export WANDB_API_KEY=95d44e5266d5325cb6a1b4dda1b8d100de903ace
export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890
# numpy==1.20.0

# basic h1 training
python legged_gym/scripts/train.py --task=h1 --headless --num_envs=4096 --max_iterations=2000
# check
python legged_gym/scripts/play.py --task=h1 

# h1_m training
python legged_gym/scripts/train.py --task=h1_m  --num_envs=4096 --max_iterations=20000 --headless --sim_device=cuda:3 --rl_device=cuda:3 --run_name=CVAE_vt
python legged_gym/scripts/train.py --task=h1_m  --num_envs=4096 --max_iterations=2000 --headless --sim_device=cuda:3 --rl_device=cuda:3 --run_name=CVAE

# h1_m check
python legged_gym/scripts/play.py --task=h1_m 


# basic g1 training
python legged_gym/scripts/train.py --task=g1 --headless --num_envs=4096 --max_iterations=2000  --sim_device=cuda:1 --rl_device=cuda:1
# check
python legged_gym/scripts/play.py --task=g1 --num_envs=24

# g1_m training
python legged_gym/scripts/train.py --task=g1_m  --num_envs=4096 --max_iterations=10000 --headless --sim_device=cuda:2 --rl_device=cuda:2 --run_name=CVAE_vt
python legged_gym/scripts/train.py --task=g1_m  --num_envs=4096 --max_iterations=2000 --headless --sim_device=cuda:3 --rl_device=cuda:3 --run_name=CVAE

# g1_m check
python legged_gym/scripts/legged_gym_play.py --task=g1_m --num_envs=24 --record_video

Total number of episodes: 24
legged_gym/scripts/legged_gym_play.py:149: DeprecationWarning: Starting with ImageIO v3 the behavior of this function will switch to that of iio.v3.imread. To keep the current behavior (and make this warning disappear) use `import imageio.v2 as imageio` or call `imageio.v2.imread` directly.
  imgs = [imageio.imread(f) for f in frames]
Traceback (most recent call last):
  File "legged_gym/scripts/legged_gym_play.py", line 161, in <module>
    play(args)
  File "legged_gym/scripts/legged_gym_play.py", line 150, in play
    imageio.mimsave(video_file, imgs, fps=25)
  File "/home/user/miniforge3/envs/unitree-rl/lib/python3.8/site-packages/imageio/v2.py", line 494, in mimwrite
    with imopen(uri, "wI", **imopen_args) as file:
  File "/home/user/miniforge3/envs/unitree-rl/lib/python3.8/site-packages/imageio/core/imopen.py", line 281, in imopen
    raise err_type(err_msg)
ValueError: Could not find a backend to open `/home/user/mly/unitree_rl_gym/logs/g1_m/exported/video/play.mp4`` with iomode `wI`.
Based on the extension, the following plugins might add capable backends:
  FFMPEG:  pip install imageio[ffmpeg]
  pyav:  pip install imageio[pyav]