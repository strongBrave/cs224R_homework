export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidiaexport MUJOCO_PY_MUJOCO_PATH=/home/yujunhao/data/yujunhao/jushen/cs224r/.mujoco/mujoco210
python train.py agent.num_critics=2 utd=1