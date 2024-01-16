# AI dog vanilla
A simple model to let the actor(AI dog) well interact with breeder, currently have three type of model
 - DogNetworkLinear: simple model of gemm
 - DogNetworkLinearRecall: model of gemm with recall chunk
 - DogNetworkTransformerRecall: model of transformer with recall chunk

and currently have two type of reward
 - simple: use the distance as reward
 - HP_MP: use the "vital sign" as reward

# Quick start
```
# Install MuJoCo: https://github.com/openai/mujoco-py#install-mujoco
# mac
wget https://mujoco.org/download/mujoco210-macos-x86_64.tar.gz mujoco210.tar.gz
# linux
wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz mujoco210.tar.gz
tar zxvf mujoco210.tar.gz
mkdir ~/.mujoco
mv mujoco210 ~/.mujoco/
# mac
brew install gcc@9

conda create --name env_mj python=3.8
conda activate env_mj
git clone https://github.com/ches51311/mujoco-worldgen.git
cd mujoco-worldgen
pip install wheel
pip install -r requirements.txt
pip install -e .
cd -
pip install -r requirements.txt
# other option please refer to cmd.sh
time python3 main.py --net_type=transformer_recall --reward_type=simple --times=5000
```
# Status
The DogNetworkTransformerRecall with simple reward can success, but with HP_MP reward failed. 