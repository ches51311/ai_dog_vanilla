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
pip install -e .
./bin/examine.py examples/ai_dog.py
```