apt-get install git wget cmake python3-opencv -y
git clone https://github.com/bengreenberg5/dqn

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
sh Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc
conda create --name rl python=3.9 -y
conda activate rl

cd dqn/dqn
pip install -r ../requirements.txt
python run.py
