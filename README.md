PyTorch implementation of [DQN](http://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf).



https://user-images.githubusercontent.com/3505187/163687604-ab68cad9-9fd8-4726-abb3-593e471d5600.mp4 https://user-images.githubusercontent.com/3505187/163687606-8bde3b6a-77ef-43c5-90ad-fbc4e1b348e1.mp4



https://user-images.githubusercontent.com/3505187/163687607-3144e3a9-75c1-43b4-89de-01ece01458a4.mp4




https://user-images.githubusercontent.com/3505187/163687610-d2161632-e710-4a40-9777-a503eed2ca50.mp4



## Setup

Run the following commands:

```
apt-get update
apt-get install git wget cmake python3-opencv ffmpeg -y
```

Create a virtual environment and install requirements:

```
cd dqn
pip install -r requirements.txt
```

## Training

Run:

```
cd dqn
python run.py --env BreakoutNoFrameskip-v4 --config rainbow
```

where `env` is a gym environment name, and `config` is the name of a file in the "configs" directory (minus the .gin extension).

## Evaluation

Run:

```
python generate_video.py --env BreakoutNoFrameskip-v4 --run gcp-breakout
```

where `run` is the name of a directory with agent checkpoints.
