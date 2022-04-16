PyTorch implementation of [DQN](http://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf).

https://user-images.githubusercontent.com/3505187/163688241-8d951b23-fb82-4771-b57d-b9662565661d.mp4 


https://user-images.githubusercontent.com/3505187/163688243-0232504a-2ac6-45e1-a1d7-bfe9ca35d305.mp4


https://user-images.githubusercontent.com/3505187/163688276-f14b710b-a81b-4e11-a053-3fb31fc0b619.mp4


https://user-images.githubusercontent.com/3505187/163688429-e6476bd5-6f71-4a24-adbd-a67e7c43dded.mp4


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
