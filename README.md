PyTorch implementations of [DQN](http://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf).

## Setup

Run the following commands:

```
apt-get update
apt-get install git wget cmake python3-opencv
```

Create a virtual environment, then:

```
cd dqn
pip install -r requirements.txt
```

## Training

Choose a gym environment and config file, then run:

```
cd dqn
python run.py --env BreakoutNoFrameskip-v4 --config rainbow
```

## Evaluation

Choose a folder containing agent checkpoints, then run:

```
python generate_video.py -- env BreakoutNoFrameskip-v4 --run gcp-breakout
```
