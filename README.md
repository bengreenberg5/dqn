PyTorch implementation of [DQN](http://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf).



<table><tr>
<td> <video src="videos/breakout.mp4" type=video/mp4 style="width: 250px;"/> </td>
</tr></table>



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
