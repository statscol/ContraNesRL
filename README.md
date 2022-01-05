# ContraNesRL
### Style Transfer Test

Trying to play classic nes games using Reinforcement Learning

![test output](https://github.com/statscol/ContraNesRL/blob/main/demo.png?raw=true)

PPO model using stable_baselines3 api and Gym-contra to train an agent to play classic Contra game on Nes

Use the train module to train the agent (Cuda enabled will speed training), -lr to define learning rate, -t for time steps to train for


```console
pip install -r requirements.txt
python train.py -lr 0.000001 -t 100000
python play.py
```

Or to test only use play

```console
python play.py
```

