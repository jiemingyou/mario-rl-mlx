# mario-rl-mlx

![Mario](mario.gif)

DDQN Super Mario agent written using the MLX framework for Apple Silicon
devices. The code is adapted from the
[official PyTorch tutorial](https://pytorch.org/tutorials/intermediate/mario_rl_tutorial.html).

**Project structure:**

| File             | Description              |
| ---------------- | ------------------------ |
| `train.py`       | Main training script     |
| `agent.py`       | DDQN agent               |
| `model.py`       | CNN model                |
| `environment.py` | Gym environment wrappers |
| `logger.py`      | Logging utilities        |

## Train the agent

The code was written solely for learning purposes and it's not tuned and
optimized for actual training. The agent can be trained on the first level of
Super Mario Bros using the gym-super-mario-bros environment.

To train the agent, run the following command:

```bash
python train.py --epochs 40
```

## Test the agent

To test the agent, run the following command:

```bash
python play.py --model path/to/weights.safetensors
```

## (Poor) Results

The agent was trained for 3800 episodes*, accounting to around 1 million steps.
The entire training process took around 2.5 hours on a 2021 MacBook Pro with M1
Pro and 16GB of RAM.

\* According to the original tutorial, the agent has to be trained around 40000
episodes for to achieve good results.
