# mario-rl-mlx

DDQN Super Mario agent written using the MLX framework for Apple Silicon
devices. The code is adapted from the official PyTorch tutorial.

Project structure:

- `train.py`: Main training script
- `agent.py`: DDQN agent
- `model.py`: CNN model
- `environment.py`: Gym environment wrappers
- `logger.py`: Logging utilities

## Train the agent

The code was written solely for learning purposes and it's not tuned and
optimized for actual training. The agent can be trained on the first level of
Super Mario Bros using the gym-super-mario-bros environment.

To train the agent, run the following command:

```bash
python train.py --epochs 1000
```
