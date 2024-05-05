import gym_super_mario_bros
from gym.wrappers import FrameStack
from nes_py.wrappers import JoypadSpace

import time
import argparse
import numpy as np
from matplotlib import pyplot as plt

from agent import Mario
from environment import (
    SkipFrame,
    GrayScaleObservation,
    ResizeObservation,
    ObservationToMLX,
)


def run_agent(model_weight_path: str = None, savefig: bool = False) -> None:
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Path to model weights")
    args = parser.parse_args()

    # Get the model weight path from command line arguments
    model_weight_path = args.model

    # Create the environment
    env = gym_super_mario_bros.make(
        "SuperMarioBros-1-1-v0",
        render_mode="human",
        apply_api_compatibility=True,
    )
    env = JoypadSpace(env, [["right"], ["right", "A"]])
    env = SkipFrame(env, skip=4)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=84)
    env = FrameStack(env, num_stack=4)
    env = ObservationToMLX(env)

    # Create the agent
    agent = Mario(
        state_dim=(4, 84, 84),
        action_dim=env.action_space.n,
        save_dir="",
    )

    # Load the trained model weights
    if model_weight_path:
        agent.net.load_weights(model_weight_path)
        print(f"Loaded model at {model_weight_path}")

    # Run the agent
    for ep in range(5):
        observations = []
        done = False
        state, _ = env.reset()
        while not done:
            if savefig:
                observations.append(np.array(state[0]))
            action = agent.act(state)
            state, reward, done, _, info = env.step(action)
            time.sleep(0.05)

        if savefig:
            # Save the observations as images
            for idx, obs in enumerate(observations):
                plt.matshow(obs, cmap="gray")
                plt.axis("off")
                plt.savefig(f"frames/ep{ep}_obs{idx}.jpg")
                plt.close()


if __name__ == "__main__":
    run_agent()
