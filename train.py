import argparse
import datetime
from pathlib import Path
import gym_super_mario_bros
from gym.wrappers import FrameStack
from nes_py.wrappers import JoypadSpace

from agent import Mario
from logger import MetricLogger
from environment import (
    SkipFrame,
    GrayScaleObservation,
    ResizeObservation,
    ObservationToMLX,
)


def main():
    parser = argparse.ArgumentParser(description="Train a Super Mario playing agent.")
    parser.add_argument(
        "--epochs", type=int, default=40, help="Number of episodes to train."
    )
    args = parser.parse_args()

    env = gym_super_mario_bros.make(
        "SuperMarioBros-1-1-v3",
        render_mode="rgb",
        apply_api_compatibility=True,
    )
    env = JoypadSpace(env, [["right"], ["right", "A"]])
    env = SkipFrame(env, skip=4)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=84)
    env = FrameStack(env, num_stack=4)
    env = ObservationToMLX(env)

    save_dir = Path("checkpoints") / datetime.datetime.now().strftime(
        "%Y-%m-%dT%H-%M-%S"
    )
    save_dir.mkdir(parents=True)

    mario = Mario(
        state_dim=(4, 84, 84),
        action_dim=env.action_space.n,
        save_dir=save_dir,
    )

    logger = MetricLogger(save_dir)
    episodes = args.epochs

    for e in range(episodes):

        # Reset environment
        state, _ = env.reset()

        # Play the game!
        while True:

            # Run agent on the state
            action = mario.act(state)

            # Agent performs action
            next_state, reward, done, _, info = env.step(action)

            # Store the transition in memory
            mario.cache(state, next_state, action, reward, done)

            # Learn
            q, loss = mario.learn()

            # Logging
            logger.log_step(reward, loss, q)

            # Update state
            state = next_state

            # Check if end of game
            if done or info["flag_get"]:
                break

        logger.log_episode()

        if (e % 20 == 0) or (e == episodes - 1):
            logger.record(
                episode=e,
                epsilon=mario.exploration_rate,
                step=mario.curr_step,
            )


if __name__ == "__main__":
    main()
