import copy
import numpy as np
import mlx.nn as nn
import mlx.core as mx
from random import sample
from collections import deque
import mlx.optimizers as optim

from agent import MarioNet


class Mario:
    def __init__(self, state_dim, action_dim, save_dir):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.save_dir = save_dir

        # DQN agent
        self.net = MarioNet(self.state_dim, self.action_dim)

        # Hyperparameters
        self.exploration_rate = 1
        self.exploration_rate_decay = 0.9999975  # 0.99999975
        self.exploration_rate_min = 0.1
        self.curr_step = 0
        self.save_every = 1e3
        self.gamma = 0.9
        self.burnin = 1e2  # 1e4
        self.learn_every = 3
        self.sync_every = 1e2  # 1e4

        # Memory replay
        self.memory = deque(maxlen=1000)  # 100000
        self.batch_size = 12  # 32

        # Loss and optimizer
        self.optimizer = optim.Adam(learning_rate=0.00025)
        self.loss_and_grad_fn = nn.value_and_grad(self.net, self.loss_fn)

    def act(self, state: mx.array):
        """
        Given a state, choose an epsilon-greedy action and update value of step.

        Inputs:
        state(``mx.array``): A single observation of the current state, dimension is (state_dim)
        Outputs:
        ``action_idx`` (``int``): An integer representing which action Mario will perform
        """

        # EXPLORE
        if np.random.rand() < self.exploration_rate:
            action_idx = np.random.randint(self.action_dim)

        # EXPLOIT
        else:
            state = state[None, :]  # unsqueeze to add batch dimension
            if state.shape[0] == 1:  # if we don't have enough frames to make a stack
                return np.random.randint(self.action_dim)
            action_values = self.net(state, model="online")
            action_idx = mx.argmax(action_values, axis=1).item()

        # decrease exploration_rate
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

        # increment step
        self.curr_step += 1
        return action_idx

    def cache(self, state, next_state, action, reward, done):
        """
        Store the experience to self.memory (replay buffer)

        Inputs:
        state (``mx.array``),
        next_state (``mx.array``),
        action (``int``),
        reward (``float``),
        done(``bool``))
        """

        self.memory.append([state, next_state, action, reward, done])

    def recall(self):
        """
        Retrieve a batch of experiences from memory
        """
        batch = sample(self.memory, self.batch_size)
        state, next_state, action, reward, done = zip(*batch)

        state = mx.array(state)
        next_state = mx.array(next_state)
        action = mx.array(action).reshape(self.batch_size, 1)
        reward = mx.array(reward).reshape(self.batch_size, 1)
        done = mx.array(done).reshape(self.batch_size, 1)

        return state, next_state, action, reward, done

    def td_estimate(self, state, action):
        current_Q = self.net(state, model="online")[
            mx.arange(0, self.batch_size), action
        ]  # Q_online(s,a)
        return current_Q

    def td_target(self, reward, next_state, done):
        next_state_Q = self.net(next_state, model="online")
        best_action = mx.argmax(next_state_Q, axis=1)
        next_Q = self.net(next_state, model="target")[
            mx.arange(0, self.batch_size), best_action
        ]
        return reward + (1 - done) * self.gamma * next_Q

    def loss_fn(self, _, X, y):
        return mx.mean(nn.losses.smooth_l1_loss(X, y))

    def update_Q_online(self, td_estimate, td_target):
        # Get the loss and gradients
        loss, grads = self.loss_and_grad_fn(self.net, td_estimate, td_target)

        # Update the optimizer state and model parameters
        self.optimizer.update(self.net, grads)

        # Force a graph evaluation
        mx.eval(self.net.parameters(), self.optimizer.state)

        return loss.item()

    def sync_Q_target(self):
        self.net.target_conv.update(self.net.online_conv.parameters())
        self.net.target_fc.update(self.net.online_fc.parameters())

    def save(self):
        save_path = (
            self.save_dir
            / f"mario_net_{int(self.curr_step // self.save_every)}.safetensors"
        )
        self.net.save_weights(save_path)
        print(f"Model checkpoint saved to {save_path} at step {self.curr_step}")

    def learn(self):
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()

        if self.curr_step % self.save_every == 0:
            self.save()

        if self.curr_step < self.burnin:
            return None, None

        if self.curr_step % self.learn_every != 0:
            return None, None

        # Sample from memory
        state, next_state, action, reward, done = self.recall()

        # Get TD Estimate
        td_est = self.td_estimate(state, action)

        # Get TD Target
        td_tgt = self.td_target(reward, next_state, done)

        # Backpropagate loss through Q_online
        loss = self.update_Q_online(td_est, td_tgt)

        return (td_est.mean().item(), loss)
