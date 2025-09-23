import numpy as np
import gymnasium as gym
import torch
from torch import nn
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from tqdm import tqdm

gamma = 0.99
gae = 0.95
num_epochs = 100
horizon = 1024
updates_per_iteration = 10
covariance = 0.5
epsilon = 0.1

class Actor(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ELU(),
            nn.Linear(32, 32),
            nn.ELU(),
            nn.Linear(32, output_size)
        )

    def forward(self, x):

        return self.net(x)


class Critic(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ELU(),
            nn.Linear(32, 32),
            nn.ELU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):

        return self.net(x)

env = gym.make("Hopper-v5")
state_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]

actor = Actor(state_dim, act_dim)
critic = Critic(state_dim)

covariance_matrix = torch.eye(act_dim)

def compute_target(rewards, values):
    target = 0
    T = len(rewards)
    for t in range(T - 1):
        delta = rewards[t] + gamma*values[t+1] - values[t]
        target += ((gamma*gae)**t)*delta

    target += values[0]

    return target


def rollout(env, actor, critic, timesteps):
    rewards = []
    V_targs = []
    values = []
    actions = []
    logprobs = []
    all_obs = []

    obs = env.reset()[0]
    for i in range(timesteps):
        obs = torch.tensor(obs).float()
        all_obs.append(obs)
        value = critic(obs)
        action_mean = actor(obs)
        dist = MultivariateNormal(action_mean, covariance_matrix)
        action = dist.sample()
        logprob = dist.log_prob(action)
        obs, reward, done , _ , _ = env.step(action.numpy())

        values.append(value)
        actions.append(action)
        logprobs.append(logprob)
        rewards.append(reward)

    rewards = torch.tensor(rewards)
    values = torch.cat(values)
    actions = torch.stack(actions, dim=0)
    all_obs = torch.stack(all_obs, dim=0)
    logprobs = torch.tensor(logprobs)


    for i in range(timesteps):
        v_targ = compute_target(rewards[i:], values[i:])
        V_targs.append(v_targ)

    V_targs = torch.tensor(V_targs)

    return rewards, V_targs, values, actions, logprobs, all_obs

def train(num_epochs, updates_per_iteration, horizon):

    optimizer_actor = Adam(actor.parameters())
    optimizer_critic = Adam(critic.parameters())
    writer = SummaryWriter("ppo")

    for n in tqdm(range(num_epochs)):
        r, V_targ, V, actions, logprobs, all_obs = rollout(env,actor,critic, timesteps=horizon)
        total_reward = r.sum().item()

        avg_actor_loss = 0
        avg_critic_loss = 0

        for i in range(updates_per_iteration):

            #Compute necessary terms
            advantage = V_targ - V.detach()
            #advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
            new_values = critic(all_obs)

            new_action_means = actor(all_obs)
            dist = MultivariateNormal(new_action_means, covariance_matrix)
            new_logprobs = dist.log_prob(actions)
            ratio = torch.exp(new_logprobs - logprobs)

            #LOSSES
            critic_loss = torch.mean((new_values - V_targ)**2)
            term1 = ratio*advantage
            term2 = torch.clip(ratio, 1-epsilon, 1+epsilon)*advantage
            actor_loss = -torch.min(term1,term2).mean()

            #BACKPROP

            optimizer_critic.zero_grad()
            critic_loss.backward(retain_graph = True)
            optimizer_critic.step()

            optimizer_actor.zero_grad()
            actor_loss.backward()
            optimizer_actor.step()

            avg_actor_loss += actor_loss.item()
            avg_critic_loss += critic_loss.item()

        writer.add_scalar("Total reward", total_reward, n)
        writer.add_scalar("Avg critic loss", avg_critic_loss/updates_per_iteration, n)
        writer.add_scalar("AVG actor loss", avg_actor_loss/updates_per_iteration, n)
    
    writer.close()

train(num_epochs, updates_per_iteration=updates_per_iteration, horizon=horizon)