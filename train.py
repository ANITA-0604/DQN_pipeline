from data import DataLoader
from env import TradingEnv
from agent import DQN_agent
import torch
import numpy as np
import matplotlib.pyplot as plt

def train() -> None:
    dataloader = DataLoader("AAPL", begin_date='2023-05-01', end_date='2024-12-31')
    df = dataloader.load_data()
    env = TradingEnv(df)
    agent = DQN_agent(state_size=env.window_size * 5, action_size=3)
    # action : 0: Hold, 1: Buy, 2: Sell

    episode = 50
    rewards = []  # Track total rewards per episode
    epsilons = []  # Track epsilon values per episode

    for ep in range(episode):
        state = env.reset()
        total_reward = 0

        while True:
            action = agent.act(state)
            next_state, reward, done = env.step(action)

            agent.remember(state, action, reward, next_state, done)
            agent.train()
            state = next_state
            total_reward += reward
            if done:
                break

        # Decay epsilon after each episode
        agent.epsilon = max(0.1, agent.epsilon * 0.995)

        # Log metrics
        rewards.append(total_reward)
        epsilons.append(agent.epsilon)
        print(f"Episode {ep + 1}, Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.4f}")

    # Save model after training
    torch.save(agent.model.state_dict(), "dqn_trading_model.pt")

    # Plot training metrics
    plt.figure(figsize=(12, 6))

    # Plot total rewards
    plt.subplot(2, 1, 1)
    plt.plot(rewards, label="Total Reward")
    plt.title("Total Rewards Over Episodes")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid()
    plt.legend()

    # Plot epsilon decay
    plt.subplot(2, 1, 2)
    plt.plot(epsilons, label="Epsilon", color="orange")
    plt.title("Epsilon Decay Over Episodes")
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")
    plt.grid()
    plt.legend()

    plt.tight_layout()
    plt.show()


def main():
    train()

if __name__ == "__main__":
    main()

