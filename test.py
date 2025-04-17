from data import DataLoader
import torch
import matplotlib.pyplot as plt
from env import TradingEnv
from agent import DQN_agent
import numpy as np

def test(env : TradingEnv, agent: DQN_agent, episode: int= 1) -> None:

    agent.epsilon = 0.0
    agent.model.eval()

    for ep in range(episode):
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = agent.act(state)
            state, reward, done = env.step(action)
            total_reward += reward
        print(f"[Test] Episode {ep+1}: Total Reward = {total_reward:.2f}")
    
    evaluate(env)


def main():
    dataloader = DataLoader("AAPL", begin_date='2025-01-01', end_date='2025-03-31')
    df = dataloader.load_data()
    print("hi",df)
    env = TradingEnv(df)
    agent = DQN_agent(state_size=env.window_size * 5, action_size=3)


    # modify : pass model path and episode as args
    model_path = "dqn_trading_model.pt"
    # test_episode depends on demand
    # One-time test on a fixed period is recommended to use 1 episode
    # if we need to do Stochastic policy or Monte carlo testing, it'll be recommended to use higher episode

    episode = 5

    
    print(f"📦 Loading model from {model_path}")
    agent.model.load_state_dict(torch.load(model_path))
    test(env, agent, episode)

def evaluate(env: TradingEnv):
    history = env.get_history()
    returns = np.diff(history) / history[:-1]  # daily returns
    total_return = (history[-1] - history[0]) / history[0]

    sharpe = returns.mean() / (returns.std() + 1e-8) * np.sqrt(252)  # assume 252 trading days
    drawdown = np.max(np.maximum.accumulate(history) - history)

    print(f"📈 Final Net Worth: {history[-1]:.2f}")
    print(f"💵 Total Return: {total_return * 100:.2f}%")
    print(f"📉 Max Drawdown: {drawdown:.2f}")
    print(f"⚖️ Sharpe Ratio: {sharpe:.2f}")

    plt.plot(history)
    plt.title("Net Worth Over Time")
    plt.xlabel("Step")
    plt.ylabel("Net Worth ($)")
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()

    