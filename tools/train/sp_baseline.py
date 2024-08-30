from evosim.elements.agents import L1Agent
from evosim.maps.sp_1 import SinglePlayerMap
from evosim.policy import PPO
from evosim.utils.logger import get_logger

logger = get_logger()


def simulate(trained_agent):
    pass


def train():
    policy = PPO()
    agent = L1Agent(policy)
    env = SinglePlayerMap(agent)

    num_episodes = 100
    for episode in range(num_episodes):
        logger.info(f"Training on episode -> {episode+1}")

        obs, terminated, truncated = env.reset()

        while not terminated and not truncated:
            action, log_probs = agent.act(obs)
            obs, reward, terminated, truncated = env.step(action)
            agent.observe(obs, action, log_probs, reward)
    agent.save()


if __name__ == "__main__":
    train()
