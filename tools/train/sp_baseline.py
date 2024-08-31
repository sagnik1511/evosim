from evosim.elements.agents import L1Agent
from evosim.maps.sp_1 import SinglePlayerMap
from evosim.policy import PPO
from evosim.utils.logger import get_logger

logger = get_logger()

# Constants
# Environment Constants
ENV_N = 32
CHANNELS_N = 3
ACT_N = 4
OBSTACLE_PCT = 0.1
RESOURCE_PCT = 0.2
WASTE_MOVE_PENALTY = 0.01
DEATH_PENALTY = 1
FINISH_REWARD = 2
NUM_EPISODES = 100

# Agent Constant
LR = 1e-4
GAMMA = 0.05
EPS_CLIP = 0.2
EPOCH_K = 100
AGENT_HP = 1
AGENT_RUN_DELTA = 0.01


def simulate(trained_agent_path):
    pass


def train():

    # Defining objects
    policy = PPO(
        env_side_length=ENV_N,
        env_channels=CHANNELS_N,
        env_actions=ACT_N,
        lr=LR,
        gamma=GAMMA,
        eps_clip=EPS_CLIP,
        k_epochs=EPOCH_K,
    )
    agent = L1Agent(policy, hp=AGENT_HP, run_delta=AGENT_RUN_DELTA)
    env = SinglePlayerMap(
        agent,
        side_length=ENV_N,
        obstacle_percent=OBSTACLE_PCT,
        resource_percent=RESOURCE_PCT,
        waste_move_penalty=WASTE_MOVE_PENALTY,
        death_penalty=DEATH_PENALTY,
        finish_reward=FINISH_REWARD,
    )

    for episode in range(NUM_EPISODES):
        logger.info(f"Training on episode -> {episode+1}")

        # Initialize total reward
        total_reward = 0
        obs, terminated, truncated = env.reset()

        # Interact and through the env until it is terminated
        while not terminated and not truncated:

            # Act afte observing current state
            action, log_probs = agent.act(obs)

            # fetch next state after performing action
            obs, reward, terminated, truncated = env.step(action)

            # Accumulating episode reward
            total_reward += reward

            # Learn over the steps
            agent.observe(obs, action, log_probs, reward)

        logger.info(f"Episode:{episode+1} total reward -> {total_reward}")

    # Save the updated agent
    agent.save()


if __name__ == "__main__":
    train()
