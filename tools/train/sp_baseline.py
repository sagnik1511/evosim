from evosim.elements.agents import L1Agent
from evosim.maps.sp_1 import SinglePlayerMap, SinglePlayerMapLogger
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
TRAIN_EPISODES = 1

# Agent Constant
LR = 1e-4
GAMMA = 0.05
EPS_CLIP = 0.2
EPOCH_K = 100
AGENT_HP = 2
AGENT_RUN_DELTA = 0.01


def train():
    """Train over Simulation with Given Policy"""
    train_config = {
        "side_length": ENV_N,
        "env_channels": CHANNELS_N,
        "num_actions": ACT_N,
        "obstacle_percentage": OBSTACLE_PCT,
        "resource_percentage": RESOURCE_PCT,
        "waste_move_penalty": WASTE_MOVE_PENALTY,
        "death_penalty": DEATH_PENALTY,
        "finish_reward": FINISH_REWARD,
        "train_episodes": TRAIN_EPISODES,
        "learning_rate": LR,
        "gamma": GAMMA,
        "eps_clip": EPS_CLIP,
        "epoch_k": EPOCH_K,
        "agent_base_hp": AGENT_HP,
        "agent_run_delta": AGENT_RUN_DELTA,
    }

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
    wnb_logger = SinglePlayerMapLogger("evosim-sp-train", config=train_config)

    # Train over Episodes
    for episode in range(TRAIN_EPISODES):
        logger.info(f"Training on episode -> {episode+1}")

        # Initialize total reward
        total_reward = 0
        total_setps_traversed = 0
        obs, terminated, truncated = env.reset()

        # Interact and through the env until it is terminated
        while not terminated and not truncated:
            total_setps_traversed += 1

            # Act afte observing current state
            action, log_probs = agent.act(obs)

            # fetch next state after performing action
            obs, reward, terminated, truncated = env.step(action)
            wnb_logger.log_step(
                episode, total_setps_traversed, obs, agent.hp, total_reward
            )

            # Accumulating episode reward
            total_reward += reward

            # Learn over the steps
            agent.observe(obs, action, log_probs, reward)

        logger.info(
            f"Episode[{episode+1}] Total Steps -> {total_setps_traversed} Total Reward -> {total_reward}"
        )
        wnb_logger.log_episode(episode, total_setps_traversed, total_reward)

    # Save the updated agent
    agent.save()
    wnb_logger.finish()


if __name__ == "__main__":
    train()
