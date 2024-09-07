"""PPO Training Job"""

from configparser import ConfigParser
from typing import Any, Dict

from evosim.elements.agents import L1Agent
from evosim.maps.sp_1 import SinglePlayerMap, SinglePlayerMapLogger
from evosim.policy import PPO
from evosim.utils.logger import get_logger

logger = get_logger()


def load_config(job_name: str) -> Dict[str, Any]:
    """Load configuration for training

    Args:
        job_name (str): Name of the Job

    Returns:
        Dict[str, Any]: Job Config
    """

    parser = ConfigParser()
    parser.read("config.ini")

    job = parser[job_name]

    train_config = {
        "policy_name": job["POLICY"],
        "side_length": int(job["ENV_N"]),
        "env_channels": int(job["CHANNELS_N"]),
        "num_actions": int(job["ACT_N"]),
        "obstacle_percentage": float(job["OBSTACLE_PCT"]),
        "resource_percentage": float(job["RESOURCE_PCT"]),
        "waste_move_penalty": float(job["WASTE_MOVE_PENALTY"]),
        "death_penalty": float(job["DEATH_PENALTY"]),
        "finish_reward": float(job["FINISH_REWARD"]),
        "train_episodes": int(job["TRAIN_EPISODES"]),
        "learning_rate": float(job["LR"]),
        "gamma": float(job["GAMMA"]),
        "eps_clip": float(job["EPS_CLIP"]),
        "epoch_k": int(job["EPOCH_K"]),
        "agent_base_hp": int(job["AGENT_HP"]),
        "agent_run_delta": float(job["AGENT_RUN_DELTA"]),
    }

    return train_config


def train():
    """Train over Simulation with PPO Policy"""

    # Update your job name here
    job_name = "PPO_Baseline"

    config = load_config(job_name)
    logger.info(config)

    # Defining objects
    policy = PPO(
        env_side_length=config["side_length"],
        env_channels=config["env_channels"],
        env_actions=config["num_actions"],
        lr=config["learning_rate"],
        gamma=config["gamma"],
        eps_clip=config["eps_clip"],
        k_epochs=config["epoch_k"],
    )
    agent = L1Agent(
        policy, hp=config["agent_base_hp"], run_delta=config["agent_run_delta"]
    )
    env = SinglePlayerMap(
        agent,
        side_length=config["side_length"],
        obstacle_percent=config["obstacle_percentage"],
        resource_percent=config["resource_percentage"],
        waste_move_penalty=config["waste_move_penalty"],
        death_penalty=config["death_penalty"],
        finish_reward=config["finish_reward"],
    )
    wnb_logger = SinglePlayerMapLogger("evosim-sp-train", config=config)

    # Train over Episodes
    for episode in range(config["train_episodes"]):
        logger.info("Training on episode -> %s", episode + 1)

        # Initialize total reward
        total_reward = 0
        total_setps_traversed = 0
        obs, terminated, truncated = env.reset()

        # Interact and through the env until it is terminated
        while not terminated and not truncated:
            total_setps_traversed += 1

            # Act afte observing current state
            action, log_probs, value = agent.act(obs)

            # fetch next state after performing action
            obs, reward, terminated, truncated = env.step(action)
            wnb_logger.log_step(
                episode, total_setps_traversed, obs, agent.hp, total_reward
            )

            # Accumulating episode reward
            total_reward += reward

            # Learn over the steps
            agent.observe(obs, action, reward, log_probs=log_probs, value=value)

        logger.info(
            "Episode[%s] Total Steps -> %s Total Reward -> %s",
            episode + 1,
            total_setps_traversed,
            total_reward,
        )
        wnb_logger.log_episode(episode, total_setps_traversed, total_reward)

    # Save the updated agent
    agent.save()
    wnb_logger.finish()


if __name__ == "__main__":
    train()
