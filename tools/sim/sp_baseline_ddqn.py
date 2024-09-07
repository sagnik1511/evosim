from configparser import ConfigParser
from typing import Any, Dict

from evosim.elements.agents import L1Agent
from evosim.maps.sp_1 import SinglePlayerMap, SinglePlayerMapLogger
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
        "sim_episodes": int(job["SIM_EPISODES"]),
        "sim_fps": int(job["SIM_FPS"]),
        "learning_rate": float(job["LR"]),
        "gamma": float(job["GAMMA"]),
        "eps": float(job["EPS"]),
        "eps_min": float(job["EPS_MIN"]),
        "eps_decay": float(job["EPS_DECAY"]),
        "buffer_size": int(job["BUFFER_SIZE"]),
        "batch_size": int(job["BATCH_SIZE"]),
        "agent_learn_counter": int(job["AGENT_LEARN_COUNTER"]),
        "agent_base_hp": int(job["AGENT_HP"]),
        "agent_run_delta": float(job["AGENT_RUN_DELTA"]),
    }

    return train_config


def simulate(trained_agent_path):
    """Run simuation over saved Agent"""

    # Update your job name here
    job_name = "DDQN_Baseline"

    config = load_config(job_name)
    logger.info(config)

    # Loading stored agent
    agent = L1Agent.load(trained_agent_path)

    env = SinglePlayerMap(
        agent,
        side_length=config["side_length"],
        obstacle_percent=config["obstacle_percentage"],
        resource_percent=config["resource_percentage"],
        waste_move_penalty=config["waste_move_penalty"],
        death_penalty=config["death_penalty"],
        finish_reward=config["finish_reward"],
    )

    wnb_logger = SinglePlayerMapLogger(
        "evosim-sp-sim", config=config, sim_fps=config["sim_fps"]
    )

    # Running sim
    for episode in range(config["sim_episodes"]):
        logger.info("Simulating on episode -> %s", episode + 1)

        # Initialize total reward
        total_reward = 0
        total_setps_traversed = 0
        obs, terminated, truncated = env.reset()

        # Interact and through the env until it is terminated
        while not terminated and not truncated:
            total_setps_traversed += 1

            # Act afte observing current state
            action = agent.act(obs)

            # fetch next state after performing action
            obs, reward, terminated, truncated = env.step(action)
            wnb_logger.log_step(
                episode, total_setps_traversed, obs, agent.hp, total_reward
            )

            # Accumulating episode reward
            total_reward += reward

        logger.info(
            "Episode[%s] Total Steps -> %s Total Reward -> %s",
            episode + 1,
            total_setps_traversed,
            total_reward,
        )
        wnb_logger.log_episode(episode, total_setps_traversed, total_reward)

    wnb_logger.finish()


if __name__ == "__main__":
    simulate("/Users/tensored/evosim/agents/L1Agent.pkl")
