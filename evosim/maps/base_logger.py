from abc import ABC, abstractmethod


class BaseLogger(ABC):

    def __init__(self, project_name):
        self.project_name = project_name

    @abstractmethod
    def log_step(self, episode, obs, *args):
        pass

    @abstractmethod
    def log_episode(self, episode, *args):
        pass
