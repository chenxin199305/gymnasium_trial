from typing import Optional
import numpy as np
import gymnasium as gym


class GridWorldEnv(gym.Env):

    def __init__(self, size: int = 5):
        # The size of the square grid
        self.size = size

        # Define the agent and target location; randomly chosen in `reset` and updated in `step`
        self._agent_location = np.array([-1, -1], dtype=np.int32)
        self._target_location = np.array([-1, -1], dtype=np.int32)

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`-1}^2
        self.observation_space = gym.spaces.Dict(
            {
                "agent": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "target": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),
            }
        )

        # We have 4 actions, corresponding to "right", "up", "left", "down"
        self.action_space = gym.spaces.Discrete(4)
        # Dictionary maps the abstract actions to the directions on the grid
        self._action_to_direction = {
            0: np.array([1, 0]),  # right
            1: np.array([0, 1]),  # up
            2: np.array([-1, 0]),  # left
            3: np.array([0, -1]),  # down
        }

    def _get_obs(self):
        """
        Returns the current observation of the environment.

        This method provides a dictionary containing the agent's current location and
        the target's location. The observation is structured to allow the agent to make
        informed decisions based on its position relative to the target.

        :return: A dictionary with two keys: "agent" mapping to the agent's current
                 location, and "target" mapping to the target's location.
        :rtype: dict[str, tuple[int, int]]
        """
        return {
            "agent": self._agent_location,
            "target": self._target_location
        }

    def _get_info(self):
        """
        Retrieve the information about the current state, specifically the distance between
        the agent's location and the target location.

        This method calculates the L1 norm (Manhattan distance) between the agent's current
        position and the target position. The result is returned as a dictionary containing
        the computed distance under the key "distance".

        :return: A dictionary containing the Manhattan distance between the agent's location
                 and the target location.
        :rtype: dict[str, float]
        """
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """
        reset() 的目的是为环境启动一个新剧集，并具有两个参数：seed 和 options。
        seed 可用于将随机数生成器初始化为确定性状态，options 可用于指定 reset 中使用的值。
        在 reset 的第一行，您需要调用 super().reset(seed=seed)，这将初始化随机数生成器（np_random）以在 reset() 的其余部分中使用。
        """
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the agent's location uniformly at random
        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        # We will sample the target's location randomly until it does not coincide with the agent's location
        self._target_location = self._agent_location
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = self.np_random.integers(
                0, self.size, size=2, dtype=int
            )

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        """
        Perform a single step in the environment based on the given action.

        This method maps the provided action to a direction of movement, updates the agent's
        location while ensuring it remains within the grid bounds, and determines whether
        the episode has terminated or been truncated. It also computes the reward and gathers
        the observation and additional information about the current state of the environment.

        :param action: int
            An integer representing the action to be taken. The action must be an element
            of {0, 1, 2, 3}, which corresponds to specific directions of movement.

        :return: tuple
            A tuple containing the following elements:
            - observation: The current observation of the environment after taking the action.
            - reward: float
              The reward received after performing the action. It is 1 if the episode
              terminates successfully, otherwise 0.
            - terminated: bool
              A boolean indicating whether the episode has terminated. This happens when
              the agent reaches the target location.
            - truncated: bool
              A boolean indicating whether the episode has been truncated. This is always
              False in this implementation.
            - info: dict
              A dictionary containing additional information about the environment's state.

        """
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction[action]
        # We use `np.clip` to make sure we don't leave the grid bounds
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )

        # An environment is completed if and only if the agent has reached the target
        terminated = np.array_equal(self._agent_location, self._target_location)
        truncated = False
        reward = 1 if terminated else 0  # the agent is only reached at the end of the episode
        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info


"""
环境注册后，您可以通过 gymnasium.pprint_registry() 进行检查，这将输出所有已注册的环境，
然后可以使用 gymnasium.make() 初始化该环境。
可以使用 gymnasium.make_vec() 实例化环境的向量化版本，该版本具有并行运行的同一环境的多个实例。
"""
gym.register(
    id="gymnasium_env/GridWorld-v0",
    entry_point=GridWorldEnv,
)

gym.pprint_registry()
