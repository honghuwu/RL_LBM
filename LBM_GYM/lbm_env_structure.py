import gymnasium as gym
import numpy as np
from typing import Tuple, Dict
from MyCode.LBM import lbm_solver

class LBMEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}

    def __init__(self, config: Dict):
        super().__init__()
        # 初始化LBM求解器
        self.lbm = lbm_solver(
            nx=801, ny=401,
            bc_type=[0,1,1,1],
            bc_value=[[0.1, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            Red=1000,
            air_c=300,
            air_para=[2.0, 4.0, 12.0, 5.0],
            air_o=[200.0, 200.0]
        )
        
        # 定义动作空间，这里调整攻角，后续根据控制策略进行调整
        self.action_space = gym.spaces.Box(low=-15.0, high=15.0, shape=(1,))
        
        # 定义观测空间（速度场幅值）
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.lbm.nx, self.lbm.ny),
            dtype=np.float32
        )
        
        self.reward = 0.0

    def reset(self, seed=None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed = seed)
        self.lbm.init()
        info = {}
        observation = self._get_obs()
        return observation, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        # 执行动作，以更改攻角为例
        self.lbm.air_para[3] = action[0]
        self.lbm.generate_mask()
        reward_list = []
        # 推进LBM模拟
        for _ in range(10):
            self.lbm.collide_and_stream()
            self.lbm.update_macro_var()
            self.lbm.apply_bc()
            reward = self._calculate_reward()
            reward_list.append(reward)

        # 获取观测和奖励
        obs = self._get_obs()
        rewards = np.array(reward_list).mean()
        
        terminated = False
        truncated = False
        info = {}
        return obs, rewards, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        vel = self.lbm.vel.to_numpy()
        return np.linalg.norm(vel, axis=2)

    def _calculate_reward(self) -> float:
        # 奖励函数设计
        return -np.mean(self._get_obs())

    def render(self, mode='human'):
        if mode == 'rgb_array':
            return self.lbm.vel.to_numpy()
        return None

    def close(self):
        pass