import gymnasium as gym
import numpy as np
from gymnasium import spaces
import taichi as ti
import time
import sys
import os

from lbm_with_column import lbm_solver
from tools.show_tools.visualizer import LBMVisualizer

# 控制参数字典
control_params = {
    # LBM 求解器参数
    "nx": 400,
    "ny": 200,
    "Red": 1000,
    "inlet_velocity": 0.1,
    "air_c": 100,
    "air_para": [0, 0, 12.0, -20.0],
    "air_o": [100.0, 100.0],
    
    # 环境控制参数
    "control_range": 10.0,  # 控制范围 (-10.0, 10.0)
    "control_step": 10,     # 每次控制后执行的 LBM 步数
    "reset_steps": 200,     # 重置时执行的 LBM 步数
    "render_init_steps": 50, # 首次渲染前执行的 LBM 步数
    "max_episode_steps": 200, # 最大步数
    
    # 渲染参数
    "render_interval": 10,   # 渲染间隔
    
    # 奖励计算参数
    "reward_clip": 100.0     # 奖励裁剪范围 (-100.0, 100.0)
}


class LBMEnv(gym.Env):

    metadata = {"render_modes": ["human", None], "render_fps": 30}

    def __init__(self, config=None):
        super(LBMEnv, self).__init__()
        if config is None:
            config = {}

        # 渲染开关（config 中传 render_mode="human" 启用渲染）
        self.render_mode = config.get("render_mode", None)
        self.render_interval = config.get("render_interval", control_params["render_interval"])  # 每隔多少 step 渲染一次
        
        # Episode 长度限制
        self.max_episode_steps = config.get("max_episode_steps", control_params["max_episode_steps"])

        # 初始化 LBM 求解器
        self.lbm = lbm_solver(
            nx=control_params["nx"],
            ny=control_params["ny"],
            Red=control_params["Red"],
            inlet_velocity=control_params["inlet_velocity"],
            air_c=control_params["air_c"],
            air_para=control_params["air_para"],
            air_o=control_params["air_o"]
        )

        # 初始化可视化器与 GUI（延迟到第一次渲染时创建）
        self.visualizer = LBMVisualizer(self.lbm.nx, self.lbm.ny)
        self.gui = None

        # 动作空间
        self.action_space = spaces.Box(
            low=np.array([-control_params["control_range"]], dtype=np.int8),
            high=np.array([control_params["control_range"]], dtype=np.int8),
            dtype=np.int8
        )


        # 观测空间
        self.lbm.init()
        self.observation_space = spaces.Box(
                    low=-np.inf, high=np.inf, shape=(642, 5), dtype=np.float32
                )

        # 计步器
        self.step_counter = 0
        self.reward = 0.0
        # 初始化标志
        self.has_been_initialized = False

    # def reset(self, seed=None, options=None):
    #     super().reset(seed=seed)
    #     self.lbm.reset()

    #     # 这可以避免在刚初始化后立即调用output()时的CUDA内存访问问题
    #     for _ in range(5):
    #         self.lbm.step()
        
    #     self.step_counter = 0
    #     obs = self.lbm.output().astype(np.float32)
    #     info = {}
    #     return obs, info


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.lbm.reset()
        for _ in range(control_params["reset_steps"]):
            self.lbm.step()
        self.step_counter = 0
        self.has_been_initialized = True
        obs = self.lbm.output().astype(np.float32)
        info = {}
        return obs, info


    def step(self, action):

        control_val = float(np.clip(action[0], -control_params["control_range"], control_params["control_range"]))
        self.lbm.control(control_val)

        for _ in range(control_params["control_step"]):
            self.lbm.step()

        obs = self.lbm.output().astype(np.float32)
        
        drag_lift = self.lbm.get_reward()

        cd = drag_lift[2]
        cl = drag_lift[3]
        if abs(cd) < 1e-8:  # cd接近0时
            reward = 0.0  # 给予中性奖励
        else:
            reward = cl / cd
            # 检查结果是否为有效数值
            if not np.isfinite(reward):
                reward = 0.0
        # 限制奖励范围，避免极值
        reward = np.clip(reward, -control_params["reward_clip"], control_params["reward_clip"])
        self.step_counter += 1
        # 自动渲染
        if self.render_mode == "human" and self.step_counter % self.render_interval == 0:
            self.render()   
        # 检查episode是否应该结束
        terminated = False
        truncated = self.step_counter >= self.max_episode_steps  # 达到最大步数时截断
        info = {
            "Drag": drag_lift[0],
            "Lift": drag_lift[1],
            "CD": cd,
            "CL": cl
        }

        return obs, reward, terminated, truncated, info

    def render(self, mode='human'):
        if self.gui is None:
            self.gui = ti.GUI("LBMEnv", res=(self.lbm.nx, self.visualizer.combined_height))

        if self.step_counter == 0:
            for _ in range(control_params["render_init_steps"]):
                self.lbm.step()

        # 更新可视化数据
        self.visualizer.update_visualization(self.lbm.vel)
        self.gui.set_image(self.visualizer.get_combined_image())

        # 绘制边界和控制器
        self.visualizer.draw_boundary_points(self.gui, self.lbm.boundary_pos)
        if hasattr(self.lbm, 'controller_pos'):
            self.visualizer.draw_controller_points(self.gui, self.lbm.controller_pos)

        # 绘制文字信息
        drag_lift_vec = self.lbm.calculate_drag_lift()
        drag_lift = np.array([drag_lift_vec[0], drag_lift_vec[1], drag_lift_vec[2], drag_lift_vec[3]])
        #drag = drag_lift[0]
        #lift = drag_lift[1]
        cd = drag_lift[2]
        cl = drag_lift[3]

        self.visualizer.draw_info_text(
            self.gui,
            step=self.step_counter,
            reynolds=self.lbm.Red,
            cd=cd,
            cl=cl
        )

        self.gui.show()

    def close(self):
        if self.gui is not None:
            self.gui.close()
            self.gui = None


# 测试代码
if __name__ == '__main__':
    # 打印控制参数
    print("控制参数配置:")
    for key, value in control_params.items():
        print(f"  {key}: {value}")
    
    env = LBMEnv(config={"render_mode": "human", "render_interval": 1})
    print("测试开始")
    obs, info = env.reset()
    for i in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"步数: {env.step_counter}, 角度: {env.lbm.theta}, 奖励: {reward:.4f}, CD: {info['CD']:.4f}, CL: {info['CL']:.4f}", flush=True) 
        time.sleep(0.05)  # 给 GUI 时间刷新
    env.close()
    #测试reset
    obs, info = env.reset()
    print("重置后的观测空间形状:", obs.shape)
    print("重置后的信息:", info)



