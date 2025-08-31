import gymnasium as gym
import numpy as np
from gymnasium import spaces
import taichi as ti
import time


ti.init(arch = ti.cpu)

from core.lbm_rotated import lbm_solver
from tools.show_tools.visualizer import LBMVisualizer

class LBMEnv(gym.Env):
    metadata = {"render_modes": ["human", None], "render_fps": 30}

    def __init__(self, config=None):
        super(LBMEnv, self).__init__()
        if config is None:
            config = {}

        # 渲染开关（config 中传 render_mode="human" 启用渲染）
        self.render_mode = config.get("render_mode", None)
        self.render_interval = config.get("render_interval", 10)  # 每隔多少 step 渲染一次
        
        # Episode 长度限制
        self.max_episode_steps = config.get("max_episode_steps", 200)  # 默认最大200步

        #角度
        self.theta = self.theta0 = 20
        # 初始化 LBM 求解器
        self.lbm = lbm_solver(
            nx=400,
            ny=200,
            Red=1000,
            inlet_velocity=0.1,
            air_c=100,
            air_para=[0, 0, 12.0, -self.theta],
            air_o=[100.0, 100.0]
        )

        # 初始化可视化器与 GUI（延迟到第一次渲染时创建）
        self.visualizer = LBMVisualizer(self.lbm.nx, self.lbm.ny)
        self.gui = None

        # 动作空间
        self.action_space = spaces.Box(
            low=np.array([-5.0], dtype=np.float32),
            high=np.array([5.0], dtype=np.float32),
            dtype=np.float32
        )


        # 观测空间
        #self.lbm.init()

        sample_obs = self.lbm.output().astype(np.float32)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=sample_obs.shape, dtype=np.float32
        )

        self._target_obs_len = sample_obs.shape[0]


        # 计步器
        self.step_counter = 0

        self.reward = 0.0

    def _fix_obs_shape(self, obs):
        cur_len = obs.shape[0]
        target_len = self._target_obs_len
        if cur_len > target_len:
            # 裁剪
            obs = obs[:target_len, :]
        elif cur_len < target_len:
            # 填充
            pad = np.zeros((target_len - cur_len, obs.shape[1]), dtype=obs.dtype)
            obs = np.vstack([obs, pad])
        return obs


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.lbm.init()
        self.step_counter = 0
        obs = self.lbm.output().astype(np.float32)
        obs = self._fix_obs_shape(obs)
        print("obs.shape =", obs.shape, "obs_space =", self.observation_space.shape)
        info = {}
        return obs, info

    # def reset(self, seed=None, options=None):
    #     super().reset(seed=seed)
    #     # 只有在第一次调用或仿真崩溃后才真正重置
    #     if not self.has_been_initialized or self.lbm_solver.is_unstable:
    #         print("Performing a REAL reset of the LBM simulation.")
    #         self.has_been_initialized = True
        
    #     # 对于常规的"episode"结束，我们不重置流场
    #     # 只是返回当前流场的状态
    #     state = self.lbm_solver.get_state()
    #     info = {} # 重置时info通常为空
    #     return state, info


    def step(self, action):
        '''
        施加动作,对theta进行5度以内的加减,并且上下限是10-30度
        '''
        control_val = float(np.clip(action[0], -10.0, 10.0))

        if self.theta + control_val < 10.0 :
            self.theta = 10.0
        elif self.theta + control_val > 30.0:
            self.theta = 30.0
        else:
            self.theta += control_val

        self.lbm.control(self.theta)

        for _ in range(10):
            self.lbm.step()

        obs = self.lbm.output().astype(np.float32)

        obs = self._fix_obs_shape(obs)

        # 获取升阻力数据并转换为 numpy 数组
        drag_lift_vec = self.lbm.calculate_drag_lift()
        drag_lift = np.array([drag_lift_vec[0], drag_lift_vec[1], drag_lift_vec[2], drag_lift_vec[3]])
        cd = drag_lift[2]
        cl = drag_lift[3]

        reward = cl/cd -0.05 * np.abs(self.theta - self.theta0)

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
            for _ in range(50):
                self.lbm.step()

        # 更新可视化数据
        self.visualizer.update_visualization(self.lbm.vel)
        self.gui.set_image(self.visualizer.get_combined_image())

        # 绘制边界和控制器
        self.visualizer.draw_boundary_points(self.gui, self.lbm.boundary_pos)
        #if hasattr(self.lbm, 'controller_pos'):
        #    self.visualizer.draw_controller_points(self.gui, self.lbm.controller_pos)

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

        # 事件轮询
        for e in self.gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                self.close()
                exit()

        self.gui.show()

    def close(self):
        if self.gui is not None:
            self.gui.close()
            self.gui = None


# 测试代码
if __name__ == '__main__':
    env = LBMEnv(config={"render_mode": "human", "render_interval": 1})
    print("测试开始")
    obs, info = env.reset()
    for i in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"实际渲染间隔: {env.render_interval}", flush=True) 
        time.sleep(0.05)  # 给 GUI 时间刷新
    env.close()


