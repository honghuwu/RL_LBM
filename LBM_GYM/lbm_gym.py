import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tools.obstacles_generate.naca_genarate import obstacles_generate
from show_tools import LBMVisualizer
import gymnasium as gym
import numpy as np
import taichi as ti
import taichi.math as tm

ti.init(arch=ti.gpu, device_memory_fraction=0.8)
# ti.init(arch = ti.cpu)

@ti.data_oriented
class lbm_solver(gym.Env):

    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}

    def __init__(
        self,
        nx,
        ny,
        Red,
        inlet_velocity,
        air_para=None, # 翼型参数
        air_c=None, #弦长
        air_o=None, # 翼型原点
        air_d=0,  # 旋转中心与顶点的距离
        name='LBM Solver'  # 默认窗口标题,
    ):
        super().__init__()
        self.name = name
        self.nx = nx
        self.ny = ny
        self.Red = Red
        self.air_para = air_para
        self.air_o = air_o
        self.air_d = air_d
        self.air_c = air_c

        # 宏观物理量
        self.niu = (inlet_velocity * self.air_c) / self.Red
        self.tau = 3.0 * self.niu + 0.5
        self.inv_tau = 1.0 / self.tau
        self.dt = 1

        # LBM 参数
        self.w = ti.types.vector(9, float)(4, 1, 1, 1, 1, 1/4, 1/4, 1/4, 1/4) / 9.0
        self.e = ti.types.matrix(9, 2, float)([0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1])
        self.cs2 = 1.0 / 3.0

        # --- 欧拉场 (流场) 数据结构 ---
        self.rho = ti.field(float, shape=(nx, ny))
        self.vel = ti.Vector.field(2, float, shape=(nx, ny))
        self.f_old = ti.Vector.field(9, float, shape=(nx, ny))
        self.f_new = ti.Vector.field(9, float, shape=(nx, ny))
        self.euler_force = ti.Vector.field(2, float, shape=(nx, ny))

        # --- 拉格朗日场 (沉浸边界) 数据结构 ---
        m, p, t, alpha = self.air_para[0], self.air_para[1], self.air_para[2], self.air_para[3]

        self.num_boundary_pts, x_airfoil_np, y_airfoil_np = obstacles_generate(m, p, t, self.air_c, alpha, rot_d=self.air_d)

        boundary_pos_np = np.stack((
            x_airfoil_np + self.air_o[0],
            y_airfoil_np + self.air_o[1]
        ), axis=1)

        self.boundary_pos = ti.Vector.field(2, float, shape=self.num_boundary_pts)
        self.boundary_pos.from_numpy(boundary_pos_np.astype(np.float32))

        self.boundary_vel = ti.Vector.field(2, float, shape=self.num_boundary_pts)
        self.boundary_vel.fill(0) # 静止翼型，期望速度为0

        self.boundary_force = ti.Vector.field(2, float, shape=self.num_boundary_pts)
        self.interp_vel = ti.Vector.field(2, float, shape=self.num_boundary_pts)
        self.boundary_rho = ti.field(float, shape=self.num_boundary_pts)

        # 流动参数
        self.inlet_velocity = inlet_velocity # 入流速度

        # IBM 参数
        self.it = ti.field(dtype=ti.i32, shape=())
        self.it[None] = 0  # 初始化为0

        # 定义动作空间，这里调整攻角，后续根据控制策略进行调整
        self.action_space = gym.spaces.Box(low=-15.0, high=15.0, shape=(1,))
        
        # 定义观测空间（速度场幅值）
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.lbm.nx, self.lbm.ny),
            dtype=np.float32
        )
        
        self.reward = 0.0

    @ti.func
    def f_eq(self, i, j):
        vel_ij = self.vel[i, j]
        rho_ij = self.rho[i, j]
        uv_sq = vel_ij.dot(vel_ij)
        
        result = ti.Vector.zero(float, 9)
        for k in ti.static(range(9)):
            e_k = tm.vec2(self.e[k, 0], self.e[k, 1])
            eu = e_k.dot(vel_ij)
            result[k] = self.w[k] * rho_ij * (1.0 + eu / self.cs2 + 0.5 * (eu / self.cs2)**2 - 0.5 * uv_sq / self.cs2)
        return result

    @ti.kernel
    def init(self):
        self.vel.fill(0)
        self.rho.fill(1.0)
        for i, j in self.vel:
            self.vel[i, j].y += 1e-3 * (ti.random() - 0.5)
        for i, j in self.rho:
            self.f_old[i, j] = self.f_eq(i, j)
            self.f_new[i, j] = self.f_eq(i, j)
            

    @ti.func
    def _phi(self, x):
        r = abs(x)
        answer=0.0
        if r < 1.0:
            answer=(3.0 - 2.0 * r + ti.sqrt(1.0 + 4.0 * r - 4.0 * r ** 2)) * 0.125
        elif r < 2.0:
            answer=(5.0 - 2.0 * r - ti.sqrt(-7.0 + 12.0 * r - 4.0 * r ** 2)) * 0.125
        else:
            answer=0.0
        return answer

    @ti.func
    def discrete_delta(self, r_vec):
        return self._phi(r_vec.x) * self._phi(r_vec.y)

    @ti.kernel
    def interpolate_velocity(self):
        for k in self.boundary_pos:
            lag_pos = self.boundary_pos[k]
            ix_base, iy_base = int(lag_pos.x), int(lag_pos.y)
            interp_v = tm.vec2(0.0, 0.0)
            num = 0
            rho = 0.0
            for i_offset in range(-1, 3):
                for j_offset in range(-1, 3):
                    i, j = ix_base + i_offset, iy_base + j_offset
                    if 0 <= i < self.nx and 0 <= j < self.ny:
                        euler_pos = tm.vec2(float(i), float(j))
                        weight = self.discrete_delta(lag_pos - euler_pos)
                        interp_v += self.vel[i, j] * weight
                        num += 1
                        rho += self.rho[i, j]
            self.boundary_rho[k] = rho / num
            self.interp_vel[k] = interp_v

    @ti.kernel
    def calculate_boundary_force(self):
        for k in self.boundary_pos:
            # 使用插值得到的边界点密度
            rho_k = 1.0  # 默认密度，如果需要更精确可以从周围网格点插值
            # force = self.ibm_alpha * (self.boundary_vel[k] - self.interp_vel[k]) / self.dt
            force = 2 * rho_k * (self.boundary_vel[k] - self.interp_vel[k]) / self.dt
            self.boundary_force[k] = force

    @ti.kernel
    def spread_force(self):
        self.euler_force.fill(0)
        for k in self.boundary_pos:
            lag_pos = self.boundary_pos[k]
            lag_force = self.boundary_force[k]
            ix_base, iy_base = int(lag_pos.x), int(lag_pos.y)

            for i_offset in range(-1, 3):
                for j_offset in range(-1, 3):
                    i, j = ix_base + i_offset, iy_base + j_offset
                    if 0 <= i < self.nx and 0 <= j < self.ny:
                        euler_pos = tm.vec2(float(i), float(j))
                        weight = self.discrete_delta(lag_pos - euler_pos)
                        ti.atomic_add(self.euler_force[i, j], lag_force * weight)

    @ti.kernel
    def update_vel(self):
        for i, j in ti.ndrange(self.nx, self.ny):
            self.vel[i, j] += self.euler_force[i,j] / ( self.rho[i, j] * 2 )

    @ti.func
    def vol_force(self, i, j):
        # 计算体积力项 (Guo格式)
        f = ti.Vector.zero(float, 9)
        for k in ti.static(range(9)):
            term1 = 3*(tm.vec2(self.e[k, 0], self.e[k, 1]) - self.vel[i,j]) 
            term2 = 9*(tm.dot(tm.vec2(self.e[k, 0], self.e[k, 1]), self.vel[i,j])) * tm.vec2(self.e[k, 0], self.e[k, 1]) 
            f[k] = (1-0.5*self.inv_tau)*self.w[k]*tm.dot(term1 + term2, self.euler_force[i,j])
        return f

    @ti.kernel
    def collision(self):
        for i, j in ti.ndrange(self.nx, self.ny):
            feq_ij = self.f_eq(i, j)
            vol_f = self.vol_force(i, j)  
            
            for k in ti.static(range(9)):
                self.f_new[i,j][k] = (self.f_old[i,j][k] - 
                                    self.inv_tau * (self.f_old[i,j][k] - feq_ij[k]) + vol_f[k])

    @ti.kernel
    def streaming(self):
        """ LBM迁移步 (Pull Scheme) - 周期性边界版本 """
        for i, j in ti.ndrange(self.nx, self.ny):
            for k in ti.static(range(9)):
                # 计算源网格点坐标
                ip = i - ti.cast(self.e[k, 0], ti.i32)
                jp = j - ti.cast(self.e[k, 1], ti.i32)

                # 左右边界保持周期性
                if ip < 0: ip = self.nx - 1
                if ip > self.nx - 1: ip = 0
                
                # 上下边界也使用周期性边界条件
                if jp < 0: jp = self.ny - 1
                if jp > self.ny - 1: jp = 0
                
                # 从源网格点拉取数据
                self.f_old[i, j][k] = self.f_new[ip, jp][k]

    @ti.kernel
    def update_macro_vars(self):
        for i, j in ti.ndrange(self.nx, self.ny):
            # 1. 从分布函数计算密度和动量
            new_rho = 0.0
            momentum = tm.vec2(0.0, 0.0)
            for k in ti.static(range(9)):
                f_val = self.f_old[i, j][k]  
                new_rho += f_val
                e_k_vec = tm.vec2(self.e[k, 0], self.e[k, 1])
                momentum += f_val * e_k_vec
            
            self.rho[i, j] = new_rho
            
            if new_rho > 1e-6:
                self.vel[i, j] = momentum / new_rho
            else:
                self.vel[i, j] = tm.vec2(0.0, 0.0)

    @ti.kernel
    def apply_bc(self):
        """ 施加外围边界条件 - 周期性边界版本 """
        i = 0  # 左边界位置
        # for j in range(self.ny):
        #     # 设定边界条件
        #     self.vel[i, j] = tm.vec2(self.inlet_velocity, 0.0)
        #     self.rho[i, j] = self.rho[i+1, j]  # 固定密度
        #     self.f_old[i, j] = self.f_eq(i, j) - self.f_eq(i+1, j) + self.f_old[1, j]

        for j in range(self.ny):
            # 设定边界条件
            v = ti.abs(0.1 * ti.sin(self.it[None] / 1000)) + 0.05
            self.vel[i, j] = tm.vec2(v, 0.0)
            self.rho[i, j] = self.rho[i+1, j]  # 固定密度
            self.f_old[i, j] = self.f_eq(i, j) - self.f_eq(i+1, j) + self.f_old[1, j]

        # 右边界：出流条件
        for j in range(self.ny):
            self.vel[self.nx - 1, j] = self.vel[self.nx - 2, j]
            self.rho[self.nx - 1, j] = self.rho[self.nx - 2, j]
            self.f_old[self.nx - 1, j] = self.f_eq(self.nx - 1, j)


    @ti.kernel
    def calculate_drag_lift(self) -> tm.vec4:
        """ 计算升力、阻力、升力系数和阻力系数 """
        drag = 0.0
        lift = 0.0

        for k in self.boundary_pos:
            force = -self.boundary_force[k]
            # 计算升力和阻力
            drag += force.dot(tm.vec2(1.0, 0.0))
            lift += force.dot(tm.vec2(0.0, 1.0))

        # 计算升力系数和阻力系数
        rho_ref = 1.0  # 参考密度
        dynamic_pressure = 0.5 * rho_ref * self.inlet_velocity * self.inlet_velocity * self.air_c
        
        cd = 0.0
        cl = 0.0
        if dynamic_pressure > 1e-12:
            cd = drag / dynamic_pressure
            cl = lift / dynamic_pressure

        return tm.vec4(drag, lift, cd, cl)

    def step(self):
        self.interpolate_velocity()
        self.calculate_boundary_force()
        self.spread_force() 
        self.update_vel()
        self.collision()
        self.streaming()
        self.update_macro_vars()
        self.apply_bc()