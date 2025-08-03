"""
=============================================================================
LBM求解器 - 基础版本 (lbm.py)
=============================================================================

文件描述:
    基于Taichi的格子玻尔兹曼方法(LBM)流体求解器，结合浸没边界法(IBM)
    处理翼型绕流问题。这是项目的基础版本，实现了核心的LBM-IBM耦合算法。

主要功能:
    - D2Q9格子玻尔兹曼方法求解不可压缩流体
    - 浸没边界法处理复杂几何边界(NACA翼型)
    - Guo力源项格式处理体积力
    - 实时可视化(matplotlib动画)
    - 升阻力计算和监控

核心算法:
    1. LBM碰撞-迁移步骤
    2. IBM插值-力计算-分布三步骤
    3. Guo力源项修正
    4. 周期性边界条件

技术特点:
    - 使用Taichi GPU加速计算
    - 离散Delta函数实现IBM
    - 自定义涡度颜色映射
    - 双场可视化(涡度场+速度场)

版本信息:
    - 版本: v1.0 (基础版本)
    - 创建日期: 2024
    - 可视化: matplotlib动画
    - 边界条件: 周期性边界
    - 力源项: Guo格式

作者: 于志鸿
=============================================================================
"""

import sys
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.colors

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tools.obstacles_generate.naca_genarate import obstacles_generate
import taichi as ti
import taichi.math as tm
from taichi.types import i32

ti.init(arch=ti.gpu, device_memory_fraction=0.8)
# ti.init(arch = ti.cpu)
@ti.data_oriented
class lbm_solver:
    def __init__(
        self,
        nx,
        ny,
        Red,
        air_para=None, # 翼型参数
        air_c=None, #弦长
        air_o=None, # 翼型原点
        air_d=0,  # 旋转中心与顶点的距离
        name='LBM Solver'  # 默认窗口标题,
    ):
        self.name = name
        self.nx = nx
        self.ny = ny
        self.Red = Red
        self.air_para = air_para
        self.air_o = air_o
        self.air_d = air_d
        self.air_c = air_c

        # 宏观物理量
        '''niu的定义需要重新考虑'''
        self.niu = (0.1 * self.air_c) / self.Red
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

        # IBM 参数
        self.ibm_alpha = 1.2 
        self.it = ti.field(dtype=ti.i32, shape=())
        self.it[None] = 0  # 初始化为0

    @ti.func
    def _phi(self, r):
        """ 一维离散Delta核函数 (辅助函数) """
        r_abs = abs(r)
        res = 0.0
        if r_abs < 1.0:
            res = (1/8) * (3 - 2*r_abs + tm.sqrt(1 + 4*r_abs - 4*r_abs**2))
        elif r_abs < 2.0:
            res = (1/8) * (5 - 2*r_abs - tm.sqrt(-7 + 12*r_abs - 4*r_abs**2))
        return res

    @ti.func
    def discrete_delta(self, r_vec):
        """ 二维离散Delta函数, r_vec是(拉格朗日点 - 欧拉点)的距离向量 """
        return self._phi(r_vec.x) * self._phi(r_vec.y)

    @ti.kernel
    def interpolate_velocity(self):
        """ IBM步骤1: 插值 - 计算每个拉格朗日点上的流体速度 """
        for k in self.boundary_pos:
            lag_pos = self.boundary_pos[k]
            ix_base, iy_base = int(lag_pos.x), int(lag_pos.y)

            interp_v = tm.vec2(0.0, 0.0)
            # 遍历周围 4x4 的欧拉格点 (因为Delta函数的影响范围是 +/- 2)
            for i_offset in range(-1, 3):
                for j_offset in range(-1, 3):
                    i, j = ix_base + i_offset, iy_base + j_offset
                    if 0 <= i < self.nx and 0 <= j < self.ny:
                        euler_pos = tm.vec2(float(i), float(j))
                        weight = self.discrete_delta(lag_pos - euler_pos)
                        # 【修改】从 self.vel 读取速度，而不是不存在的 inter_vel
                        interp_v += self.vel[i, j] * weight
            self.interp_vel[k] = interp_v

    @ti.kernel
    def calculate_boundary_force(self):
        """ IBM步骤2: 力计算 - 根据速度差计算反馈力 """
        for k in self.boundary_pos:
            force = self.ibm_alpha * (self.boundary_vel[k] - self.interp_vel[k]) / self.dt
            self.boundary_force[k] = force

    @ti.kernel
    def spread_force(self):
        """ IBM步骤3: 分布 - 将拉格朗日点的力散播回欧拉网格 """
        self.euler_force.fill(0) # 每一步开始前必须清零力场
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

    @ti.func
    def guo_force_term(self, i, j):
        """计算Guo力源项G_i"""
        F = self.euler_force[i, j]
        u = self.vel[i, j]  # 使用受力前的速度
        rho = self.rho[i, j]
        G_i = ti.Vector.zero(float, 9)
        
        for k in ti.static(range(9)):
            e_k = tm.vec2(self.e[k, 0], self.e[k, 1])
            eu = e_k.dot(u)
            F_dot_e = F.dot(e_k)
            F_dot_u = F.dot(u)
            
            term1 = F_dot_e / self.cs2
            term2 = eu * F_dot_e / (self.cs2 * self.cs2)
            term3 = F_dot_u / self.cs2
            
            G_i[k] = self.w[k] * (1.0 - 0.5/self.tau) * (term1 + term2 - term3)
        
        return G_i

    @ti.func
    def f_eq(self, i, j):
        """ 计算平衡分布函数 """
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
        """ 初始化流场 """
        self.vel.fill(0)
        self.rho.fill(1.0)
        # 根据初始化的宏观量，计算初始的平衡分布函数
        for i, j in self.rho:
            self.f_old[i, j] = self.f_eq(i, j)
            self.f_new[i, j] = self.f_eq(i, j)

    @ti.kernel
    def collision(self):
        """LBM碰撞步（加入Guo力源项）"""
        for i, j in ti.ndrange(self.nx, self.ny):
            feq_ij = self.f_eq(i, j)
            G_i = self.guo_force_term(i, j)  # 计算力源项
            
            for k in ti.static(range(9)):
                # 加入力源项修正
                self.f_new[i,j][k] = (self.f_old[i,j][k] - 
                                    self.inv_tau * (self.f_old[i,j][k] - feq_ij[k]) +
                                    self.dt * G_i[k])

    @ti.kernel
    def streaming(self):
        """ LBM迁移步 (Pull Scheme) """
        for i, j in ti.ndrange(self.nx, self.ny):
            for k in ti.static(range(9)):
                # 计算源网格点坐标
                ip = i - ti.cast(self.e[k, 0], ti.i32)
                jp = j - ti.cast(self.e[k, 1], ti.i32)

                # 周期性边界（注意：非周期性边界会在apply_bc中被重置）
                if ip < 0: ip = self.nx - 1
                if ip > self.nx - 1: ip = 0
                if jp < 0: jp = self.ny - 1
                if jp > self.ny - 1: jp = 0
                
                # 从源网格点拉取数据
                self.f_old[i, j][k] = self.f_new[ip, jp][k]

    @ti.kernel
    def update_macro_vars(self):
        """ 【新增】根据分布函数和IBM力，更新最终的宏观密度和速度 """
        for i, j in ti.ndrange(self.nx, self.ny):
            # 1. 从分布函数计算密度和动量
            new_rho = 0.0
            momentum = tm.vec2(0.0, 0.0)
            for k in ti.static(range(9)):
                f_val = self.f_old[i, j][k] # 使用迁移后的 f_old
                new_rho += f_val
                e_k_vec = tm.vec2(self.e[k, 0], self.e[k, 1])
                momentum += f_val * e_k_vec
            
            self.rho[i, j] = new_rho
            
            # 2. 【核心修正】使用郭氏力格式更新最终速度
            # vel = (momentum + 0.5 * dt * F) / rho
            force_ij = self.euler_force[i, j]
            if new_rho > 1e-6:
                self.vel[i, j] = (momentum + 0.5 * self.dt * force_ij) / new_rho
            else:
                self.vel[i, j] = tm.vec2(0.0, 0.0)

    @ti.kernel
    def apply_bc(self):
        """ 施加外围边界条件 """

        for j in range(self.ny):
            sin_val = ti.sin(self.it[None] / 1000.0)  
            self.vel[0, j] = ti.abs(sin_val) * ti.Vector([0.3, 0.0]) + ti.Vector([0.05, 0.0])
            # self.vel[0, j] = ti.Vector([0.3, 0.0])
            self.rho[0, j] = 1.0
            self.f_old[0, j] = self.f_eq(0, j)


        for j in range(self.ny):
            self.vel[self.nx - 1, j] = self.vel[self.nx - 2, j]
            self.rho[self.nx - 1, j] = self.rho[self.nx - 2, j]
            self.f_old[self.nx - 1, j] = self.f_eq(self.nx - 1, j)

        for i in range(self.nx):
            self.vel[i, self.ny - 1] = self.vel[i, self.ny - 2]
            self.rho[i, self.ny - 1] = self.rho[i, self.ny - 2]
            self.f_old[i, self.ny - 1] = self.f_eq(i, self.ny - 1)

        for i in range(self.nx):
            self.vel[i, 0] = self.vel[i, 1]
            self.rho[i, 0] = self.rho[i, 1]
            self.f_old[i, 0] = self.f_eq(i, 0)

    @ti.kernel
    def caculate_drag_lift(self) -> tm.vec2:
        """ 计算升力和阻力 """
        drag = 0.0
        lift = 0.0

        for k in self.boundary_pos:
            force = -self.boundary_force[k]
            vel = -self.interp_vel[k]
            # 计算升力和阻力
            drag += force.dot(tm.vec2(1.0, 0.0))
            lift -= force.dot(tm.vec2(0.0, 1.0))

        return tm.vec2(drag, lift)


    def step(self):
        # 1. IBM: 根据 t-1 时刻的速度场计算 t 时刻的力
        self.interpolate_velocity()
        self.calculate_boundary_force()
        self.spread_force() 

        # 2. LBM碰撞: 使用 t-1 的宏观量，从 f_old 计算碰撞，结果写入 f_new
        self.collision()

        # 3. LBM迁移: 从 f_new 拉取数据到 f_old (采用周期性边界)
        self.streaming()
        
        # 4. 宏观量更新: 根据 t 时刻的分布函数(f_old)和力(euler_force)计算 t 时刻的最终宏观量(rho, vel)
        self.update_macro_vars()

        # 5. 施加边界条件: 用 t 时刻最终的宏观量，修正边界上的分布函数(f_old)，为下一轮做准备
        self.apply_bc()

    def solver(self, steps=10000):
        """ 运行指定步数的LBM求解 """
        self.init()
        for _ in range(steps):
            self.step()
            self.it[None] += 1
            if self.it[None] % 100 == 0:
                drag_lift = self.caculate_drag_lift()
                drag = drag_lift.x
                lift = drag_lift.y
                if ti.abs(drag) > 1e-6:
                    k = lift / drag
                print(f"迭代步数: {self.it[None]}, 阻力: {drag}, 升力: {lift},升阻比: {k if ti.abs(drag) > 1e-6 else 'N/A'}")




    def show(self):
        matplotlib.use('TkAgg')  # 设置后端
 
        self.init()
        boundary_pts = self.boundary_pos.to_numpy()
        
        # 创建自定义涡度颜色映射（参考only_rotate.ipynb）
        colors = [
            (1, 1, 0),        # 黄色
            (0.953, 0.490, 0.016),  # 橙色
            (0, 0, 0),        # 黑色
            (0.176, 0.976, 0.529),  # 绿色
            (0, 1, 1),        # 青色
        ]
        my_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("my_cmap", colors)
        
        # 创建图形和子图
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 6))
        
        def update_plot(frame):
            # 执行多个时间步
            for _ in range(10):
                self.it[None] += 1
                self.step()
            
            if self.it[None] % 100 == 0:
                drag_lift = self.caculate_drag_lift()
                drag = drag_lift.x
                lift = drag_lift.y
                if drag > 1e-6:
                    k = lift / drag
                print(f"迭代步数: {self.it[None]}, 阻力: {drag}, 升力: {lift},升阻比: {k if drag > 1e-6 else 'N/A'}")
            
            # 获取速度场数据
            vel = self.vel.to_numpy()
            
            # 计算涡度场 (∂v/∂x - ∂u/∂y)
            ugrad = np.gradient(vel[:, :, 0])  # u分量的梯度
            vgrad = np.gradient(vel[:, :, 1])  # v分量的梯度
            vorticity = vgrad[0] - ugrad[1]    # 涡度 = ∂v/∂x - ∂u/∂y
            
            # 计算速度大小
            vel_mag = np.linalg.norm(vel, axis=-1)
            
            # 清除之前的图像
            ax1.clear()
            ax2.clear()
            
            # 显示涡度场（左侧）
            im1 = ax1.imshow(vorticity.T, origin='lower', cmap=my_cmap, 
                            vmin=-0.02, vmax=0.02, extent=[0, self.nx, 0, self.ny])
            ax1.scatter(boundary_pts[:, 0], boundary_pts[:, 1], c='black', s=1)
            ax1.set_title('Vorticity Field')
            ax1.set_xlabel('x')
            ax1.set_ylabel('y')
            
            # 显示速度场（右侧）
            im2 = ax2.imshow(vel_mag.T, origin='lower', cmap='viridis', 
                            vmin=0, vmax=0.15, extent=[0, self.nx, 0, self.ny])
            ax2.scatter(boundary_pts[:, 0], boundary_pts[:, 1], c='black', s=1)
            ax2.set_title('Velocity Magnitude')
            ax2.set_xlabel('x')
            ax2.set_ylabel('y')
            
            return [im1, im2]
        
        # 创建动画
        ani = FuncAnimation(fig, update_plot, interval=50, blit=False, cache_frame_data=False)

        plt.tight_layout()
        plt.show()



if __name__ == '__main__':
    lbm = lbm_solver(
        nx=801,
        ny=401,
        Red=1000, # 雷诺数降低以匹配较低的格子速度
        air_c=300, # 弦长适当减小
        air_para=[0, 0, 12.0, -30.0],
        air_o=[200.0, 200.0],
    )
    lbm.show()
    # lbm.solver(steps=10000)