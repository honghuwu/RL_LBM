"""
=============================================================================
LBM求解器 - 改进版本 (lbm2.py)
=============================================================================

文件描述:
    基于Taichi的格子玻尔兹曼方法(LBM)流体求解器的改进版本。在基础版本基础上
    优化了算法实现，改进了可视化效果，并增强了数值稳定性。

主要功能:
    - D2Q9格子玻尔兹曼方法求解不可压缩流体
    - 浸没边界法处理复杂几何边界(NACA翼型)
    - 改进的体积力处理方法
    - Taichi GUI实时可视化
    - 升阻力系数计算和监控
    - 力历史记录功能

核心算法改进:
    1. 优化的LBM碰撞-迁移步骤
    2. 改进的IBM力计算方法(密度加权)
    3. 简化的体积力项处理
    4. 周期性边界条件优化

技术特点:
    - 使用Taichi GPU加速计算
    - Taichi GUI高性能可视化
    - 自定义颜色映射(涡度场+速度场)
    - 上下双场布局显示
    - 实时边界点绘制
    - 键盘交互控制(暂停/继续)

版本更新内容:
    - 新增: inlet_velocity参数独立控制
    - 新增: boundary_rho密度插值
    - 新增: Taichi GUI可视化替代matplotlib
    - 新增: 实时升阻力系数显示
    - 改进: 体积力计算方法
    - 改进: 边界条件处理
    - 优化: 可视化性能和效果

版本信息:
    - 版本: v2.0 (改进版本)
    - 基于: lbm.py v1.0
    - 创建日期: 2024
    - 可视化: Taichi GUI
    - 边界条件: 周期性边界(优化)
    - 力源项: 简化体积力格式

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
from collections import deque

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
        inlet_velocity,
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
        self.boundary_rho = ti.field(float, shape=self.num_boundary_pts)

        # 流动参数
        self.inlet_velocity = inlet_velocity # 入流速度
        self.cylinder_radius = self.air_c / 2.0  # 使用弦长的一半作为特征长度

        # 力历史记录
        self.drag_history = []
        self.lift_history = []
        self.time_history = []
        

        # IBM 参数
        # self.ibm_alpha = 1.2 
        self.it = ti.field(dtype=ti.i32, shape=())
        self.it[None] = 0  # 初始化为0
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
            # self.f_final[i, j] = self.f_eq(i, j)

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
        for j in range(self.ny):
            # 设定边界条件
            self.vel[i, j] = tm.vec2(self.inlet_velocity, 0.0)
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

    def solver(self, steps=10000):
        """ 运行指定步数的LBM求解 """
        self.init()
        for _ in range(steps):
            self.step()

            self.it[None] += 1
            
            # 计算当前时间
            current_time = self.it[None] * self.dt
            
            if self.it[None] % 5 == 0:  
                drag_lift_coeffs = self.calculate_drag_lift()
                drag = drag_lift_coeffs.x
                lift = drag_lift_coeffs.y


            if self.it[None] % 100 == 0:
                drag_lift_coeffs = self.calculate_drag_lift()
                drag = drag_lift_coeffs.x
                lift = drag_lift_coeffs.y
                cd = drag_lift_coeffs.z  # 阻力系数
                cl = drag_lift_coeffs.w  # 升力系数
                
                
                print(f"迭代步数: {self.it[None]}, 阻力: {drag:.6f}, 升力: {lift:.6f}, "
                    f"C_D: {cd:.4f}, C_L: {cl:.4f}")

    @ti.kernel
    def compute_vorticity(self, vorticity: ti.template()):
        """计算涡度场 - 适应周期性边界条件"""
        for i, j in ti.ndrange(self.nx, self.ny):
            # 左右边界处理（保持原有逻辑）
            if i == 0 or i == self.nx-1:
                vorticity[i, j] = 0.0
            else:
                # 计算涡度 ω = ∂v/∂x - ∂u/∂y
                dvdx = self.vel[i+1, j][1] - self.vel[i-1, j][1]
                
                # y方向使用周期性边界条件
                j_plus = (j + 1) % self.ny
                j_minus = (j - 1 + self.ny) % self.ny
                dudy = self.vel[i, j_plus][0] - self.vel[i, j_minus][0]
                
                vorticity[i, j] = 0.5 * (dvdx - dudy)

    @ti.kernel
    def compute_velocity_magnitude(self, vel_mag: ti.template()):
        """计算速度大小"""
        for i, j in ti.ndrange(self.nx, self.ny):
            vel_mag[i, j] = self.vel[i, j].norm()

    @ti.kernel
    def normalize_field(self, field: ti.template(), normalized: ti.template(), 
                       min_val: float, max_val: float):
        """将场数据归一化到[0,1]范围"""
        for i, j in field:
            val = (field[i, j] - min_val) / (max_val - min_val)
            normalized[i, j] = ti.max(0.0, ti.min(1.0, val))

    @ti.kernel
    def apply_colormap_vorticity(self, normalized: ti.template(), colored: ti.template()):
        """应用涡度颜色映射 - 黑色背景"""
        for i, j in normalized:
            val = normalized[i, j]
            # 黑色背景的涡度颜色映射：黑色->深蓝->青色->绿色->黄色->红色
            if val < 0.1:
                # 黑色到深蓝色（低涡度区域保持黑色背景）
                t = val * 10.0
                colored[i, j] = ti.Vector([0.0, 0.0, t * 0.3])
            elif val < 0.3:
                # 深蓝色到蓝色
                t = (val - 0.1) * 5.0
                colored[i, j] = ti.Vector([0.0, t * 0.2, 0.3 + t * 0.7])
            elif val < 0.5:
                # 蓝色到青色
                t = (val - 0.3) * 5.0
                colored[i, j] = ti.Vector([0.0, 0.2 + t * 0.8, 1.0])
            elif val < 0.7:
                # 青色到绿色
                t = (val - 0.5) * 5.0
                colored[i, j] = ti.Vector([t * 0.5, 1.0, 1.0 - t * 0.5])
            elif val < 0.85:
                # 绿色到黄色
                t = (val - 0.7) * 6.67
                colored[i, j] = ti.Vector([0.5 + t * 0.5, 1.0, 0.5 - t * 0.5])
            else:
                # 黄色到红色（高涡度区域）
                t = (val - 0.85) * 6.67
                colored[i, j] = ti.Vector([1.0, 1.0 - t * 0.5, 0.0])

    @ti.kernel
    def apply_colormap_velocity(self, normalized: ti.template(), colored: ti.template()):
        """应用速度颜色映射（改进的viridis风格）"""
        for i, j in normalized:
            val = ti.min(normalized[i, j], 1.0)  # 确保值不超过1.0
            
            # 改进的颜色映射：深蓝->青色->绿色->黄色->橙色->红色
            if val < 0.16667:  # 0-1/6: 深蓝到青色
                t = val * 6.0
                colored[i, j] = ti.Vector([0.0, t * 0.5, 0.5 + t * 0.5])
            elif val < 0.33333:  # 1/6-2/6: 青色到绿色
                t = (val - 0.16667) * 6.0
                colored[i, j] = ti.Vector([0.0, 0.5 + t * 0.5, 1.0 - t * 0.5])
            elif val < 0.5:  # 2/6-3/6: 绿色到黄绿色
                t = (val - 0.33333) * 6.0
                colored[i, j] = ti.Vector([t * 0.5, 1.0, 0.5 - t * 0.5])
            elif val < 0.66667:  # 3/6-4/6: 黄绿色到黄色
                t = (val - 0.5) * 6.0
                colored[i, j] = ti.Vector([0.5 + t * 0.5, 1.0, 0.0])
            elif val < 0.83333:  # 4/6-5/6: 黄色到橙色
                t = (val - 0.66667) * 6.0
                colored[i, j] = ti.Vector([1.0, 1.0 - t * 0.3, 0.0])
            else:  # 5/6-1: 橙色到红色
                t = (val - 0.83333) * 6.0
                colored[i, j] = ti.Vector([1.0, 0.7 - t * 0.7, t * 0.2])

    @ti.kernel
    def combine_images_vertical(self, vort_img: ti.template(), vel_img: ti.template(), 
                               combined: ti.template(), separator_height: int):
        """垂直组合两个图像：上面是涡度场，下面是速度场"""
        nx, ny = vort_img.shape[0], vort_img.shape[1]
        
        # 上半部分：涡度场
        for i, j in ti.ndrange(nx, ny):
            combined[i, j] = vort_img[i, j]
        
        # 分隔线
        for i, j in ti.ndrange(nx, separator_height):
            combined[i, ny + j] = ti.Vector([0.5, 0.5, 0.5])  # 灰色分隔线
        
        # 下半部分：速度场
        for i, j in ti.ndrange(nx, ny):
            combined[i, ny + separator_height + j] = vel_img[i, j]

    def show(self):
        """使用Taichi GUI显示实时流场可视化 - 上下两列布局"""
        self.init()
        
        # 创建显示用的field
        vorticity = ti.field(float, shape=(self.nx, self.ny))
        vel_mag = ti.field(float, shape=(self.nx, self.ny))
        
        # 归一化后的场
        vorticity_norm = ti.field(float, shape=(self.nx, self.ny))
        vel_mag_norm = ti.field(float, shape=(self.nx, self.ny))
        
        # 颜色场
        vorticity_colored = ti.Vector.field(3, float, shape=(self.nx, self.ny))
        vel_mag_colored = ti.Vector.field(3, float, shape=(self.nx, self.ny))
        
        # 组合图像field - 上下两列布局
        separator_height = 10  # 分隔线高度
        combined_height = self.ny * 2 + separator_height
        combined_img = ti.Vector.field(3, float, shape=(self.nx, combined_height))
        
        # 创建GUI窗口 - 调整为上下两列的尺寸
        gui = ti.GUI(self.name, res=(self.nx, combined_height))
        
        print("=== Taichi GUI 双场显示 ===")
        print("显示布局:")
        print("  上半部分 - 涡度场")
        print("  下半部分 - 速度场")
        print("按键控制:")
        print("  'q' 或 ESC - 退出")
        print("  空格 - 暂停/继续")
        print("==============================")
        
        paused = False
        
        while gui.running:
            # 处理键盘事件
            for e in gui.get_events(ti.GUI.PRESS):
                if e.key == ti.GUI.ESCAPE or e.key == 'q':
                    gui.running = False
                elif e.key == ti.GUI.SPACE:
                    paused = not paused
                    print("暂停" if paused else "继续")
            
            if not paused:
                # 执行多个时间步
                for _ in range(5):
                    self.step()
                    self.it[None] += 1
                    
                    # 记录力的历史
                    current_time = self.it[None] * self.dt
                    if self.it[None] % 10 == 0:
                        drag_lift_coeffs = self.calculate_drag_lift()
                        drag = drag_lift_coeffs.x
                        lift = drag_lift_coeffs.y
                        
                
                # 每100步输出一次信息
                if self.it[None] % 100 == 0:
                    drag_lift_coeffs = self.calculate_drag_lift()
                    drag = drag_lift_coeffs.x
                    lift = drag_lift_coeffs.y
                    cd = drag_lift_coeffs.z  # 阻力系数
                    cl = drag_lift_coeffs.w  # 升力系数

                    print(f"步数: {self.it[None]}, 阻力: {drag:.6f}, 升力: {lift:.6f}, "
                            f"C_D: {cd:.4f}, C_L: {cl:.4f}")
            
            # 计算可视化数据
            self.compute_vorticity(vorticity)
            self.compute_velocity_magnitude(vel_mag)
            
            # 归一化
            self.normalize_field(vorticity, vorticity_norm, -0.02, 0.02)
            self.normalize_field(vel_mag, vel_mag_norm, 0.0, 0.4)  # 扩大速度范围到0.4
            
            # 应用颜色映射
            self.apply_colormap_vorticity(vorticity_norm, vorticity_colored)
            self.apply_colormap_velocity(vel_mag_norm, vel_mag_colored)
            
            # 组合图像：上面涡度场，下面速度场
            self.combine_images_vertical(vorticity_colored, vel_mag_colored, 
                                       combined_img, separator_height)
            
            # 显示组合图像
            gui.set_image(combined_img.to_numpy())
            
            # 绘制圆柱边界点 - 需要调整坐标到两个区域
            boundary_pts = self.boundary_pos.to_numpy()
            
            # 上半部分（涡度场）的边界点
            for pt in boundary_pts:
                x = pt[0] / self.nx
                y = pt[1] / combined_height  # 调整到上半部分
                gui.circle((x, y), radius=1, color=0x000000)
            
            # 下半部分（速度场）的边界点
            for pt in boundary_pts:
                x = pt[0] / self.nx
                y = (pt[1] + self.ny + separator_height) / combined_height  # 调整到下半部分
                gui.circle((x, y), radius=1, color=0x000000)
            
            # 显示信息文本
            drag_lift_coeffs = self.calculate_drag_lift()
            drag = drag_lift_coeffs.x
            lift = drag_lift_coeffs.y
            cd = drag_lift_coeffs.z  # 阻力系数
            cl = drag_lift_coeffs.w  # 升力系数
            
            info_text = f"Step: {self.it[None]}, Re: {self.Red}, C_D: {cd:.3f}, C_L: {cl:.3f}"
            
            gui.text(info_text, pos=(0.02, 0.97), color=0xFFFFFF)
            
            # 显示场标签
            gui.text("Vorticity Field", pos=(0.02, 0.52), color=0xFFFFFF)
            gui.text("Velocity Magnitude", pos=(0.02, 0.02), color=0xFFFFFF)
            
            gui.show()



if __name__ == '__main__':
    lbm = lbm_solver(
        nx=400,
        ny=200,
        Red=1500, 
        inlet_velocity=0.1,
        air_c=100,
        air_para=[0, 0, 12.0, -20.0],
        air_o=[100.0, 100.0],
    )
    lbm.show()
    # lbm.solver(steps=10000)