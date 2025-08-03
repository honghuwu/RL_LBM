#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
圆柱绕流LBM求解器 - 沉浸边界法实现（周期性边界版本）

本程序实现了基于格子玻尔兹曼方法(LBM)和沉浸边界法(IBM)的圆柱绕流数值模拟。
主要功能包括：
1. D2Q9格子玻尔兹曼方法求解不可压缩流动
2. 沉浸边界法处理复杂几何边界（圆柱）
3. 实时计算升力、阻力系数和斯特劳哈尔数
4. 双场可视化显示（涡度场和速度场）
5. 支持周期性边界条件（上下边界）和入流/出流边界条件（左右边界）

物理模型：
- 雷诺数可调的圆柱绕流
- 卡门涡街现象的捕捉和分析
- 基于Dupuis格式的力源项处理

边界条件：
- 左边界：入流条件（固定速度和密度）
- 右边界：出流条件（零梯度）
- 上下边界：周期性边界条件

技术特点：
- 使用Taichi框架实现GPU加速计算
- 实时可视化和交互式控制
- 高精度的斯特劳哈尔数计算（基于Welch功率谱密度方法）
- Dupuis格式的力源项处理，确保数值稳定性


作者：LBM项目组
版本：2.2（周期性边界版本）
"""

import sys
import os
import numpy as np
from scipy import signal
from collections import deque

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tools.obstacles_generate.column_generate import obstacles_generate_cylinder
import taichi as ti
import taichi.math as tm
from taichi.types import i32

ti.init(arch=ti.gpu, device_memory_fraction=0.8)
# ti.init(arch = ti.cpu)

@ti.data_oriented
class CylinderFlowSolver:
    def __init__(
        self,
        nx,
        ny,
        Red,
        inlet_velocity=0.5,  # 入流速度
        cylinder_radius=50,  # 圆柱半径
        cylinder_center=None,  # 圆柱中心位置
        name='Cylinder Flow LBM Solver'  # 默认窗口标题
    ):
        self.name = name
        self.nx = nx
        self.ny = ny
        self.Red = Red
        self.cylinder_radius = cylinder_radius
        
        # 设置圆柱中心位置（默认在计算域的1/4处）
        if cylinder_center is None:
            self.cylinder_center = [nx // 4, ny // 2]
        else:
            self.cylinder_center = cylinder_center

        # 宏观物理量
        self.inlet_velocity = inlet_velocity
        self.niu = (self.inlet_velocity * self.cylinder_radius * 2) / self.Red  # 基于圆柱直径
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
        # 生成圆柱边界点
        self.num_boundary_pts, x_cylinder_np, y_cylinder_np = obstacles_generate_cylinder(
            radius=self.cylinder_radius, 
            center_x=self.cylinder_center[0], 
            center_y=self.cylinder_center[1],
            num_points = int(self.cylinder_radius * 2 * 3.141592653589793 /0.5)
        )

        boundary_pos_np = np.stack((x_cylinder_np, y_cylinder_np), axis=1)

        self.boundary_pos = ti.Vector.field(2, float, shape=self.num_boundary_pts)
        self.boundary_pos.from_numpy(boundary_pos_np.astype(np.float32))

        self.boundary_vel = ti.Vector.field(2, float, shape=self.num_boundary_pts)
        self.boundary_vel.fill(0)  # 静止圆柱，期望速度为0

        self.boundary_force = ti.Vector.field(2, float, shape=self.num_boundary_pts)
        self.interp_vel = ti.Vector.field(2, float, shape=self.num_boundary_pts)

        # IBM 参数
        self.ibm_alpha = 2.5  # 反馈系数
        self.it = ti.field(dtype=ti.i32, shape=())
        self.it[None] = 0  # 初始化为0
        
        # 斯特劳哈尔数计算相关
        self.lift_history = deque(maxlen=2000)  # 存储升力历史，最多2000个点
        self.drag_history = deque(maxlen=2000)  # 存储阻力历史
        self.time_history = deque(maxlen=2000)  # 存储时间历史
        self.strouhal_number = 0.0
        self.dominant_frequency = 0.0
        self.inlet_velocity = inlet_velocity  # 入流速度

    @ti.func
    def _get_opposite_direction(self, k):
        """ 获取相反方向的索引 """
        # D2Q9格子的相反方向映射
        # 0->0, 1->3, 2->4, 3->1, 4->2, 5->7, 6->8, 7->5, 8->6
        opposite_map = ti.Vector([0, 3, 4, 1, 2, 7, 8, 5, 6])
        return opposite_map[k]

    @ti.func
    def _phi(self, r):
        """ 一维离散Delta核函数 (辅助函数) """
        r_abs = abs(r)
        res = 0.0
        if r_abs < 2.0:
            res = (1 + ti.cos(r_abs * 3.1415926535 /2 ) ) / 4.0
        else:
            res = 0.0
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
        self.euler_force.fill(0)  # 每一步开始前必须清零力场
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
    def f_eq_with_force(self, i, j):
        """计算带力修正的平衡分布函数 (Dupuis格式)"""
        F = self.euler_force[i, j]
        u = self.vel[i, j]
        rho = self.rho[i, j]
        
        # Dupuis格式：使用修正速度计算平衡分布函数
        # u_corrected = u + F * dt / (2 * rho)
        u_corrected = u  # 默认值
        if rho > 1e-6:
            u_corrected = u + F * self.dt / (2.0 * rho)
        
        uv_sq = u_corrected.dot(u_corrected)
        
        result = ti.Vector.zero(float, 9)
        for k in ti.static(range(9)):
            e_k = tm.vec2(self.e[k, 0], self.e[k, 1])
            eu = e_k.dot(u_corrected)
            result[k] = self.w[k] * rho * (1.0 + eu / self.cs2 + 0.5 * (eu / self.cs2)**2 - 0.5 * uv_sq / self.cs2)
        
        return result

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
        """LBM碰撞步（使用Dupuis力格式）"""
        for i, j in ti.ndrange(self.nx, self.ny):
            # Dupuis格式：直接使用带力修正的平衡分布函数
            feq_ij = self.f_eq_with_force(i, j)
            
            for k in ti.static(range(9)):
                # 标准BGK碰撞，不需要额外的力源项
                self.f_new[i,j][k] = (self.f_old[i,j][k] - 
                                    self.inv_tau * (self.f_old[i,j][k] - feq_ij[k]))

    @ti.kernel
    def streaming(self):
        """ LBM迁移步 (Pull Scheme) - 周期性边界版本 """
        for i, j in ti.ndrange(self.nx, self.ny):
            for k in ti.static(range(9)):
                # 计算源网格点坐标
                ip = i - ti.cast(self.e[k, 0], ti.i32)
                jp = j - ti.cast(self.e[k, 1], ti.i32)

                # 边界处理
                if ip >= 0 and ip < self.nx:
                    # x方向在范围内
                    if jp >= 0 and jp < self.ny:
                        # y方向也在范围内：正常迁移
                        self.f_old[i, j][k] = self.f_new[ip, jp][k]
                    else:
                        # y方向越界：应用周期性边界条件
                        jp_periodic = (jp + self.ny) % self.ny
                        self.f_old[i, j][k] = self.f_new[ip, jp_periodic][k]
                else:
                    # x方向越界：暂时保持原值，将在apply_bc中处理
                    self.f_old[i, j][k] = self.f_new[i, j][k]

    @ti.kernel
    def update_macro_vars(self):
        """ 根据分布函数更新宏观密度和速度 (Dupuis格式) """
        for i, j in ti.ndrange(self.nx, self.ny):
            # 1. 从分布函数计算密度和动量
            new_rho = 0.0
            momentum = tm.vec2(0.0, 0.0)
            for k in ti.static(range(9)):
                f_val = self.f_old[i, j][k]  # 使用迁移后的 f_old
                new_rho += f_val
                e_k_vec = tm.vec2(self.e[k, 0], self.e[k, 1])
                momentum += f_val * e_k_vec
            
            self.rho[i, j] = new_rho
            
            # 2. Dupuis格式：直接从动量计算速度，不需要额外的力修正
            # vel = momentum / rho
            if new_rho > 1e-6:
                self.vel[i, j] = momentum / new_rho
            else:
                self.vel[i, j] = tm.vec2(0.0, 0.0)

    @ti.kernel
    def apply_bc(self):
        """ 施加外围边界条件 - 周期性边界版本 """
        # 左边界：入流条件
        for j in range(self.ny):
            self.vel[0, j] = ti.Vector([self.inlet_velocity, 0.0])
            self.rho[0, j] = 1.0
            self.f_old[0, j] = self.f_eq(0, j)

        # 右边界：出流条件（零梯度）
        for j in range(self.ny):
            self.vel[self.nx - 1, j] = self.vel[self.nx - 2, j]
            self.rho[self.nx - 1, j] = self.rho[self.nx - 2, j]
            self.f_old[self.nx - 1, j] = self.f_eq(self.nx - 1, j)

        # 上下边界：周期性边界条件（已在streaming中处理，这里不需要额外操作）
        # 周期性边界条件意味着上下边界的值会自动从对应的边界复制

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
        # 公式: C_D = F_D / (0.5 * ρ * U² * D)
        # 公式: C_L = F_L / (0.5 * ρ * U² * D)
        # 其中: F_D是阻力, F_L是升力, ρ是密度, U是来流速度, D是圆柱直径
        
        rho_ref = 1.0  # 参考密度
        diameter = 2.0 * self.cylinder_radius  # 圆柱直径
        dynamic_pressure = 0.5 * rho_ref * self.inlet_velocity * self.inlet_velocity * diameter
        
        # 避免除零错误
        cd = 0.0
        cl = 0.0
        if dynamic_pressure > 1e-12:
            cd = drag / dynamic_pressure
            cl = lift / dynamic_pressure

        return tm.vec4(drag, lift, cd, cl)

    def calculate_strouhal_number(self, min_samples=1000):
        """
        计算斯特劳哈尔数
        Args:
            min_samples: 最少需要的样本数量
        Returns:
            strouhal_number: 斯特劳哈尔数
            dominant_frequency: 主频率
        """
        if len(self.lift_history) < min_samples:
            return 0.0, 0.0
        
        # 转换为numpy数组
        lift_data = np.array(self.lift_history)
        time_data = np.array(self.time_history)
        
        # 去除直流分量（平均值）
        lift_data = lift_data - np.mean(lift_data)
        
        # 计算采样频率
        if len(time_data) > 1:
            dt = time_data[1] - time_data[0]
            fs = 1.0 / dt
        else:
            return 0.0, 0.0
        
        # 使用Welch方法计算功率谱密度
        try:
            frequencies, psd = signal.welch(lift_data, fs=fs, nperseg=min(len(lift_data)//4, 256))
            
            # 找到主频率（排除零频率）
            non_zero_idx = frequencies > 0
            if np.any(non_zero_idx):
                psd_non_zero = psd[non_zero_idx]
                freq_non_zero = frequencies[non_zero_idx]
                
                # 找到功率谱密度最大值对应的频率
                max_idx = np.argmax(psd_non_zero)
                dominant_freq = freq_non_zero[max_idx]
                
                # 计算斯特劳哈尔数: St = f * D / U
                # 其中 f 是涡脱落频率，D 是圆柱直径，U 是来流速度
                diameter = 2.0 * self.cylinder_radius
                strouhal = dominant_freq * diameter / self.inlet_velocity
                
                return strouhal, dominant_freq
            else:
                return 0.0, 0.0
                
        except Exception as e:
            print(f"计算斯特劳哈尔数时出错: {e}")
            return 0.0, 0.0

    def update_force_history(self, drag, lift, current_time):
        """更新力的历史记录"""
        self.drag_history.append(drag)
        self.lift_history.append(lift)
        self.time_history.append(current_time)

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
        for step in range(steps):
            self.step()
            self.it[None] += 1
            
            # 计算当前时间
            current_time = self.it[None] * self.dt
            
            if self.it[None] % 10 == 0:  # 每10步记录一次力的历史
                drag_lift_coeffs = self.calculate_drag_lift()
                drag = drag_lift_coeffs.x
                lift = drag_lift_coeffs.y
                self.update_force_history(drag, lift, current_time)
            
            if self.it[None] % 100 == 0:
                drag_lift_coeffs = self.calculate_drag_lift()
                drag = drag_lift_coeffs.x
                lift = drag_lift_coeffs.y
                cd = drag_lift_coeffs.z  # 阻力系数
                cl = drag_lift_coeffs.w  # 升力系数
                
                # 计算斯特劳哈尔数（需要足够的数据点）
                if len(self.lift_history) >= 500:
                    st, freq = self.calculate_strouhal_number(min_samples=500)
                    self.strouhal_number = st
                    self.dominant_frequency = freq
                    print(f"迭代步数: {self.it[None]}, 阻力: {drag:.6f}, 升力: {lift:.6f}, "
                          f"C_D: {cd:.4f}, C_L: {cl:.4f}, St: {st:.4f}, f: {freq:.6f}")
                else:
                    print(f"迭代步数: {self.it[None]}, 阻力: {drag:.6f}, 升力: {lift:.6f}, "
                          f"C_D: {cd:.4f}, C_L: {cl:.4f}, St: 计算中...")
        
        # 最终计算斯特劳哈尔数
        if len(self.lift_history) >= 500:
            final_st, final_freq = self.calculate_strouhal_number()
            final_drag_lift_coeffs = self.calculate_drag_lift()
            final_cd = final_drag_lift_coeffs.z
            final_cl = final_drag_lift_coeffs.w
            
            print(f"\n=== 最终结果 ===")
            print(f"斯特劳哈尔数: {final_st:.4f}")
            print(f"主频率: {final_freq:.6f} Hz")
            print(f"阻力系数 C_D: {final_cd:.4f}")
            print(f"升力系数 C_L: {final_cl:.4f}")
            print(f"圆柱直径: {2*self.cylinder_radius}")
            print(f"入流速度: {self.inlet_velocity}")
            print(f"雷诺数: {self.Red}")
            
            # 理论值对比
            if 40 <= self.Red <= 200:
                theoretical_st = 0.198 * (1 - 19.7/self.Red)  # 经验公式
                # 圆柱绕流的理论阻力系数（经验公式）
                if self.Red > 1:
                    theoretical_cd = 1.0 + 10.0/self.Red  # 简化的经验公式
                else:
                    theoretical_cd = 0.0
                
                print(f"\n=== 理论值对比 ===")
                print(f"理论斯特劳哈尔数: {theoretical_st:.4f}")
                print(f"理论阻力系数: {theoretical_cd:.4f}")
                print(f"St相对误差: {abs(final_st - theoretical_st)/theoretical_st*100:.2f}%")
                print(f"C_D相对误差: {abs(final_cd - theoretical_cd)/theoretical_cd*100:.2f}%")
        else:
            print("数据不足，无法计算可靠的斯特劳哈尔数")

    @ti.kernel
    def compute_vorticity(self, vorticity: ti.template()):
        """计算涡度场 - 适应周期性边界条件"""
        for i, j in ti.ndrange(self.nx, self.ny):
            # 边界处理
            if i == 0 or i == self.nx-1:
                vorticity[i, j] = 0.0  # x方向边界上涡度设为零
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
        """应用涡度颜色映射"""
        for i, j in normalized:
            val = normalized[i, j]
            # 自定义涡度颜色映射：蓝色->绿色->黄色->红色
            if val < 0.25:
                # 蓝色到青色
                t = val * 4.0
                colored[i, j] = ti.Vector([0.0, t, 1.0])
            elif val < 0.5:
                # 青色到绿色
                t = (val - 0.25) * 4.0
                colored[i, j] = ti.Vector([0.0, 1.0, 1.0 - t])
            elif val < 0.75:
                # 绿色到黄色
                t = (val - 0.5) * 4.0
                colored[i, j] = ti.Vector([t, 1.0, 0.0])
            else:
                # 黄色到红色
                t = (val - 0.75) * 4.0
                colored[i, j] = ti.Vector([1.0, 1.0 - t, 0.0])

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
                        self.update_force_history(drag, lift, current_time)
                
                # 每100步输出一次信息
                if self.it[None] % 100 == 0:
                    drag_lift_coeffs = self.calculate_drag_lift()
                    drag = drag_lift_coeffs.x
                    lift = drag_lift_coeffs.y
                    cd = drag_lift_coeffs.z  # 阻力系数
                    cl = drag_lift_coeffs.w  # 升力系数
                    
                    if len(self.lift_history) >= 200:
                        st, freq = self.calculate_strouhal_number(min_samples=200)
                        self.strouhal_number = st
                        self.dominant_frequency = freq
                        print(f"步数: {self.it[None]}, 阻力: {drag:.6f}, 升力: {lift:.6f}, "
                              f"C_D: {cd:.4f}, C_L: {cl:.4f}, St: {st:.4f}, f: {freq:.6f}")
                    else:
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
            if hasattr(self, 'strouhal_number') and self.strouhal_number > 0:
                info_text += f", St: {self.strouhal_number:.4f}"
            
            gui.text(info_text, pos=(0.02, 0.97), color=0xFFFFFF)
            
            # 显示场标签
            gui.text("Vorticity Field", pos=(0.02, 0.52), color=0xFFFFFF)
            gui.text("Velocity Magnitude", pos=(0.02, 0.02), color=0xFFFFFF)
            
            gui.show()


if __name__ == '__main__':
    # 创建圆柱绕流求解器
    cylinder_solver = CylinderFlowSolver(
        nx=800,
        ny=400,
        Red=100,  # 雷诺数
        cylinder_radius=20,  # 圆柱半径
        cylinder_center=[200, 200],  # 圆柱中心位置
    )
    
    # 运行可视化
    cylinder_solver.show()
    
    # # 或者运行求解器
    # cylinder_solver.solver(steps=40000)