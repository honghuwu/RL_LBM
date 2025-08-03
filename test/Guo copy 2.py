"""
圆柱绕流LBM求解器 - 沉浸边界法实现 (版本 2.1)

本程序实现了基于格子玻尔兹曼方法(LBM)和沉浸边界法(IBM)的圆柱绕流数值模拟。
主要功能包括：
1. D2Q9格子玻尔兹曼方法求解不可压缩流动
2. 沉浸边界法处理复杂几何边界（圆柱）
3. 实时计算升力、阻力系数和斯特劳哈尔数
4. 双场可视化显示（涡度场和速度场）
5. 支持周期性边界条件（上下边界和左右边界）

边界条件设置：
- 左边界：入流条件 (固定速度和密度)
- 右边界：出流条件 (零梯度)
- 上下边界：周期性边界条件
- 圆柱表面：无滑移边界条件 (沉浸边界法)

物理模型：
- 雷诺数可调的圆柱绕流
- 卡门涡街现象的捕捉和分析
- 基于Guo格式的力源项处理

技术特点：
- 使用Taichi框架实现GPU加速计算
- 实时可视化和交互式控制
- 高精度的斯特劳哈尔数计算（基于Welch功率谱密度方法）

作者：LBM项目组
版本：2.1 (周期性边界版本)
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

try:
    ti.init(arch=ti.gpu, device_memory_fraction=0.6)
    print("使用GPU后端")
except Exception as e:
    print(f"GPU初始化失败: {e}")
    print("切换到CPU后端")
    ti.init(arch=ti.cpu)

@ti.data_oriented
class CylinderFlowSolver:
    def __init__(
        self,
        nx,
        ny,
        Red,
        inlet_velocity=0.1,  # 入流速度
        cylinder_radius=50,  # 圆柱半径
        cylinder_center=None,  # 圆柱中心位置
        name='Cylinder Flow LBM Solver'  # 默认窗口标题
    ):
        self.name = name
        self.nx = nx
        self.ny = ny
        self.Red = Red
        self.cylinder_radius = cylinder_radius
        self.cylinder_center = cylinder_center if cylinder_center else (nx // 4, ny // 2)
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
        self.cs4 = self.cs2 * self.cs2

        # --- 欧拉场 (流场) 数据结构 ---
        self.rho = ti.field(float, shape=(nx, ny))
        self.vel = ti.Vector.field(2, float, shape=(nx, ny))
        self.f_old = ti.Vector.field(9, float, shape=(nx, ny))
        self.f_new = ti.Vector.field(9, float, shape=(nx, ny))
        # self.f_final = ti.Vector.field(9, float, shape=(nx, ny))
        self.euler_force = ti.Vector.field(2, float, shape=(nx, ny))

        # --- 拉格朗日场 (沉浸边界) 数据结构 ---
        # 生成圆柱边界点
        self.num_boundary_pts, x_cylinder_np, y_cylinder_np = obstacles_generate_cylinder(
            radius=self.cylinder_radius, 
            center_x=self.cylinder_center[0], 
            center_y=self.cylinder_center[1],
            num_points = int(self.cylinder_radius * 2 * 3.141592653589793 )
        )

        boundary_pos_np = np.stack((x_cylinder_np, y_cylinder_np), axis=1)

        self.boundary_pos = ti.Vector.field(2, float, shape=self.num_boundary_pts)
        self.boundary_pos.from_numpy(boundary_pos_np.astype(np.float32))

        self.boundary_vel = ti.Vector.field(2, float, shape=self.num_boundary_pts)
        self.boundary_vel.fill(0)  # 静止圆柱，期望速度为0

        self.boundary_force = ti.Vector.field(2, float, shape=self.num_boundary_pts)
        self.interp_vel = ti.Vector.field(2, float, shape=self.num_boundary_pts)

        self.boundary_rho = ti.field(float, shape=self.num_boundary_pts)
        # IBM 参数
        # self.ibm_alpha = 2.5  # 反馈系数
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
            # force = self.ibm_alpha * (self.boundary_vel[k] - self.interp_vel[k]) / self.dt
            force = 2 * self.boundary_rho[k] * (self.boundary_vel[k] - self.interp_vel[k]) / self.dt
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

    # @ti.kernel
    # def update_f_final(self):
    #     """ 更新f_new，使用Guo格式的力源项 """
    #     for i, j in ti.ndrange(self.nx, self.ny):
    #         F = self.euler_force[i, j]  # 2维物理力
    #         rho_ij = self.rho[i, j]
    #         u_ij = self.vel[i, j]
            
    #         # Guo格式的力源项系数
    #         coeff = 1.0 - 0.5 * self.inv_tau
            
    #         for k in ti.static(range(9)):
    #             e_k = tm.vec2(self.e[k, 0], self.e[k, 1])
    #             eu = e_k.dot(u_ij)
    #             eF = e_k.dot(F)
    #             uF = u_ij.dot(F)
                
    #             # Guo格式的力源项
    #             force_term = coeff * self.w[k] * (
    #                 (eF / self.cs2) + 
    #                 (eu * eF - self.cs2 * uF) / self.cs4
    #             )
                
    #             # 应用力源项
    #             self.f_final[i, j][k] = self.f_new[i, j][k] + force_term


    @ti.kernel
    def collision(self):
        for i, j in ti.ndrange(self.nx, self.ny):
            feq_ij = self.f_eq(i, j)
            vol_f = self.vol_force(i, j)  # 正确调用vol_force函数
            
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
                self.vel[i, j] = momentum / new_rho + self.euler_force[i,j] / ( self.rho[i, j] * 2 )
                # self.vel[i, j] = momentum / new_rho
            else:
                self.vel[i, j] = tm.vec2(0.0, 0.0)

    @ti.kernel
    def apply_bc(self):
        """ 施加外围边界条件 - 周期性边界版本 """
        # 左边界：入流条件
        for j in range(self.ny):
            self.vel[0, j] = ti.Vector([self.inlet_velocity, 0.0])  # 入流速度
            self.rho[0, j] = self.rho[1,j]
            self.f_old[0, j] = self.f_eq(0, j)-self.f_eq(1, j) + self.f_eq(1, j)

        # 右边界：出流条件
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
        rho_ref = 1.0  # 参考密度
        diameter = 2.0 * self.cylinder_radius  # 圆柱直径
        dynamic_pressure = 0.5 * rho_ref * self.inlet_velocity * self.inlet_velocity * diameter
        
        cd = 0.0
        cl = 0.0
        if dynamic_pressure > 1e-12:
            cd = drag / dynamic_pressure
            cl = lift / dynamic_pressure

        return tm.vec4(drag, lift, cd, cl)

    def calculate_strouhal_number(self, min_samples=1000):
        """
        基于FFT的斯特劳哈尔数计算
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
            dt_sample = time_data[1] - time_data[0]
            fs = 1.0 / dt_sample
        else:
            return 0.0, 0.0
        
        # 使用FFT方法计算频谱
        try:
            # 数据预处理：应用汉宁窗减少频谱泄漏
            N = len(lift_data)
            window = np.hanning(N)
            windowed_data = lift_data * window
            
            # 执行FFT变换
            fft_result = np.fft.fft(windowed_data)
            
            # 计算功率谱密度（取模的平方）
            psd = np.abs(fft_result) ** 2
            
            # 生成对应的频率数组
            frequencies = np.fft.fftfreq(N, dt_sample)
            
            # 只取正频率部分（由于对称性）
            positive_freq_mask = frequencies > 0
            frequencies_positive = frequencies[positive_freq_mask]
            psd_positive = psd[positive_freq_mask]
            
            # 设置有效频率范围，避免噪声影响
            # 下限：排除零频率和极低频噪声
            # 上限：限制在合理的涡脱落频率范围内
            freq_min = 0.001  # 最小频率阈值
            freq_max = min(fs/4, 1.0)  # 最大频率阈值（取奈奎斯特频率的1/4或1.0Hz中的较小值）
            
            valid_freq_mask = (frequencies_positive >= freq_min) & (frequencies_positive <= freq_max)
            
            if np.any(valid_freq_mask):
                freq_valid = frequencies_positive[valid_freq_mask]
                psd_valid = psd_positive[valid_freq_mask]
                
                # 找到功率谱密度最大值对应的频率
                max_idx = np.argmax(psd_valid)
                dominant_freq = freq_valid[max_idx]
                
                # 可选：进行峰值精细化（抛物线插值）
                if 0 < max_idx < len(psd_valid) - 1:
                    # 使用抛物线插值提高频率精度
                    y1, y2, y3 = psd_valid[max_idx-1], psd_valid[max_idx], psd_valid[max_idx+1]
                    a = (y1 - 2*y2 + y3) / 2
                    b = (y3 - y1) / 2
                    
                    if a != 0:
                        # 抛物线顶点的偏移量
                        offset = -b / (2*a)
                        # 限制偏移量在合理范围内
                        offset = np.clip(offset, -0.5, 0.5)
                        
                        # 计算精细化后的频率
                        freq_resolution = freq_valid[1] - freq_valid[0] if len(freq_valid) > 1 else 0
                        dominant_freq = dominant_freq + offset * freq_resolution
                
                # 计算斯特劳哈尔数: St = f * D / U
                diameter = 2.0 * self.cylinder_radius
                strouhal = dominant_freq * diameter / self.inlet_velocity
                
                return strouhal, dominant_freq
            else:
                return 0.0, 0.0
                
        except Exception as e:
            print(f"FFT计算斯特劳哈尔数时出错: {e}")
            return 0.0, 0.0

    def update_force_history(self, drag, lift, current_time):
        """更新力的历史记录"""
        self.drag_history.append(drag)
        self.lift_history.append(lift)
        self.time_history.append(current_time)

    def step(self):
        self.interpolate_velocity()
        self.calculate_boundary_force()
        self.spread_force() 
        self.update_macro_vars()
        # self.update_vel()
        self.collision()
        self.streaming()
        
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
            
            # if self.it[None] % 1000 == 0:
            #     print(self.interp_vel)

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
        cylinder_radius=10,  # 圆柱半径
        cylinder_center=[200, 200],  # 圆柱中心位置
        inlet_velocity = 0.2

    )
    

    # cylinder_solver.show()    

    cylinder_solver.solver(steps=80000)