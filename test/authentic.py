import sys
import matplotlib
import numpy as np
from matplotlib import cm
from collections import deque
import matplotlib.pyplot as plt

import taichi as ti
import taichi.math as tm

ti.init(arch=ti.gpu)

@ti.data_oriented
class CylinderFlowLBM:
    def __init__(
        self, 
        nx, 
        ny, 
        dx=1.0, 
        dt=1.0, 
        Red=100.0, 
        bc_type=[0,1,1,1], 
        bc_value=None, 
        cylinder_center=None,
        cylinder_radius=20.0,
        name='LBM Solver - Cylinder Flow'
    ):
        self.name = name
        # 计算域大小
        self.nx = nx
        self.ny = ny
        # 雷诺数
        self.Red = Red  
        # 圆柱参数
        self.cylinder_center = cylinder_center if cylinder_center else [nx//4, ny//2]
        self.cylinder_radius = cylinder_radius
        
        # 边界速度
        U = bc_value[0][0]  # bc_value格式: [[左速度], [上速度], [右速度], [下速度]]
        # 运动粘度
        self.niu = (U * 2.0 * self.cylinder_radius) / self.Red
        # 弛豫时间 来源于LBM
        self.tau = 3.0 * self.niu + 0.5
        # 弛豫时间的倒数
        self.inv_tau = 1.0 / self.tau
        
        # 场变量
        self.rho = ti.field(float, shape=(nx, ny))
        self.vel = ti.Vector.field(2, float, shape=(nx, ny))
        self.mask = ti.field(float, shape=(nx, ny))  # 圆柱障碍物标记
        self.f_old = ti.Vector.field(9, float, shape=(nx, ny))
        self.f_new = ti.Vector.field(9, float, shape=(nx, ny))
        
        # LBM参数
        self.w = ti.types.vector(9, float)(4, 1, 1, 1, 1, 1 / 4, 1 / 4, 1 / 4, 1 / 4) / 9.0
        self.e = ti.types.matrix(9, 2, int)([0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1])
        
        # 边界条件
        self.bc_type = ti.field(int, 4)
        self.bc_type.from_numpy(np.array(bc_type, dtype=np.int32))
        self.bc_value = ti.Vector.field(2, float, shape=4)
        self.bc_value.from_numpy(np.array(bc_value, dtype=np.float32))
        
        # 力系数历史记录
        self.drag_history = deque(maxlen=1000)
        self.lift_history = deque(maxlen=1000)
        self.time_history = deque(maxlen=1000)
        self.step_count = 0
        
        # 斯特劳哈尔数计算相关
        self.strouhal_history = deque(maxlen=100)  # 斯特劳哈尔数历史
        self.lift_for_fft = deque(maxlen=2048)     # 用于FFT分析的升力系数数据
        self.current_strouhal = 0.0                # 当前斯特劳哈尔数
        self.vortex_frequency = 0.0                # 涡脱落频率
        self.dt_physical = dt                      # 物理时间步长
        
        # 生成圆柱mask
        self.generate_cylinder_mask()

    @ti.kernel
    def generate_cylinder_mask(self):
        """生成圆柱障碍物mask"""
        cx, cy = self.cylinder_center[0], self.cylinder_center[1]
        r = self.cylinder_radius
        
        for i, j in ti.ndrange(self.nx, self.ny):
            distance = tm.sqrt((i - cx) ** 2 + (j - cy) ** 2)
            if distance <= r:
                self.mask[i, j] = 1.0
            else:
                self.mask[i, j] = 0.0

    @ti.func
    def f_eq(self, i, j):
        """计算平衡分布函数"""
        eu = self.e @ self.vel[i, j]
        uv = tm.dot(self.vel[i, j], self.vel[i, j])
        return self.w * self.rho[i, j] * (1 + 3 * eu + 4.5 * eu * eu - 1.5 * uv)

    @ti.kernel
    def init(self):
        """初始化场变量"""
        self.vel.fill(0)
        self.rho.fill(1)
        for i, j in self.rho:
            self.f_old[i, j] = self.f_new[i, j] = self.f_eq(i, j)

    @ti.kernel
    def collide_and_stream(self):
        """碰撞和流动步骤"""
        for i, j in ti.ndrange((1, self.nx - 1), (1, self.ny - 1)):
            for k in ti.static(range(9)):
                ip = i - self.e[k, 0]
                jp = j - self.e[k, 1]
                feq = self.f_eq(ip, jp)
                self.f_new[i, j][k] = (1 - self.inv_tau) * self.f_old[ip, jp][k] + feq[k] * self.inv_tau

    @ti.kernel
    def update_macro_var(self):
        """更新宏观变量（密度和速度）"""
        for i, j in ti.ndrange((1, self.nx - 1), (1, self.ny - 1)):
            self.rho[i, j] = 0
            self.vel[i, j] = 0, 0
            for k in ti.static(range(9)):
                self.f_old[i, j][k] = self.f_new[i, j][k]
                self.rho[i, j] += self.f_new[i, j][k]
                self.vel[i, j] += tm.vec2(self.e[k, 0], self.e[k, 1]) * self.f_new[i, j][k]

            self.vel[i, j] /= self.rho[i, j]

    @ti.kernel
    def apply_bc(self):
        """施加边界条件"""
        # 左右边界
        for j in range(1, self.ny - 1):
            # 左边界 (入口)
            self.apply_bc_core(1, 0, 0, j, 1, j)
            # 右边界 (出口)
            self.apply_bc_core(1, 2, self.nx - 1, j, self.nx - 2, j)

        # 上下边界
        for i in range(self.nx):
            self.apply_bc_core(1, 1, i, self.ny - 1, i, self.ny - 2)
            self.apply_bc_core(1, 3, i, 0, i, 1)

        # 圆柱障碍物边界条件处理
        for i, j in ti.ndrange(self.nx, self.ny):
            if self.mask[i, j] == 1:
                self.vel[i, j] = 0, 0  # 固壁边界处速度为零
                # 寻找最近的流体节点作为邻居节点
                inb = i
                jnb = j
                # 简单的邻居节点选择策略
                if i > 0 and self.mask[i-1, j] == 0:
                    inb = i - 1
                elif i < self.nx-1 and self.mask[i+1, j] == 0:
                    inb = i + 1
                elif j > 0 and self.mask[i, j-1] == 0:
                    jnb = j - 1
                elif j < self.ny-1 and self.mask[i, j+1] == 0:
                    jnb = j + 1
                else:
                    # 如果周围都是障碍物，使用默认邻居
                    inb = max(0, min(i-1, self.nx-1))
                    jnb = max(0, min(j-1, self.ny-1))
                
                self.apply_bc_core(0, 0, i, j, inb, jnb)

    @ti.func
    def apply_bc_core(self, outer, dr, ibc, jbc, inb, jnb):
        """边界条件核心函数"""
        # 确保索引在有效范围内
        inb = max(0, min(inb, self.nx-1))
        jnb = max(0, min(jnb, self.ny-1))
        
        if outer == 1:  
            if self.bc_type[dr] == 0:
                self.vel[ibc, jbc] = self.bc_value[dr]
            elif self.bc_type[dr] == 1:
                self.vel[ibc, jbc] = self.vel[inb, jnb]

        self.rho[ibc, jbc] = self.rho[inb, jnb]
        self.f_old[ibc, jbc] = self.f_eq(ibc, jbc) - self.f_eq(inb, jnb) + self.f_old[inb, jnb]

    def calculate_drag_lift(self):
        """计算圆柱的阻力和升力系数"""
        vel_np = self.vel.to_numpy()
        rho_np = self.rho.to_numpy()
        mask_np = self.mask.to_numpy()
        
        # 计算圆柱表面的力
        fx_total = 0.0
        fy_total = 0.0
        
        cx, cy = self.cylinder_center
        r = self.cylinder_radius
        
        # 遍历圆柱边界附近的点
        for i in range(max(0, int(cx - r - 2)), min(self.nx, int(cx + r + 3))):
            for j in range(max(0, int(cy - r - 2)), min(self.ny, int(cy + r + 3))):
                if mask_np[i, j] == 1:  # 圆柱内部点
                    # 检查相邻的流体点
                    for di, dj in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < self.nx and 0 <= nj < self.ny and mask_np[ni, nj] == 0:
                            # 计算压力和粘性力
                            u, v = vel_np[ni, nj]
                            rho = rho_np[ni, nj]
                            
                            # 简化的力计算（基于速度梯度）
                            fx_total += di * (rho - 1.0) + self.niu * u
                            fy_total += dj * (rho - 1.0) + self.niu * v
        
        # 计算无量纲力系数
        U_inlet = self.bc_value[0][0]  # 入口速度
        dynamic_pressure = 0.5 * 1.0 * U_inlet * U_inlet  # 动压
        reference_area = 2.0 * self.cylinder_radius  # 参考面积
        
        if dynamic_pressure > 0 and reference_area > 0:
            cd = fx_total / (dynamic_pressure * reference_area)
            cl = fy_total / (dynamic_pressure * reference_area)
        else:
            cd = cl = 0.0
            
        return cd, cl

    def calculate_strouhal_number(self):
        """计算斯特劳哈尔数"""
        # 需要足够的数据点进行频率分析
        if len(self.lift_for_fft) < 512:
            return 0.0
            
        # 将升力系数数据转换为numpy数组
        lift_data = np.array(list(self.lift_for_fft))
        
        # 去除直流分量（平均值）
        lift_data = lift_data - np.mean(lift_data)
        
        # 应用汉宁窗减少频谱泄漏
        window = np.hanning(len(lift_data))
        lift_data_windowed = lift_data * window
        
        # 进行FFT分析
        fft_result = np.fft.fft(lift_data_windowed)
        frequencies = np.fft.fftfreq(len(lift_data), d=self.dt_physical)
        
        # 只考虑正频率部分
        positive_freq_mask = frequencies > 0
        positive_frequencies = frequencies[positive_freq_mask]
        positive_amplitudes = np.abs(fft_result[positive_freq_mask])
        
        # 找到主频率（最大幅值对应的频率）
        if len(positive_amplitudes) > 0:
            max_amplitude_index = np.argmax(positive_amplitudes)
            dominant_frequency = positive_frequencies[max_amplitude_index]
            
            # 计算斯特劳哈尔数: St = f * D / U
            U_inlet = self.bc_value[0][0]  # 入口速度
            D = 2.0 * self.cylinder_radius  # 圆柱直径
            
            if U_inlet > 0 and D > 0:
                strouhal_number = dominant_frequency * D / U_inlet
                self.vortex_frequency = dominant_frequency
                return strouhal_number
        
        return 0.0

    def analyze_vortex_shedding(self):
        """分析涡脱落特性"""
        if len(self.lift_history) < 100:
            return
            
        # 计算升力系数的统计特性
        recent_lift = list(self.lift_history)[-100:]
        lift_std = np.std(recent_lift)
        lift_amplitude = np.max(recent_lift) - np.min(recent_lift)
        
        # 判断是否存在明显的涡脱落
        if lift_std > 0.01 and lift_amplitude > 0.02:  # 阈值可调
            # 计算斯特劳哈尔数
            st = self.calculate_strouhal_number()
            if st > 0:
                self.current_strouhal = st
                self.strouhal_history.append(st)
        
        return self.current_strouhal

    def step(self):
        """执行一个时间步"""
        self.collide_and_stream()
        self.update_macro_var()
        self.apply_bc()
        
        # 每10步计算一次力系数
        if self.step_count % 10 == 0:
            cd, cl = self.calculate_drag_lift()
            self.drag_history.append(cd)
            self.lift_history.append(cl)
            self.time_history.append(self.step_count)
            
            # 收集升力数据用于FFT分析
            self.lift_for_fft.append(cl)
            
            # 每100步分析一次涡脱落特性
            if self.step_count % 100 == 0 and self.step_count > 1000:
                self.analyze_vortex_shedding()
        
        self.step_count += 1

    def solve(self):
        """主求解循环"""
        gui = ti.GUI(self.name, (self.nx, 2 * self.ny))
        self.init()
        
        print(f"圆柱绕流LBM求解器")
        print(f"计算域: {self.nx} x {self.ny}")
        print(f"雷诺数: {self.Red}")
        print(f"圆柱中心: ({self.cylinder_center[0]}, {self.cylinder_center[1]})")
        print(f"圆柱半径: {self.cylinder_radius}")
        print(f"入口速度: {self.bc_value[0][0]}")
        print("按ESC退出...")
        
        while not gui.get_event(ti.GUI.ESCAPE, ti.GUI.EXIT):
            for _ in range(10):
                self.step()

            # 可视化
            vel = self.vel.to_numpy()
            mask = self.mask.to_numpy()
            
            # 计算涡量
            ugrad = np.gradient(vel[:, :, 0])
            vgrad = np.gradient(vel[:, :, 1])
            vor = ugrad[1] - vgrad[0]
            
            # 计算速度幅值
            vel_mag = (vel[:, :, 0] ** 2.0 + vel[:, :, 1] ** 2.0) ** 0.5
            
            # 在圆柱内部设置特殊值
            vor[mask == 1] = 0
            vel_mag[mask == 1] = 0
            
            # 颜色映射
            colors = [
                (1, 1, 0),
                (0.953, 0.490, 0.016),
                (0, 0, 0),
                (0.176, 0.976, 0.529),
                (0, 1, 1),
            ]
            my_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("my_cmap", colors)
            vor_img = cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=-0.02, vmax=0.02), cmap=my_cmap).to_rgba(vor)
            vel_img = cm.plasma(vel_mag / 0.15)
            
            # 合并图像
            img = np.concatenate((vor_img, vel_img), axis=1)
            gui.set_image(img)
            
            # 显示当前信息
            if len(self.drag_history) > 0:
                gui.text(f'Step: {self.step_count}', pos=(0.02, 0.95), color=0xFFFFFF)
                gui.text(f'Cd: {self.drag_history[-1]:.4f}', pos=(0.02, 0.90), color=0xFFFFFF)
                gui.text(f'Cl: {self.lift_history[-1]:.4f}', pos=(0.02, 0.85), color=0xFFFFFF)
                gui.text(f'Re: {self.Red}', pos=(0.02, 0.80), color=0xFFFFFF)
                
                # 显示斯特劳哈尔数信息
                if self.current_strouhal > 0:
                    gui.text(f'St: {self.current_strouhal:.4f}', pos=(0.02, 0.75), color=0x00FF00)
                    gui.text(f'f: {self.vortex_frequency:.6f}', pos=(0.02, 0.70), color=0x00FF00)
                else:
                    gui.text('St: 计算中...', pos=(0.02, 0.75), color=0xFFFF00)
            
            gui.show()

    def plot_force_coefficients(self):
        """绘制力系数历史曲线和斯特劳哈尔数分析"""
        if len(self.drag_history) < 10:
            print("数据不足，无法绘制曲线")
            return
            
        plt.figure(figsize=(15, 12))
        
        # 阻力系数
        plt.subplot(3, 2, 1)
        plt.plot(self.time_history, self.drag_history, 'b-', linewidth=1.5)
        plt.title('阻力系数 Cd 随时间变化')
        plt.xlabel('时间步')
        plt.ylabel('Cd')
        plt.grid(True, alpha=0.3)
        
        # 升力系数
        plt.subplot(3, 2, 2)
        plt.plot(self.time_history, self.lift_history, 'r-', linewidth=1.5)
        plt.title('升力系数 Cl 随时间变化')
        plt.xlabel('时间步')
        plt.ylabel('Cl')
        plt.grid(True, alpha=0.3)
        
        # 斯特劳哈尔数历史
        if len(self.strouhal_history) > 0:
            plt.subplot(3, 2, 3)
            st_steps = np.arange(len(self.strouhal_history)) * 100 + 1000
            plt.plot(st_steps, self.strouhal_history, 'g-', linewidth=2, marker='o', markersize=4)
            plt.title('斯特劳哈尔数 St 随时间变化')
            plt.xlabel('时间步')
            plt.ylabel('St')
            plt.grid(True, alpha=0.3)
            
            # 添加理论值参考线（Re=120时的经验值约为0.2）
            if self.Red > 40:
                theoretical_st = 0.198 * (1 - 19.7/self.Red)  # 经验公式
                plt.axhline(y=theoretical_st, color='orange', linestyle='--', 
                           label=f'理论值 ≈ {theoretical_st:.3f}')
                plt.legend()
        
        # 升力系数频谱分析
        if len(self.lift_for_fft) >= 512:
            plt.subplot(3, 2, 4)
            lift_data = np.array(list(self.lift_for_fft))
            lift_data = lift_data - np.mean(lift_data)
            
            # FFT分析
            fft_result = np.fft.fft(lift_data * np.hanning(len(lift_data)))
            frequencies = np.fft.fftfreq(len(lift_data), d=self.dt_physical)
            
            # 只显示正频率部分
            positive_freq_mask = (frequencies > 0) & (frequencies < 0.5)
            positive_frequencies = frequencies[positive_freq_mask]
            positive_amplitudes = np.abs(fft_result[positive_freq_mask])
            
            plt.semilogy(positive_frequencies, positive_amplitudes, 'purple', linewidth=1.5)
            plt.title('升力系数频谱分析')
            plt.xlabel('频率')
            plt.ylabel('幅值 (对数)')
            plt.grid(True, alpha=0.3)
            
            # 标记主频率
            if len(positive_amplitudes) > 0:
                max_idx = np.argmax(positive_amplitudes)
                dominant_freq = positive_frequencies[max_idx]
                plt.axvline(x=dominant_freq, color='red', linestyle='--', 
                           label=f'主频率: {dominant_freq:.6f}')
                plt.legend()
        
        # 相位图（Cl vs Cd）
        if len(self.drag_history) > 100:
            plt.subplot(3, 2, 5)
            recent_cd = self.drag_history[-min(500, len(self.drag_history)):]
            recent_cl = self.lift_history[-min(500, len(self.lift_history)):]
            plt.plot(recent_cd, recent_cl, 'b-', alpha=0.7, linewidth=1)
            plt.scatter(recent_cd[-1], recent_cl[-1], color='red', s=50, zorder=5)
            plt.title('升力-阻力相位图')
            plt.xlabel('Cd')
            plt.ylabel('Cl')
            plt.grid(True, alpha=0.3)
        
        # 统计信息文本
        plt.subplot(3, 2, 6)
        plt.axis('off')
        
        # 计算统计信息
        if len(self.drag_history) > 100:
            cd_mean = np.mean(self.drag_history[-100:])
            cl_mean = np.mean(self.lift_history[-100:])
            cl_std = np.std(self.lift_history[-100:])
            cl_amplitude = np.max(self.lift_history[-100:]) - np.min(self.lift_history[-100:])
            
            stats_text = f"""
            === 流动特性统计 (最后100步) ===

            雷诺数 Re: {self.Red}
            圆柱直径 D: {2*self.cylinder_radius:.1f}
            入口速度 U: {self.bc_value[0][0]:.3f}

            平均阻力系数 Cd: {cd_mean:.4f}
            平均升力系数 Cl: {cl_mean:.4f}
            升力系数标准差: {cl_std:.4f}
            升力系数幅值: {cl_amplitude:.4f}

            当前斯特劳哈尔数 St: {self.current_strouhal:.4f}
            涡脱落频率 f: {self.vortex_frequency:.6f}
            """
            
            # 添加理论比较
            if self.Red > 40:
                theoretical_st = 0.198 * (1 - 19.7/self.Red)
                error_percent = abs(self.current_strouhal - theoretical_st) / theoretical_st * 100 if theoretical_st > 0 else 0
                stats_text += f"\n理论斯特劳哈尔数: {theoretical_st:.4f}"
                stats_text += f"\n相对误差: {error_percent:.1f}%"
            
            plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, 
                    fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        plt.show()
        
        # 打印详细统计信息
        if len(self.drag_history) > 100:
            print(f"\n=== 圆柱绕流分析结果 ===")
            print(f"雷诺数: {self.Red}")
            print(f"平均阻力系数 Cd: {cd_mean:.4f}")
            print(f"平均升力系数 Cl: {cl_mean:.4f}")
            print(f"升力系数标准差: {cl_std:.4f}")
            if self.current_strouhal > 0:
                print(f"斯特劳哈尔数 St: {self.current_strouhal:.4f}")
                print(f"涡脱落频率: {self.vortex_frequency:.6f}")
                
                # 与理论值比较
                if self.Red > 40:
                    theoretical_st = 0.198 * (1 - 19.7/self.Red)
                    print(f"理论斯特劳哈尔数: {theoretical_st:.4f}")
                    print(f"相对误差: {abs(self.current_strouhal - theoretical_st) / theoretical_st * 100:.1f}%")
            else:
                print("斯特劳哈尔数: 未检测到明显涡脱落")


if __name__ == '__main__':
    # 圆柱绕流测试案例
    cylinder_lbm = CylinderFlowLBM(
        nx=800,          # 计算域宽度
        ny=400,          # 计算域高度
        bc_type=[0,1,1,1],  # 边界条件类型: [左,上,右,下] 0=Dirichlet, 1=Neumann
        bc_value=[[0.08, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],  # 入口速度U=0.08
        Red=100,         # 雷诺数
        cylinder_center=[200, 200],  # 圆柱中心位置
        cylinder_radius=20,  # 圆柱半径
    )
    
    try:
        cylinder_lbm.solve()
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    finally:
        # 绘制力系数曲线
        cylinder_lbm.plot_force_coefficients()