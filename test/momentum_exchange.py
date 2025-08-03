import taichi as ti
import taichi.math as tm
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

ti.init(arch=ti.gpu)

@ti.data_oriented
class CylinderFlowSolverMomentumExchange:
    def __init__(self, nx=800, ny=400, Red=100, cylinder_radius=10, cylinder_center=[200, 200]):
        # 网格参数
        self.nx = nx
        self.ny = ny
        self.name = "圆柱绕流 - 动量交换法"
        
        # 物理参数
        self.Red = Red  # 雷诺数
        self.cylinder_radius = cylinder_radius
        self.cylinder_center = cylinder_center
        
        # 计算物理参数
        self.inlet_velocity = 0.1  # 入流速度
        self.nu = self.inlet_velocity * (2 * cylinder_radius) / Red  # 运动粘度
        self.tau = 3.0 * self.nu + 0.5  # 松弛时间
        self.inv_tau = 1.0 / self.tau
        self.dt = 1.0  # 时间步长
        
        # LBM参数
        self.e = ti.Vector.field(2, dtype=ti.f32, shape=9)
        self.w = ti.field(dtype=ti.f32, shape=9)
        
        # 分布函数
        self.f_old = ti.Vector.field(9, dtype=ti.f32, shape=(nx, ny))
        self.f_new = ti.Vector.field(9, dtype=ti.f32, shape=(nx, ny))
        
        # 宏观量
        self.rho = ti.field(dtype=ti.f32, shape=(nx, ny))
        self.vel = ti.Vector.field(2, dtype=ti.f32, shape=(nx, ny))
        
        # 边界相关
        self.is_boundary = ti.field(dtype=ti.i32, shape=(nx, ny))
        self.boundary_pos = ti.Vector.field(2, dtype=ti.i32, shape=10000)
        self.boundary_count = ti.field(dtype=ti.i32, shape=())
        
        # 动量交换法相关数据结构
        self.boundary_info = ti.Vector.field(4, dtype=ti.i32, shape=10000)  # [i, j, q, qb]
        self.boundary_normal = ti.Vector.field(2, dtype=ti.f32, shape=10000)  # 边界法向量
        
        # 力的历史记录
        self.drag_history = []
        self.lift_history = []
        self.time_history = []
        self.strouhal_number = 0.0
        self.dominant_frequency = 0.0
        
        # 时间步计数器
        self.it = ti.field(dtype=ti.i32, shape=())
        
        # 初始化LBM参数
        self.init_lbm_params()

    @ti.kernel
    def init_lbm_params(self):
        """ 初始化LBM参数 """
        # D2Q9格子速度
        e_vals = [
            [0, 0], [1, 0], [0, 1], [-1, 0], [0, -1],
            [1, 1], [-1, 1], [-1, -1], [1, -1]
        ]
        for i in ti.static(range(9)):
            self.e[i] = ti.Vector(e_vals[i])
        
        # 权重系数
        w_vals = [4.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0,
                  1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0]
        for i in ti.static(range(9)):
            self.w[i] = w_vals[i]

    @ti.func
    def f_eq(self, i, j):
        """ 计算平衡分布函数 """
        rho_ij = self.rho[i, j]
        vel_ij = self.vel[i, j]
        
        feq = ti.Vector([0.0] * 9)
        for k in ti.static(range(9)):
            eu = self.e[k].dot(vel_ij)
            uv = vel_ij.dot(vel_ij)
            feq[k] = self.w[k] * rho_ij * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * uv)
        
        return feq

    @ti.kernel
    def init_boundary(self):
        """ 初始化边界条件 """
        count = 0
        
        for i, j in ti.ndrange(self.nx, self.ny):
            # 计算到圆柱中心的距离
            dx = i - self.cylinder_center[0]
            dy = j - self.cylinder_center[1]
            dist = ti.sqrt(dx * dx + dy * dy)
            
            if dist <= self.cylinder_radius:
                self.is_boundary[i, j] = 1
                if count < 10000:
                    self.boundary_pos[count] = ti.Vector([i, j])
                    count += 1
            else:
                self.is_boundary[i, j] = 0
        
        self.boundary_count[None] = count

    @ti.kernel
    def init_boundary_info(self):
        """ 初始化边界信息，用于动量交换法 """
        count = 0
        
        for i, j in ti.ndrange(self.nx, self.ny):
            if self.is_boundary[i, j] == 1:
                # 对于边界点，找到指向流体的方向
                for k in ti.static(range(1, 9)):  # 跳过静止方向
                    ni = i + ti.cast(self.e[k][0], ti.i32)
                    nj = j + ti.cast(self.e[k][1], ti.i32)
                    
                    # 检查邻居是否在网格内且为流体
                    if 0 <= ni < self.nx and 0 <= nj < self.ny:
                        if self.is_boundary[ni, nj] == 0:  # 邻居是流体
                            if count < 10000:
                                # 存储边界信息：[i, j, q, qb]
                                # q: 从边界指向流体的方向
                                # qb: 反向（从流体指向边界）
                                qb = k
                                q = 0
                                if k == 1: q = 3
                                elif k == 2: q = 4
                                elif k == 3: q = 1
                                elif k == 4: q = 2
                                elif k == 5: q = 7
                                elif k == 6: q = 8
                                elif k == 7: q = 5
                                elif k == 8: q = 6
                                
                                self.boundary_info[count] = ti.Vector([i, j, q, qb])
                                
                                # 计算边界法向量（指向流体）
                                dx = float(self.e[k][0])
                                dy = float(self.e[k][1])
                                norm = ti.sqrt(dx*dx + dy*dy)
                                if norm > 0:
                                    self.boundary_normal[count] = ti.Vector([dx/norm, dy/norm])
                                else:
                                    self.boundary_normal[count] = ti.Vector([0.0, 0.0])
                                
                                count += 1

    @ti.kernel
    def init_fields(self):
        """ 初始化流场 """
        for i, j in ti.ndrange(self.nx, self.ny):
            if self.is_boundary[i, j] == 0:
                self.rho[i, j] = 1.0
                self.vel[i, j] = ti.Vector([self.inlet_velocity, 0.0])
            else:
                self.rho[i, j] = 1.0
                self.vel[i, j] = ti.Vector([0.0, 0.0])
        
        # 根据初始化的宏观量，计算初始的平衡分布函数
        for i, j in self.rho:
            self.f_old[i, j] = self.f_eq(i, j)
            self.f_new[i, j] = self.f_eq(i, j)

    @ti.kernel
    def collision(self):
        """ LBM碰撞步 """
        for i, j in ti.ndrange(self.nx, self.ny):
            if self.is_boundary[i, j] == 0:  # 只在流体区域进行碰撞
                feq_ij = self.f_eq(i, j)
                
                for k in ti.static(range(9)):
                    self.f_new[i,j][k] = (self.f_old[i,j][k] - 
                                        self.inv_tau * (self.f_old[i,j][k] - feq_ij[k]))

    @ti.kernel
    def streaming(self):
        """ LBM迁移步 """
        for i, j in ti.ndrange(self.nx, self.ny):
            if self.is_boundary[i, j] == 0:  # 只在流体区域进行迁移
                for k in ti.static(range(9)):
                    # 计算源网格点坐标
                    ip = i - ti.cast(self.e[k][0], ti.i32)
                    jp = j - ti.cast(self.e[k][1], ti.i32)

                    # 周期性边界
                    if ip < 0: ip = self.nx - 1
                    if ip > self.nx - 1: ip = 0
                    if jp < 0: jp = self.ny - 1
                    if jp > self.ny - 1: jp = 0
                    
                    # 从源网格点拉取数据
                    self.f_old[i, j][k] = self.f_new[ip, jp][k]

    @ti.kernel
    def bounce_back(self):
        """ 边界反弹 """
        for i, j in ti.ndrange(self.nx, self.ny):
            if self.is_boundary[i, j] == 1:
                # 对边界点进行反弹
                for k in ti.static(range(1, 9)):
                    # 找到反向
                    kb = 0
                    if k == 1: kb = 3
                    elif k == 2: kb = 4
                    elif k == 3: kb = 1
                    elif k == 4: kb = 2
                    elif k == 5: kb = 7
                    elif k == 6: kb = 8
                    elif k == 7: kb = 5
                    elif k == 8: kb = 6
                    
                    # 反弹：f_k(x_b, t+1) = f_kb(x_b, t)
                    self.f_old[i, j][k] = self.f_new[i, j][kb]

    @ti.kernel
    def update_macro_vars(self):
        """ 更新宏观量 """
        for i, j in ti.ndrange(self.nx, self.ny):
            if self.is_boundary[i, j] == 0:  # 只在流体区域更新
                new_rho = 0.0
                momentum = tm.vec2(0.0, 0.0)
                
                for k in ti.static(range(9)):
                    f_val = self.f_old[i, j][k]
                    new_rho += f_val
                    e_k_vec = tm.vec2(self.e[k][0], self.e[k][1])
                    momentum += f_val * e_k_vec
                
                self.rho[i, j] = new_rho
                
                if new_rho > 1e-6:
                    self.vel[i, j] = momentum / new_rho
                else:
                    self.vel[i, j] = tm.vec2(0.0, 0.0)

    @ti.kernel
    def apply_bc(self):
        """ 施加边界条件 """
        # 左边界：入流条件
        for j in range(self.ny):
            if self.is_boundary[0, j] == 0:
                self.vel[0, j] = ti.Vector([self.inlet_velocity, 0.0])
                self.rho[0, j] = 1.0
                self.f_old[0, j] = self.f_eq(0, j)

        # 右边界：出流条件
        for j in range(self.ny):
            if self.is_boundary[self.nx - 1, j] == 0:
                self.vel[self.nx - 1, j] = self.vel[self.nx - 2, j]
                self.rho[self.nx - 1, j] = self.rho[self.nx - 2, j]
                self.f_old[self.nx - 1, j] = self.f_eq(self.nx - 1, j)

        # 上下边界：周期性边界条件
        for i in range(self.nx):
            if self.is_boundary[i, self.ny - 1] == 0:
                self.vel[i, self.ny - 1] = self.vel[i, 1]
                self.rho[i, self.ny - 1] = self.rho[i, 1]
                self.f_old[i, self.ny - 1] = self.f_old[i, 1]

            if self.is_boundary[i, 0] == 0:
                self.vel[i, 0] = self.vel[i, self.ny - 2]
                self.rho[i, 0] = self.rho[i, self.ny - 2]
                self.f_old[i, 0] = self.f_old[i, self.ny - 2]

    @ti.kernel
    def calculate_drag_lift_momentum_exchange(self) -> tm.vec4:
        """ 使用动量交换原理计算升力和阻力 """
        fx = 0.0
        fy = 0.0
        
        # 遍历所有边界信息
        for k in range(self.boundary_count[None]):
            if k < 10000:  # 安全检查
                info = self.boundary_info[k]
                i = info[0]
                j = info[1]
                q = info[2]   # 从边界指向流体的方向
                qb = info[3]  # 从流体指向边界的方向
                
                # 检查边界
                if 0 <= i < self.nx and 0 <= j < self.ny:
                    if self.is_boundary[i, j] == 1:
                        # 动量交换：g0 = f_in + f_out
                        # f_in: 碰撞前从流体进入边界的分布函数
                        # f_out: 碰撞后从边界反弹回流体的分布函数
                        f_in = self.f_new[i, j][qb]   # 碰撞前
                        f_out = self.f_old[i, j][q]   # 碰撞后（反弹）
                        
                        g0 = f_in + f_out
                        
                        # 计算动量交换
                        cx = float(self.e[q][0])
                        cy = float(self.e[q][1])
                        
                        fx += g0 * cx
                        fy += g0 * cy
        
        # 力的方向修正（负号表示流体对固体的作用力）
        drag = -fx
        lift = -fy
        
        # 计算无量纲系数
        rho_ref = 1.0
        diameter = 2.0 * self.cylinder_radius
        dynamic_pressure = 0.5 * rho_ref * self.inlet_velocity * self.inlet_velocity * diameter
        
        cd = 0.0
        cl = 0.0
        if dynamic_pressure > 1e-12:
            cd = drag / dynamic_pressure
            cl = lift / dynamic_pressure

        return tm.vec4(drag, lift, cd, cl)

    def calculate_strouhal_number(self, min_samples=1000):
        """ 计算斯特劳哈尔数 """
        if len(self.lift_history) < min_samples:
            return 0.0, 0.0
        
        lift_data = np.array(self.lift_history)
        time_data = np.array(self.time_history)
        
        lift_data = lift_data - np.mean(lift_data)
        
        if len(time_data) > 1:
            dt = time_data[1] - time_data[0]
            fs = 1.0 / dt
        else:
            return 0.0, 0.0
        
        try:
            frequencies, psd = signal.welch(lift_data, fs=fs, nperseg=min(len(lift_data)//4, 256))
            
            non_zero_idx = frequencies > 0
            if np.any(non_zero_idx):
                psd_non_zero = psd[non_zero_idx]
                freq_non_zero = frequencies[non_zero_idx]
                
                max_idx = np.argmax(psd_non_zero)
                dominant_freq = freq_non_zero[max_idx]
                
                diameter = 2.0 * self.cylinder_radius
                strouhal = dominant_freq * diameter / self.inlet_velocity
                
                return strouhal, dominant_freq
            else:
                return 0.0, 0.0
                
        except Exception as e:
            print(f"计算斯特劳哈尔数时出错: {e}")
            return 0.0, 0.0

    def update_force_history(self, drag, lift, current_time):
        """ 更新力的历史记录 """
        self.drag_history.append(drag)
        self.lift_history.append(lift)
        self.time_history.append(current_time)

    def init(self):
        """ 初始化求解器 """
        self.init_boundary()
        self.init_boundary_info()
        self.init_fields()
        self.it[None] = 0

    def step(self):
        """ 执行一个时间步 """
        # 1. 碰撞
        self.collision()
        
        # 2. 迁移
        self.streaming()
        
        # 3. 边界反弹
        self.bounce_back()
        
        # 4. 更新宏观量
        self.update_macro_vars()
        
        # 5. 施加边界条件
        self.apply_bc()

    @ti.kernel
    def compute_vorticity(self, vorticity: ti.template()):
        """ 计算涡度 """
        for i, j in ti.ndrange((1, self.nx-1), (1, self.ny-1)):
            if self.is_boundary[i, j] == 0:
                # 使用中心差分计算涡度
                dvx_dy = (self.vel[i, j+1][0] - self.vel[i, j-1][0]) / 2.0
                dvy_dx = (self.vel[i+1, j][1] - self.vel[i-1, j][1]) / 2.0
                vorticity[i, j] = dvy_dx - dvx_dy
            else:
                vorticity[i, j] = 0.0

    @ti.kernel
    def compute_velocity_magnitude(self, vel_mag: ti.template()):
        """ 计算速度大小 """
        for i, j in ti.ndrange(self.nx, self.ny):
            if self.is_boundary[i, j] == 0:
                vel_mag[i, j] = ti.sqrt(self.vel[i, j].dot(self.vel[i, j]))
            else:
                vel_mag[i, j] = 0.0

    @ti.kernel
    def normalize_field(self, field: ti.template(), normalized: ti.template(), 
                       min_val: float, max_val: float):
        """ 归一化场到[0,1]范围 """
        for i, j in field:
            val = (field[i, j] - min_val) / (max_val - min_val)
            normalized[i, j] = ti.max(0.0, ti.min(1.0, val))

    @ti.kernel
    def apply_colormap_vorticity(self, normalized: ti.template(), colored: ti.template()):
        """ 应用涡度颜色映射（蓝-白-红） """
        for i, j in normalized:
            val = normalized[i, j]
            if val < 0.25:
                # 深蓝到蓝色
                t = val * 4.0
                colored[i, j] = ti.Vector([0.0, 0.0, 0.5 + t * 0.5])
            elif val < 0.5:
                # 蓝色到白色
                t = (val - 0.25) * 4.0
                colored[i, j] = ti.Vector([t, t, 1.0])
            elif val < 0.75:
                # 白色到红色
                t = (val - 0.5) * 4.0
                colored[i, j] = ti.Vector([1.0, 1.0 - t, 1.0 - t])
            else:
                # 红色到深红
                t = (val - 0.75) * 4.0
                colored[i, j] = ti.Vector([1.0, 0.0, 0.0])

    @ti.kernel
    def apply_colormap_velocity(self, normalized: ti.template(), colored: ti.template()):
        """ 应用速度颜色映射 """
        for i, j in normalized:
            val = ti.min(normalized[i, j], 1.0)
            
            if val < 0.16667:
                t = val * 6.0
                colored[i, j] = ti.Vector([0.0, t * 0.5, 0.5 + t * 0.5])
            elif val < 0.33333:
                t = (val - 0.16667) * 6.0
                colored[i, j] = ti.Vector([0.0, 0.5 + t * 0.5, 1.0 - t * 0.5])
            elif val < 0.5:
                t = (val - 0.33333) * 6.0
                colored[i, j] = ti.Vector([t * 0.5, 1.0, 0.5 - t * 0.5])
            elif val < 0.66667:
                t = (val - 0.5) * 6.0
                colored[i, j] = ti.Vector([0.5 + t * 0.5, 1.0, 0.0])
            elif val < 0.83333:
                t = (val - 0.66667) * 6.0
                colored[i, j] = ti.Vector([1.0, 1.0 - t * 0.3, 0.0])
            else:
                t = (val - 0.83333) * 6.0
                colored[i, j] = ti.Vector([1.0, 0.7 - t * 0.7, t * 0.2])

    @ti.kernel
    def combine_images_vertical(self, vort_img: ti.template(), vel_img: ti.template(), 
                               combined: ti.template(), separator_height: int):
        """ 垂直组合两个图像 """
        nx, ny = vort_img.shape[0], vort_img.shape[1]
        
        for i, j in ti.ndrange(nx, ny):
            combined[i, j] = vort_img[i, j]
        
        for i, j in ti.ndrange(nx, separator_height):
            combined[i, ny + j] = ti.Vector([0.5, 0.5, 0.5])
        
        for i, j in ti.ndrange(nx, ny):
            combined[i, ny + separator_height + j] = vel_img[i, j]

    def show(self):
        """ 显示实时流场可视化 """
        self.init()
        
        vorticity = ti.field(float, shape=(self.nx, self.ny))
        vel_mag = ti.field(float, shape=(self.nx, self.ny))
        
        vorticity_norm = ti.field(float, shape=(self.nx, self.ny))
        vel_mag_norm = ti.field(float, shape=(self.nx, self.ny))
        
        vorticity_colored = ti.Vector.field(3, float, shape=(self.nx, self.ny))
        vel_mag_colored = ti.Vector.field(3, float, shape=(self.nx, self.ny))
        
        separator_height = 10
        combined_height = self.ny * 2 + separator_height
        combined_img = ti.Vector.field(3, float, shape=(self.nx, combined_height))
        
        gui = ti.GUI(self.name, res=(self.nx, combined_height))
        
        print("=== 动量交换法圆柱绕流 ===")
        print("显示布局:")
        print("  上半部分 - 涡度场")
        print("  下半部分 - 速度场")
        print("按键控制:")
        print("  'q' 或 ESC - 退出")
        print("  空格 - 暂停/继续")
        print("==============================")
        
        paused = False
        
        while gui.running:
            for e in gui.get_events(ti.GUI.PRESS):
                if e.key == ti.GUI.ESCAPE or e.key == 'q':
                    gui.running = False
                elif e.key == ti.GUI.SPACE:
                    paused = not paused
                    print("暂停" if paused else "继续")
            
            if not paused:
                for _ in range(5):
                    self.step()
                    self.it[None] += 1
                    
                    current_time = self.it[None] * self.dt
                    if self.it[None] % 10 == 0:
                        drag_lift_coeffs = self.calculate_drag_lift_momentum_exchange()
                        drag = drag_lift_coeffs.x
                        lift = drag_lift_coeffs.y
                        self.update_force_history(drag, lift, current_time)
                
                if self.it[None] % 100 == 0:
                    drag_lift_coeffs = self.calculate_drag_lift_momentum_exchange()
                    drag = drag_lift_coeffs.x
                    lift = drag_lift_coeffs.y
                    cd = drag_lift_coeffs.z
                    cl = drag_lift_coeffs.w
                    
                    if len(self.lift_history) >= 200:
                        st, freq = self.calculate_strouhal_number(min_samples=200)
                        self.strouhal_number = st
                        self.dominant_frequency = freq
                        print(f"步数: {self.it[None]}, 阻力: {drag:.6f}, 升力: {lift:.6f}, "
                              f"C_D: {cd:.4f}, C_L: {cl:.4f}, St: {st:.4f}, f: {freq:.6f} [动量交换法]")
                    else:
                        print(f"步数: {self.it[None]}, 阻力: {drag:.6f}, 升力: {lift:.6f}, "
                              f"C_D: {cd:.4f}, C_L: {cl:.4f} [动量交换法]")
            
            self.compute_vorticity(vorticity)
            self.compute_velocity_magnitude(vel_mag)
            
            self.normalize_field(vorticity, vorticity_norm, -0.02, 0.02)
            self.normalize_field(vel_mag, vel_mag_norm, 0.0, 0.4)
            
            self.apply_colormap_vorticity(vorticity_norm, vorticity_colored)
            self.apply_colormap_velocity(vel_mag_norm, vel_mag_colored)
            
            self.combine_images_vertical(vorticity_colored, vel_mag_colored, 
                                       combined_img, separator_height)
            
            gui.set_image(combined_img.to_numpy())
            
            boundary_pts = self.boundary_pos.to_numpy()
            
            for pt in boundary_pts:
                if pt[0] > 0 and pt[1] > 0:  # 有效边界点
                    x = pt[0] / self.nx
                    y = pt[1] / combined_height
                    gui.circle((x, y), radius=1, color=0x000000)
            
            for pt in boundary_pts:
                if pt[0] > 0 and pt[1] > 0:
                    x = pt[0] / self.nx
                    y = (pt[1] + self.ny + separator_height) / combined_height
                    gui.circle((x, y), radius=1, color=0x000000)
            
            drag_lift_coeffs = self.calculate_drag_lift_momentum_exchange()
            drag = drag_lift_coeffs.x
            lift = drag_lift_coeffs.y
            cd = drag_lift_coeffs.z
            cl = drag_lift_coeffs.w
            
            info_text = f"Step: {self.it[None]}, Re: {self.Red}, C_D: {cd:.3f}, C_L: {cl:.3f} [动量交换法]"
            if hasattr(self, 'strouhal_number') and self.strouhal_number > 0:
                info_text += f", St: {self.strouhal_number:.4f}"
            
            gui.text(info_text, pos=(0.02, 0.97), color=0xFFFFFF)
            gui.text("Vorticity Field", pos=(0.02, 0.52), color=0xFFFFFF)
            gui.text("Velocity Magnitude", pos=(0.02, 0.02), color=0xFFFFFF)
            
            gui.show()

    def solver(self, steps=10000):
        """ 运行指定步数的LBM求解 """
        self.init()
        for step in range(steps):
            self.step()
            self.it[None] += 1
            
            current_time = self.it[None] * self.dt
            
            if self.it[None] % 10 == 0:
                drag_lift_coeffs = self.calculate_drag_lift_momentum_exchange()
                drag = drag_lift_coeffs.x
                lift = drag_lift_coeffs.y
                self.update_force_history(drag, lift, current_time)
            
            if self.it[None] % 1000 == 0:
                drag_lift_coeffs = self.calculate_drag_lift_momentum_exchange()
                drag = drag_lift_coeffs.x
                lift = drag_lift_coeffs.y
                cd = drag_lift_coeffs.z
                cl = drag_lift_coeffs.w
                
                if len(self.lift_history) >= 200:
                    st, freq = self.calculate_strouhal_number(min_samples=200)
                    print(f"步数: {self.it[None]}, C_D: {cd:.4f}, C_L: {cl:.4f}, St: {st:.4f} [动量交换法]")
                else:
                    print(f"步数: {self.it[None]}, C_D: {cd:.4f}, C_L: {cl:.4f} [动量交换法]")

if __name__ == '__main__':
    # 创建基于动量交换法的圆柱绕流求解器
    cylinder_solver = CylinderFlowSolverMomentumExchange(
        nx=800,
        ny=400,
        Red=100,
        cylinder_radius=20,
        cylinder_center=[200, 200],
    )
    
    # 运行可视化
    cylinder_solver.show()
    
    # 或者运行求解器
    # cylinder_solver.solver(steps=40000)