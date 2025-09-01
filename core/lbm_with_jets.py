import sys
import os
import numpy as np
import taichi as ti
import taichi.math as tm

from tools.obstacles_generate.naca_genarate import obstacles_generate
from tools.show_tools import LBMVisualizer
from tools.output_tool.output import create_output_processor



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
        jet_params=None,  # 新增：射流参数
        name='LBM Solver with Jet'  # 窗口标题
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

        # IBM 参数
        self.it = ti.field(dtype=ti.i32, shape=())
        self.it[None] = 0  # 初始化为0
        
        # 输出处理器
        self.output_processor = None
        
        # 新增：射流相关初始化
        self.jet_regions = []  # 射流区域列表
        self.jet_velocities = []  # 射流速度列表
        self.jet_active = True  # 射流是否激活
        
        # 初始化射流参数
        if jet_params is not None:
            self.init_jets(jet_params)
        else:
            # 默认射流参数（可根据需要调整）
            default_jets = [
                {"position": [120.0, 120.0], "velocity": [0.3, 0.1], "radius": 3},  # 上表面射流
                {"position": [120.0, 80.0], "velocity": [0.2, -0.1], "radius": 2}  # 下表面射流
            ]
            self.init_jets(default_jets)

    # 新增：初始化射流函数
    def init_jets(self, jet_params):
        """
        初始化射流区域和参数
        
        jet_params: 射流参数列表，每个元素是一个字典，包含：
            - position: [x, y] 射流中心位置
            - velocity: [vx, vy] 射流速度矢量
            - radius: 射流区域半径
        """
        for params in jet_params:
            pos = ti.Vector(params["position"], dt=ti.f32)
            vel = ti.Vector(params["velocity"], dt=ti.f32)
            radius = params["radius"]
            
            # 找出射流区域内的所有边界点
            jet_points = []
            for k in range(self.num_boundary_pts):
                dist = tm.distance(self.boundary_pos[k], pos)
                if dist <= radius:
                    jet_points.append(k)
            
            if jet_points:
                self.jet_regions.append(jet_points)
                self.jet_velocities.append(vel)
                print(f"初始化射流区域: 位置 {pos}, 速度 {vel}, 包含 {len(jet_points)} 个边界点")

    # 新增：应用射流效应
    @ti.kernel
    def apply_jet_effect(self):
        """在射流区域应用额外的动量注入"""
        if self.jet_active and len(self.jet_regions) > 0:
            for r in ti.static(range(len(self.jet_regions))):
                jet_vel = self.jet_velocities[r]
                for k in ti.static(self.jet_regions[r]):
                    # 对射流区域的边界点施加额外速度
                    self.boundary_vel[k] = jet_vel
                    
                    # 通过增加边界力来实现射流动量注入
                    force = 2.0 * (self.boundary_vel[k] - self.interp_vel[k]) / self.dt
                    self.boundary_force[k] += force

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
            self.boundary_rho[k] = rho / num if num > 0 else 1.0
            self.interp_vel[k] = interp_v

    @ti.kernel
    def calculate_boundary_force(self):
        for k in self.boundary_pos:
            # 使用插值得到的边界点密度
            rho_k = self.boundary_rho[k] if self.boundary_rho[k] > 1e-6 else 1.0
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

    # 新增：更新射流参数的函数
    def update_jet_parameters(self, jet_index, new_velocity=None, new_active=None):
        """
        更新射流参数
        
        jet_index: 射流索引
        new_velocity: 新的射流速度矢量，如果为None则不更新
        new_active: 射流是否激活，如果为None则不更新
        """
        if 0 <= jet_index < len(self.jet_velocities) and new_velocity is not None:
            self.jet_velocities[jet_index] = ti.Vector(new_velocity, dt=ti.f32)
            print(f"更新射流 {jet_index} 速度为 {new_velocity}")
        
        if new_active is not None:
            self.jet_active = new_active
            print(f"射流 {'激活' if new_active else '关闭'}")

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
        
        # 新增：应用射流效应
        self.apply_jet_effect()
        
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
                
                # 新增：输出射流状态信息
                jet_info = f"射流: {'开启' if self.jet_active else '关闭'}"
                print(f"迭代步数: {self.it[None]}, 阻力: {drag:.6f}, 升力: {lift:.6f}, "
                      f"C_D: {cd:.4f}, C_L: {cl:.4f}, {jet_info}")

    def show(self):
        """使用Taichi GUI显示实时流场可视化 - 上下两列布局"""
        self.init()
        
        # 创建可视化器
        visualizer = LBMVisualizer(self.nx, self.ny)
        
        # 创建GUI窗口 - 调整为上下两列的尺寸
        gui = ti.GUI(self.name, res=(self.nx, visualizer.combined_height))
        
        while gui.running:
            # 执行多个时间步
            for _ in range(5):
                self.step()
                self.it[None] += 1
                
                # 记录力的历史
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

                jet_info = f"射流: {'开启' if self.jet_active else '关闭'}"
                print(f"步数: {self.it[None]}, 阻力: {drag:.6f}, 升力: {lift:.6f}, "
                      f"C_D: {cd:.4f}, C_L: {cl:.4f}, {jet_info}")
            
            # 更新可视化
            visualizer.update_visualization(self.vel)
            
            # 显示组合图像
            gui.set_image(visualizer.get_combined_image())
            
            # 绘制边界点和射流区域
            visualizer.draw_boundary_points(gui, self.boundary_pos)
            
            # 新增：绘制射流区域
            for i, jet_region in enumerate(self.jet_regions):
                if jet_region and self.jet_active:
                    # 取射流区域中心作为标记点
                    center_idx = jet_region[len(jet_region)//2]
                    pos = self.boundary_pos[center_idx]
                    gui.circle(pos, color=0x00ff00, radius=5)  # 用绿色圆圈标记射流位置
            
            # 显示信息文本
            drag_lift_coeffs = self.calculate_drag_lift()
            cd = drag_lift_coeffs.z  # 阻力系数
            cl = drag_lift_coeffs.w  # 升力系数
            
            visualizer.draw_info_text(gui, self.it[None], self.Red, cd, cl)
            
            # 新增：射流控制交互
            if gui.get_event(ti.GUI.PRESS):
                if gui.event.key == ti.GUI.SPACE:
                    self.jet_active = not self.jet_active
                    print(f"射流已{'激活' if self.jet_active else '关闭'}")
                elif gui.event.key == '1':
                    self.update_jet_parameters(0, new_velocity=[0.4, 0.15])  # 0.5弦长上表面
                elif gui.event.key == '2':
                    self.update_jet_parameters(1, new_velocity=[0.4, -0.15])  # 0.5弦长下表面
                elif gui.event.key == '3':
                    self.update_jet_parameters(2, new_velocity=[0.35, 0.12])  # 0.65弦长上表面
                elif gui.event.key == '4':
                    self.update_jet_parameters(3, new_velocity=[0.35, -0.12])  # 0.65弦长下表面
                elif gui.event.key == '5':
                    self.update_jet_parameters(4, new_velocity=[0.3, 0.08])  # 0.8弦长上表面
                elif gui.event.key == '6':
                    self.update_jet_parameters(5, new_velocity=[0.3, -0.08])  # 0.8弦长下表面
            
            gui.show()

    
    def output(self):
        """输出采样点的流场数据，返回展平的numpy数组"""
        # 确保输出处理器已初始化
        if self.output_processor is None:
            self.output_processor = create_output_processor(self)
        # 调用输出处理器的output方法
        return self.output_processor.output()

    def control(self):
        """控制函数 - 预留的控制接口"""
        pass
        


if __name__ == '__main__':
    # 定义射流参数 - 在机翼0.5、0.65、0.8弦长位置
    air_chord = 100  # 弦长
    air_origin_x = 100.0  # 机翼原点x坐标
    air_center_y = 100.0  # 机翼中心y坐标
    
    jet_parameters = [
        {"position": [air_origin_x + 0.5 * air_chord, air_center_y + 10], "velocity": [0.3, 0.1], "radius": 3},   # 0.5弦长位置上表面
        {"position": [air_origin_x + 0.5 * air_chord, air_center_y - 10], "velocity": [0.3, -0.1], "radius": 3},  # 0.5弦长位置下表面
        {"position": [air_origin_x + 0.65 * air_chord, air_center_y + 8], "velocity": [0.25, 0.08], "radius": 3},  # 0.65弦长位置上表面
        {"position": [air_origin_x + 0.65 * air_chord, air_center_y - 8], "velocity": [0.25, -0.08], "radius": 3}, # 0.65弦长位置下表面
        {"position": [air_origin_x + 0.8 * air_chord, air_center_y + 5], "velocity": [0.2, 0.05], "radius": 3},   # 0.8弦长位置上表面
        {"position": [air_origin_x + 0.8 * air_chord, air_center_y - 5], "velocity": [0.2, -0.05], "radius": 3}    # 0.8弦长位置下表面
    ]
    
    lbm = lbm_solver(
        nx=400,
        ny=200,
        Red=1500, 
        inlet_velocity=0.1,
        air_c=100,
        air_para=[0, 0, 12.0, -20.0],
        air_o=[100.0, 100.0],
        jet_params=jet_parameters  # 传入射流参数
    )
    
    # 运行可视化模式以观察射流效果
    lbm.show()
    # 或者运行求解模式
    # lbm.solver(steps=1000)
    # print(lbm.output())