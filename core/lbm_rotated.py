"""
=============================================================================
LBM求解器 - 智能控制版本 (lbm5.py)
=============================================================================

文件描述:
    基于Taichi的格子玻尔兹曼方法(LBM)流体求解器的智能控制版本。在v4.0基础上
    新增了强化学习控制器功能，实现了主动流动控制，为智能流体控制奠定基础。

主要功能:
    - D2Q9格子玻尔兹曼方法求解不可压缩流体
    - 浸没边界法处理复杂几何边界(NACA翼型)
    - 智能控制器系统(可移动圆柱控制器)
    - 模块化可视化系统
    - Taichi GUI实时可视化
    - 升阻力系数计算和监控
    - 强化学习接口

核心算法:
    1. 优化的LBM碰撞-迁移步骤
    2. 改进的IBM力计算方法(密度加权)
    3. 智能控制器力计算和传播
    4. 简化的体积力项处理
    5. 周期性边界条件

技术特点:
    - 使用Taichi GPU加速计算
    - 模块化可视化架构(show_tools.LBMVisualizer)
    - 面向对象的控制器设计(movable_column.controller)
    - 高性能Taichi GUI显示
    - 自定义颜色映射
    - 上下双场布局显示
    - 控制器可视化

智能控制新特性:
    - 可移动圆柱控制器
    - 控制器速度和密度插值
    - 控制器力计算和传播
    - 动态控制器位置更新
    - 强化学习控制接口
    - 控制器可视化显示

版本更新内容 (v5.0):
    - 新增: 智能控制器系统
    - 新增: movable_column.controller类集成
    - 新增: 控制器IBM力计算
    - 新增: 控制器可视化
    - 新增: move_controller()方法
    - 改进: control()函数实现
    - 优化: 控制器密度插值算法

依赖模块:
    - show_tools.LBMVisualizer: 可视化模块
    - obstacles_generate.naca_genarate: 翼型生成
    - control_tools.movable_column: 可移动控制器
    - output_tool.output: 输出处理器
    - taichi: GPU加速计算框架

版本历史:
    - v1.0: 基础LBM求解器
    - v2.0: 浸没边界法集成
    - v3.0: 模块化可视化
    - v4.0: 输出处理器集成
    - v5.0: 智能控制器系统 (当前版本)

版本信息:
    - 版本: v5.0 (智能控制版本)
    - 基于: lbm4.py v4.0
    - 创建日期: 2024
    - 可视化: 模块化Taichi GUI + 控制器显示
    - 边界条件: 周期性边界
    - 架构: 模块化设计 + 智能控制
    - 控制器: 可移动圆柱控制器

作者: LBM项目组
=============================================================================
"""

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

        # —— 保存“未加 air_o 平移”的翼型基准点（仅旋转用）——
        airfoil_base_np = np.stack((x_airfoil_np, y_airfoil_np), axis=1).astype(np.float32)

        # —— 用于保存基准点的 Taichi field —— 
        self.boundary_pos0 = ti.Vector.field(2, float, shape=self.num_boundary_pts)
        self.boundary_pos0.from_numpy(airfoil_base_np)


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
        
        # 强化学习输入处理器
        self.output_processor = None

        

    @ti.kernel
    def set_airfoil_angle(self, alpha_deg: float):
        # 角度转弧度
        rad = alpha_deg * 3.141592653589793 / 180.0
        c = ti.cos(rad)
        s = ti.sin(rad)
        ox = float(self.air_o[0])
        oy = float(self.air_o[1])
        for k in self.boundary_pos0:
            x = self.boundary_pos0[k].x
            y = self.boundary_pos0[k].y
            xr = c * x - s * y
            yr = s * x + c * y
            self.boundary_pos[k] = tm.vec2(xr + ox, yr + oy)

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

                print(f"步数: {self.it[None]}, 阻力: {drag:.6f}, 升力: {lift:.6f}, "
                        f"C_D: {cd:.4f}, C_L: {cl:.4f}")
            
            # 更新可视化
            visualizer.update_visualization(self.vel)
            
            # 显示组合图像
            gui.set_image(visualizer.get_combined_image())
            
            # 绘制边界点
            visualizer.draw_boundary_points(gui, self.boundary_pos)
            
            # 显示信息文本
            drag_lift_coeffs = self.calculate_drag_lift()
            cd = drag_lift_coeffs.z  # 阻力系数
            cl = drag_lift_coeffs.w  # 升力系数
            
            visualizer.draw_info_text(gui, self.it[None], self.Red, cd, cl)
            
            gui.show()

    
    def output(self):
        """输出采样点的流场数据，返回展平的numpy数组"""
        # 确保输出处理器已初始化
        if self.output_processor is None:
            self.output_processor = create_output_processor(self)
        # 调用输出处理器的output方法
        out = self.output_processor.output()
        #print(f"[DEBUG] step {self.it[None]} output shape = {out.shape}")
        return out


    def control(self, alpha_deg: float):
        # 运行中直接设置新攻角（旋转翼型几何）
        self.set_airfoil_angle(float(alpha_deg))



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
    # lbm.solver(steps=1000)
    # a = lbm.output()
    # #将a保存csv文件
    # np.savetxt('lbm_output.csv', a, delimiter=',')

