"""
LBM流场可视化器
提供涡度场、速度场等的可视化功能
"""

import taichi as ti
import taichi.math as tm
import numpy as np


@ti.data_oriented
class LBMVisualizer:
    """LBM流场可视化器"""
    
    def __init__(self, nx, ny):
        """
        初始化可视化器
        
        Args:
            nx: x方向网格数
            ny: y方向网格数
        """
        self.nx = nx
        self.ny = ny
        
        # 创建可视化用的field
        self.vorticity = ti.field(float, shape=(nx, ny))
        self.vel_mag = ti.field(float, shape=(nx, ny))
        
        # 归一化后的场
        self.vorticity_norm = ti.field(float, shape=(nx, ny))
        self.vel_mag_norm = ti.field(float, shape=(nx, ny))
        
        # 颜色场
        self.vorticity_colored = ti.Vector.field(3, float, shape=(nx, ny))
        self.vel_mag_colored = ti.Vector.field(3, float, shape=(nx, ny))
        
        # 组合图像field - 上下两列布局
        self.separator_height = 10  # 分隔线高度
        self.combined_height = ny * 2 + self.separator_height
        self.combined_img = ti.Vector.field(3, float, shape=(nx, self.combined_height))

    @ti.kernel
    def compute_vorticity(self, vel: ti.template(), vorticity: ti.template()):
        """计算涡度场 - 适应周期性边界条件"""
        for i, j in ti.ndrange(self.nx, self.ny):
            # 左右边界处理（保持原有逻辑）
            if i == 0 or i == self.nx-1:
                vorticity[i, j] = 0.0
            else:
                # 计算涡度 ω = ∂v/∂x - ∂u/∂y
                dvdx = vel[i+1, j][1] - vel[i-1, j][1]
                
                # y方向使用周期性边界条件
                j_plus = (j + 1) % self.ny
                j_minus = (j - 1 + self.ny) % self.ny
                dudy = vel[i, j_plus][0] - vel[i, j_minus][0]
                
                vorticity[i, j] = 0.5 * (dvdx - dudy)

    @ti.kernel
    def compute_velocity_magnitude(self, vel: ti.template(), vel_mag: ti.template()):
        """计算速度大小"""
        for i, j in ti.ndrange(self.nx, self.ny):
            vel_mag[i, j] = vel[i, j].norm()

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

    def update_visualization(self, vel_field, vorticity_range=(-0.02, 0.02), velocity_range=(0.0, 0.4)):
        """
        更新可视化数据
        
        Args:
            vel_field: 速度场
            vorticity_range: 涡度场的显示范围
            velocity_range: 速度场的显示范围
        """
        # 计算可视化数据
        self.compute_vorticity(vel_field, self.vorticity)
        self.compute_velocity_magnitude(vel_field, self.vel_mag)
        
        # 归一化
        self.normalize_field(self.vorticity, self.vorticity_norm, 
                           vorticity_range[0], vorticity_range[1])
        self.normalize_field(self.vel_mag, self.vel_mag_norm, 
                           velocity_range[0], velocity_range[1])
        
        # 应用颜色映射
        self.apply_colormap_vorticity(self.vorticity_norm, self.vorticity_colored)
        self.apply_colormap_velocity(self.vel_mag_norm, self.vel_mag_colored)
        
        # 组合图像：上面涡度场，下面速度场
        self.combine_images_vertical(self.vorticity_colored, self.vel_mag_colored, 
                                   self.combined_img, self.separator_height)

    def get_combined_image(self):
        """获取组合后的图像数据"""
        return self.combined_img.to_numpy()

    def get_vorticity_image(self):
        """获取涡度场图像数据"""
        return self.vorticity_colored.to_numpy()

    def get_velocity_image(self):
        """获取速度场图像数据"""
        return self.vel_mag_colored.to_numpy()

    def draw_boundary_points(self, gui, boundary_pos, color=0x000000):
        """
        在GUI上绘制边界点
        
        Args:
            gui: Taichi GUI对象
            boundary_pos: 边界点位置数组
            color: 边界点颜色
        """
        boundary_pts = boundary_pos.to_numpy()
        
        # 上半部分（涡度场）的边界点
        for pt in boundary_pts:
            x = pt[0] / self.nx
            y = pt[1] / self.combined_height  # 调整到上半部分
            gui.circle((x, y), radius=1, color=color)
        
        # 下半部分（速度场）的边界点
        for pt in boundary_pts:
            x = pt[0] / self.nx
            y = (pt[1] + self.ny + self.separator_height) / self.combined_height  # 调整到下半部分
            gui.circle((x, y), radius=1, color=color)

    def draw_controller_points(self, gui, controller_pos, color=0xFF0000):
        """
        在GUI上绘制控制器点
        
        Args:
            gui: Taichi GUI对象
            controller_pos: 控制器点位置数组
            color: 控制器点颜色
        """
        controller_pts = controller_pos.to_numpy()
        
        for pt in controller_pts:
            x = pt[0] / self.nx
            y = pt[1] / self.combined_height  # 调整到上半部分
            gui.circle((x, y), radius=1, color=color)

        # 下半部分（速度场）的控制器点
        for pt in controller_pts:
            x = pt[0] / self.nx
            y = (pt[1] + self.ny + self.separator_height) / self.combined_height  # 调整到下半部分
            gui.circle((x, y), radius=1, color=color)


    def draw_info_text(self, gui, step, reynolds, cd, cl):
        """
        在GUI上绘制信息文本
        
        Args:
            gui: Taichi GUI对象
            step: 当前步数
            reynolds: 雷诺数
            cd: 阻力系数
            cl: 升力系数
        """
        info_text = f"Step: {step}, Re: {reynolds}, C_D: {cd:.3f}, C_L: {cl:.3f}"
        gui.text(info_text, pos=(0.02, 0.97), color=0xFFFFFF)
        
        # 显示场标签
        gui.text("Vorticity Field", pos=(0.02, 0.52), color=0xFFFFFF)
        gui.text("Velocity Magnitude", pos=(0.02, 0.02), color=0xFFFFFF)