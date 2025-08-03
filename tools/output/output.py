"""
LBM输出工具模块
包含涡度计算、采样点选择、机翼内部检测等功能
"""

import numpy as np
import taichi as ti
import matplotlib.pyplot as plt
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# 导入lbm3模块
from MyCode.core.lbm3 import lbm_solver

@ti.data_oriented
class LBMOutputProcessor:
    """LBM输出处理器，负责数据采样和输出"""
    
    def __init__(self, solver):
        """
        初始化输出处理器
        
        Args:
            solver: LBM求解器实例
        """
        self.solver = solver
        self.nx = solver.nx
        self.ny = solver.ny
        self.air_o = solver.air_o
        self.air_c = solver.air_c
        self.boundary_pos = solver.boundary_pos
        self.num_boundary_pts = solver.num_boundary_pts
        self.vel = solver.vel
    
    @ti.kernel
    def calculate_vorticity(self, vorticity: ti.template()):
        """计算涡度场"""
        for i, j in ti.ndrange(self.nx, self.ny):
            # 边界处理
            if i == 0 or i == self.nx - 1 or j == 0 or j == self.ny - 1:
                vorticity[i, j] = 0.0
            else:
                # 计算速度梯度 (涡度 = dv/dx - du/dy)
                dvdx = (self.vel[i+1, j].y - self.vel[i-1, j].y) / 2.0
                dudy = (self.vel[i, j+1].x - self.vel[i, j-1].x) / 2.0
                vorticity[i, j] = dvdx - dudy

    def generate_outer_points_by_normal(self,distance):
        """
        通过计算法线向量来生成一组围绕输入坐标的外围点。
        
        Args:
            x_coords (np.array): 原始形状的x坐标数组 (必须是闭合且有序的)。
            y_coords (np.array): 原始形状的y坐标数组。
            distance (float): 外围点距离原始形状的偏移量。

        Returns:
            tuple: (x_outer, y_outer) 包含外围点坐标的元组。
        """
        boundary_pos_np = self.boundary_pos.to_numpy()
        x_coords = boundary_pos_np[:, 0]
        y_coords = boundary_pos_np[:, 1]
        # 确保曲线是闭合的，最后一个点和第一个点相同
        if not (np.isclose(x_coords[0], x_coords[-1]) and np.isclose(y_coords[0], y_coords[-1])):
            x_coords = np.append(x_coords, x_coords[0])
            y_coords = np.append(y_coords, y_coords[0])

        # 使用 np.roll 高效地获取每个点的前一个点和后一个点
        x_prev = np.roll(x_coords, 1)
        y_prev = np.roll(y_coords, 1)
        x_next = np.roll(x_coords, -1)
        y_next = np.roll(y_coords, -1)
        
        # 1. 计算切线向量 (从 P_prev 到 P_next)
        tangent_x = x_next - x_prev
        tangent_y = y_next - y_prev

        # 2. 计算法线向量并确定“向外”方向
        # 对于顺时针点集，(ty, -tx) 是向外的法线方向
        normal_x = tangent_y
        normal_y = -tangent_x
        
        # 3. 将法线向量归一化（变成单位向量）
        norm = np.sqrt(normal_x**2 + normal_y**2)
        # 防止除以零
        norm[norm < 1e-6] = 1e-6 
        unit_normal_x = normal_x / norm
        unit_normal_y = normal_y / norm

        # 4. 将原始点沿着单位法线方向移动指定距离
        x_outer = x_coords + distance * unit_normal_x
        y_outer = y_coords + distance * unit_normal_y
        
        return x_outer, y_outer



    def is_inside_airfoil(self, x, y):
        """判断点是否在翼型内部 - 使用射线法"""
        # 获取边界点到CPU
        boundary_pos_np = self.boundary_pos.to_numpy()
        
        # 使用射线法判断点是否在多边形内部
        # 从点(x,y)向右发射射线，计算与多边形边的交点数
        intersections = 0
        
        for i in range(self.num_boundary_pts):
            # 当前边界点和下一个边界点
            x1 = boundary_pos_np[i][0]
            y1 = boundary_pos_np[i][1]

            x2 = boundary_pos_np[(i + 1) % self.num_boundary_pts][0]
            y2 = boundary_pos_np[(i + 1) % self.num_boundary_pts][1]
            
            # 检查射线是否与边相交
            # 射线方向：从(x,y)向右 (y坐标不变)
            
            # 边的y坐标范围检查
            if min(y1, y2) < y <= max(y1, y2):
                # 计算射线与边的交点x坐标
                if y2 != y1:  # 避免除零
                    x_intersect = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
                    # 如果交点在射线右侧，计数+1
                    if x_intersect > x:
                        intersections += 1
        
        # 奇数个交点表示在内部，偶数个交点表示在外部
        return intersections % 2 == 1

    def select_sampling_points(self):
        """选取机翼边界附近和后方的采样点"""
        # 获取翼型中心位置
        center_x = self.air_o[0]
        center_y = self.air_o[1]
        chord_length = self.air_c
        
        x_min = center_x

        x_max = center_x + self.nx * 0.5

        y_min = center_y - self.ny * 0.4

        y_max = center_y + self.ny * 0.25
        valid_points = []
        #在这个区域内选取150个点
        x = np.linspace(x_min, x_max, 20)
        y = np.linspace(y_min, y_max, 20)
        X, Y = np.meshgrid(x, y)
        points = np.column_stack((X.ravel(), Y.ravel()))

        # 筛选在翼型内部的点

        for x, y in points:
            if not self.is_inside_airfoil(x, y):
                valid_points.append([x, y])

        naca_points = []

        for i in range(3):
            x_outer, y_outer = self.generate_outer_points_by_normal(3 * (i + 1))
            #只选取等间距的20个xy
            step = max(1, len(x_outer)//80)
            x_selected = x_outer[::step]
            y_selected = y_outer[::step]
            
            # 将每个点作为[x, y]对添加到naca_points
            for j in range(len(x_selected)):
                naca_points.append([x_selected[j], y_selected[j]])

        #将naca_points加入valid_points
        valid_points.extend(naca_points)

        #将每个点都转化为整数，去除重复点
        valid_points = [list(map(int, point)) for point in valid_points]
        # 将列表转换为元组进行去重，然后再转回列表
        valid_points = [list(point) for point in set(tuple(point) for point in valid_points)]

        print(len(valid_points))

        return valid_points

    def show_valid_points(self):
        valid_points = self.select_sampling_points()
        x = [point[0] for point in valid_points]
        y = [point[1] for point in valid_points]
        # 提取翼型边界点
        x_airfoil = [point[0] for point in self.boundary_pos.to_numpy()]
        y_airfoil = [point[1] for point in self.boundary_pos.to_numpy()]
        
        # 设置科研风格的图形参数
        plt.rcParams.update({
            'font.size': 14,
            'font.family': 'serif',
            'axes.linewidth': 1.2,
            'axes.labelsize': 16,
            'axes.titlesize': 18,
            'xtick.labelsize': 14,
            'ytick.labelsize': 14,
            'legend.fontsize': 14,
            'figure.dpi': 300
        })
        
        # 创建图形，使用科研标准尺寸
        fig, ax = plt.subplots(figsize=(12, 6), dpi=100)
        
        # 绘制采样点 - 使用深蓝色，较小的点
        scatter1 = ax.scatter(x, y, s=0.8, alpha=0.9,c='#d62728', 
                             label='Sampling Points', edgecolors='none')
        
        # 绘制翼型边界 - 使用深红色，稍大的点
        scatter2 = ax.scatter(x_airfoil, y_airfoil, s=2.0, c='#1f77b4' ,
                             alpha=0.9, label='Airfoil Boundary', edgecolors='none')
        
        # 设置标题和标签
        ax.set_title('Distribution of Sampling Points around NACA Airfoil', 
                    fontweight='bold', pad=20)
        ax.set_xlabel('x/c', fontweight='bold')
        ax.set_ylabel('y/c', fontweight='bold')
        
        # 设置坐标轴范围和网格
        ax.set_xlim(0, 400)
        ax.set_ylim(0, 200)
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.set_aspect('equal', adjustable='box')
        
        # 添加图例
        legend = ax.legend(loc='upper right', frameon=True, fancybox=True, 
                          shadow=True, framealpha=0.9)
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_edgecolor('gray')
        
        # 设置坐标轴样式
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.2)
        ax.spines['bottom'].set_linewidth(1.2)
        
        # 调整布局
        plt.tight_layout()
        plt.show()


    def output(self):
        """输出采样点的流场数据，返回展平的numpy数组"""
        # 创建涡度场
        vorticity = ti.field(float, shape=(self.nx, self.ny))
        
        # 计算涡度
        self.calculate_vorticity(vorticity)
        
        # 选取采样点
        sampling_points = self.select_sampling_points()
        
        # 从GPU获取数据到CPU
        vel_np = self.vel.to_numpy()
        vorticity_np = vorticity.to_numpy()
        
        # 准备输出数据
        output_data = []
        for x, y in sampling_points:
            # 获取速度分量
            u_val = vel_np[x, y][0]
            v_val = vel_np[x, y][1]
            # 获取涡度
            vort_val = vorticity_np[x, y]

            output_data.append([
                x,y,
                u_val, v_val,   # 速度分量
                vort_val,       # 涡度
            ])
        
        # 转换为numpy数组并展平
        output_array = np.array(output_data)
        flattened_array = output_array.flatten()
        
        print(f"已选取 {len(sampling_points)} 个采样点")
        print(f"返回展平数组，长度: {len(flattened_array)}")
        print(f"数据范围: u=[{output_array[:, 2].min():.4f}, {output_array[:, 2].max():.4f}], "
              f"v=[{output_array[:, 3].min():.4f}, {output_array[:, 3].max():.4f}], "
              f"vorticity=[{output_array[:, 4].min():.4f}, {output_array[:, 4].max():.4f}]")
        
        return flattened_array


def create_output_processor(solver):
    """
    创建输出处理器的工厂函数
    
    Args:
        solver: LBM求解器实例
        
    Returns:
        LBMOutputProcessor: 输出处理器实例
    """
    return LBMOutputProcessor(solver)


if __name__ == '__main__':
    solver = lbm_solver(
        nx=400, ny=200,
        Red=1000,
        inlet_velocity=0.1,
        air_c=100,
        air_para=[0, 0, 12.0, -20.0],
        air_o=[100.0, 100.0]
    )
    solver.init()
    processor = create_output_processor(solver)
    processor.show_valid_points()