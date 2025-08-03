# 产生一组圆柱的点
import numpy as np
import matplotlib.pyplot as plt


def generate_cylinder(radius, center_x=0, center_y=0, num_points=None):
    """
    生成圆柱边界点
    
    参数:
    radius: 圆柱半径
    center_x: 圆心x坐标
    center_y: 圆心y坐标
    num_points: 边界点数量，如果为None则根据周长自动计算
    
    返回:
    num_points: 实际生成的点数
    x_coords: x坐标数组
    y_coords: y坐标数组
    """
    # 计算圆周长
    circumference = 2 * np.pi * radius
    
    # 如果未指定点数，根据周长自动计算（保持与翼型相似的点密度）
    if num_points is None:
        num_points = int(circumference / 0.8)  # 与翼型生成保持一致的点间距
    
    # 生成均匀分布的角度
    theta = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    
    # 计算圆周上的点坐标
    x_coords = center_x + radius * np.cos(theta)
    y_coords = center_y + radius * np.sin(theta)
    
    return num_points, x_coords, y_coords


def obstacles_generate_cylinder(radius, center_x=0, center_y=0, num_points=None):
    """
    圆柱障碍物生成函数，与翼型生成函数接口保持一致
    
    参数:
    radius: 圆柱半径
    center_x: 圆心x坐标（相对坐标）
    center_y: 圆心y坐标（相对坐标）
    num_points: 边界点数量
    
    返回:
    num_boundary_pts: 边界点数量
    x_coords: x坐标数组
    y_coords: y坐标数组
    """
    return generate_cylinder(radius, center_x, center_y, num_points)


def test():
    """测试函数"""
    # 生成半径为50的圆柱
    radius = 20
    num_boundary_pts, x_coords, y_coords = obstacles_generate_cylinder(
        radius=radius, center_x=0, center_y=0)
    
    print(f"生成的边界点数量: {num_boundary_pts}")
    print(f"圆柱半径: {radius}")
    print(f"x坐标范围: [{x_coords.min():.2f}, {x_coords.max():.2f}]")
    print(f"y坐标范围: [{y_coords.min():.2f}, {y_coords.max():.2f}]")
    
    # 绘制圆柱
    plt.figure(figsize=(8, 8))
    plt.plot(x_coords, y_coords, 'b-', linewidth=2, label='圆柱边界')
    plt.scatter(x_coords, y_coords, s=10, c='red', alpha=0.6, label='边界点')
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'圆柱绕流 - 半径: {radius}')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    test()