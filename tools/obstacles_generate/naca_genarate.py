import numpy as np
import matplotlib.pyplot as plt


def generate_naca_airfoil(m, p, t, c=300, alpha=0, rot_d=0):
        # 规范化输入参数
        m /= 100
        p /= 10
        if p == 0:
            p = 1e-6
        t /= 100
        alpha_rad = alpha * (np.pi / 180)  # 弧度制
        # 生成等间隔的x坐标
        n =200  # 分辨率
        x = np.linspace(0, c, n)
        yt = 5 * t * (0.2969 * (x/c)**0.5 - 0.1260 * (x/c) - 0.3516 * (x/c)**2 + 
                      0.2843 * (x/c)**3 - 0.1015 * (x/c)**4) * c

        # 计算弯度线 yc 和导数 dyc/dx
        yc = np.where(x < p * c, 
                      (m / p**2) * (2 * p * x / c - (x / c)**2) * c, 
                      (m / (1 - p)**2) * ((1 - 2 * p) + 2 * p * (x / c) - (x / c)**2) * c)
        dyc_dx = np.where(x < p * c, 
                          (2 * m / p**2) * (p - x / c), 
                          (2 * m / (1 - p)**2) * (p - x / c))

        theta = np.arctan(dyc_dx)
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)

        # 上下表面坐标
        xu = x - yt * sin_theta - c * rot_d
        xl = x + yt * sin_theta - c * rot_d
        yu = yc + yt * cos_theta
        yl = yc - yt * cos_theta

        # 旋转攻角
        def rotate(x, y, alpha):
            cos_a = np.cos(alpha)
            sin_a = np.sin(alpha)
            x_rot = x * cos_a - y * sin_a
            y_rot = x * sin_a + y * cos_a
            return x_rot, y_rot

        xu, yu = rotate(xu, yu, alpha_rad)
        xl, yl = rotate(xl, yl, alpha_rad)
        x, yc = rotate(x, yc, alpha_rad)

        # 合并上表面和下表面形成完整的轮廓，顺时针排列
        x_full = np.concatenate([xu[::-1], xl[1:]])
        y_full = np.concatenate([yu[::-1], yl[1:]])

        # 计算累积弦长距离
        distances = np.sqrt(np.diff(x_full)**2 + np.diff(y_full)**2)
        cumulative_distance = np.concatenate([[0], np.cumsum(distances)])

        # 均匀分布 num_points 个点
        num_points = int(cumulative_distance[-1] / 0.8)
        target_distances = np.linspace(0, cumulative_distance[-1], num_points)
        x_uniform = np.interp(target_distances, cumulative_distance, x_full)
        y_uniform = np.interp(target_distances, cumulative_distance, y_full)

        return num_points, x_uniform, y_uniform



def obstacles_generate(m, p, t, c=300, alpha=0, rot_d=0):
    return generate_naca_airfoil(m, p, t, c, alpha, rot_d)

def test():
    num_boundary_pts, x_airfoil_np, y_airfoil_np = obstacles_generate(
        0, 0, 12, c=300, alpha=-20, rot_d=0)
    print(num_boundary_pts)
    print(x_airfoil_np)
    print(y_airfoil_np)
    plt.plot(x_airfoil_np, y_airfoil_np)
    # #横纵坐标单位一致
    plt.axis('equal')
    # #画出散点图，不要连线
    plt.scatter(x_airfoil_np, y_airfoil_np, s=1)
    # # 显示图形
    plt.show()



if __name__ == '__main__':
    test()