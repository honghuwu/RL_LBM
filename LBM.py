import sys
import matplotlib
import numpy as np
from matplotlib import cm

import taichi as ti
import taichi.math as tm

ti.init(arch=ti.gpu)

@ti.data_oriented
class lbm_solver:
    def __init__(
        self, 
        nx, 
        ny, 
        dx=1.0, 
        dt=1.0, 
        Red=100.0, 
        bc_type=[0,1,1,1], 
        bc_value=None, 
        air_para=None, 
        air_c=None, 
        air_o=None, 
        air_d=None, 
        name='LBM Solver With NACA'  # 默认窗口标题,
    ):
        self.name = name
        # 计算域大小
        self.nx = nx
        self.ny = ny
        # 雷诺数
        self.Red = Red  
        # 翼型参数
        self.air_para = air_para  
        # 翼型原点
        self.air_o = air_o  
        # 旋转中心与顶点的距离
        self.air_d = air_d  
        #弦长
        self.air_c = air_c
        #边界速度
        U = bc_value[0][0]  # bc_value格式: [[左速度], [上速度], [右速度], [下速度]]
        # 运动粘度
        self.niu = (U * self.air_c) / self.Red
        #弛豫时间 来源于LBM
        self.tau = 3.0 * self.niu + 0.5
        #弛豫时间的倒数
        self.inv_tau = 1.0 / self.tau
        #密度
        self.rho = ti.field(float, shape=(nx, ny))
        #速度
        self.vel = ti.Vector.field(2, float, shape=(nx, ny))
        #障碍物位置  需要修改
        self.mask = ti.field(float, shape=(nx, ny))
        #粒子分布场
        self.f_old = ti.Vector.field(9, float, shape=(nx, ny))
        #粒子分布场
        self.f_new = ti.Vector.field(9, float, shape=(nx, ny))
        #权重因子
        self.w = ti.types.vector(9, float)(4, 1, 1, 1, 1, 1 / 4, 1 / 4, 1 / 4, 1 / 4) / 9.0
        #方向向量
        self.e = ti.types.matrix(9, 2, int)([0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1])
        #边界条件  0 -> 狄利克雷边界 ; 1 -> 诺伊曼边界
        self.bc_type = ti.field(int, 4)
        self.bc_type.from_numpy(np.array(bc_type, dtype=np.int32))
        #边界速度的值
        self.bc_value = ti.Vector.field(2, float, shape=4)
        self.bc_value.from_numpy(np.array(bc_value, dtype=np.float32))

        # 生成NACA翼型
        m, p, t, alpha = self.air_para[0], self.air_para[1], self.air_para[2], self.air_para[3]
        self.num_points, self.x_airfoil_np, self.y_airfoil_np = self.generate_naca_airfoil(m, p, t, self.air_c, alpha)
        self.x_airfoil_np = self.x_airfoil_np + self.air_o[0]
        self.y_airfoil_np = self.y_airfoil_np + self.air_o[1]
        #转化为taichi
        self.x_airfoil = ti.field(dtype=ti.f32, shape=self.num_points)
        self.y_airfoil = ti.field(dtype=ti.f32, shape=self.num_points)
        self.x_airfoil.from_numpy(self.x_airfoil_np)
        self.y_airfoil.from_numpy(self.y_airfoil_np)

        self.generate_mask()

    def generate_naca_airfoil(self, m, p, t, c=300, alpha=0, rot_d=0):
        # 规范化输入参数
        m /= 100
        p /= 10
        if p == 0:
            p = 1e-6
        t /= 100
        alpha_rad = alpha * (np.pi / 180)  # 弧度制
        # 生成等间隔的x坐标
        n = 800  # 分辨率
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

    @ti.func
    def point_in_polygon(self, x, y, polygon_x: ti.template(), polygon_y: ti.template()):
        n = polygon_x.shape[0]
        inside = False
        p1x, p1y = polygon_x[0], polygon_y[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon_x[i % n], polygon_y[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        xinters = p1x  # 默认值
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    @ti.kernel
    def generate_mask(self):
        for i, j in ti.ndrange(self.nx, self.ny):
            if self.point_in_polygon(i, j, self.x_airfoil, self.y_airfoil):
                self.mask[i, j] = 1.0

    @ti.func  # 计算平衡分布函数
    def f_eq(self, i, j):
        eu = self.e @ self.vel[i, j]
        uv = tm.dot(self.vel[i, j], self.vel[i, j])
        return self.w * self.rho[i, j] * (1 + 3 * eu + 4.5 * eu * eu - 1.5 * uv)

    @ti.kernel
    def init(self): #初始化
        self.vel.fill(0)
        self.rho.fill(1)
        for i, j in self.rho:
            self.f_old[i, j] = self.f_new[i, j] = self.f_eq(i, j)

    @ti.kernel
    def collide_and_stream(self):  
        for i, j in ti.ndrange((1, self.nx - 1), (1, self.ny - 1)):
            for k in ti.static(range(9)):
                ip = i - self.e[k, 0]
                jp = j - self.e[k, 1]
                feq = self.f_eq(ip, jp)
                self.f_new[i, j][k] = (1 - self.inv_tau) * self.f_old[ip, jp][k] + feq[k] * self.inv_tau

    @ti.kernel
    def update_macro_var(self):  # 计算密度和速度
        for i, j in ti.ndrange((1, self.nx - 1), (1, self.ny - 1)):
            self.rho[i, j] = 0
            self.vel[i, j] = 0, 0
            for k in ti.static(range(9)):
                self.f_old[i, j][k] = self.f_new[i, j][k]
                self.rho[i, j] += self.f_new[i, j][k]
                self.vel[i, j] += tm.vec2(self.e[k, 0], self.e[k, 1]) * self.f_new[i, j][k]

            self.vel[i, j] /= self.rho[i, j]

    @ti.kernel
    def apply_bc(self):  # 施加边界条件
        #左右边界
        for j in range(1, self.ny - 1):
            # 左边界 (入口)
            self.apply_bc_core(1, 0, 0, j, 1, j)
            # 右边界 (出口)
            self.apply_bc_core(1, 2, self.nx - 1, j, self.nx - 2, j)

        #上下
        for i in range(self.nx):
            self.apply_bc_core(1, 1, i, self.ny - 1, i, self.ny - 2)
            self.apply_bc_core(1, 3, i, 0, i, 1)

        # 翼型障碍物边界条件处理
        for i, j in ti.ndrange(self.nx, self.ny):
            if  self.mask[i, j] == 1:
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

    def solve(self):
        gui = ti.GUI(self.name, (self.nx, 2 * self.ny))
        self.init()
        while not gui.get_event(ti.GUI.ESCAPE, ti.GUI.EXIT):
            for _ in range(10):
                self.collide_and_stream()
                self.update_macro_var()
                self.apply_bc()


            vel = self.vel.to_numpy()
            ugrad = np.gradient(vel[:, :, 0])
            vgrad = np.gradient(vel[:, :, 1])
            vor = ugrad[1] - vgrad[0]
            vel_mag = (vel[:, :, 0] ** 2.0 + vel[:, :, 1] ** 2.0) ** 0.5
            ## 颜色映射
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
            img = np.concatenate((vor_img, vel_img), axis=1)
            gui.set_image(img)
            gui.show()

if __name__ == '__main__':
    # NACA翼型绕流测试案例
    lbm = lbm_solver(
        nx=801,          # 计算域宽度
        ny=401,          # 计算域高度
        bc_type=[0,1,1,1],  # 边界条件类型
        bc_value=[[0.1, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],  # 入口速度U=0.1
        Red=1000,        # 雷诺数 (修改为1000)
        air_c=300,       # 翼型弦长
        air_para=[2.0, 4.0, 12.0, 5.0],  # NACA 2412: m=2%, p=40%, t=12%, alpha=5度
        air_o=[200.0, 200.0],  # 翼型中心位置坐标
    )
    lbm.solve()
