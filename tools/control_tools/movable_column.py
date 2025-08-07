import numpy as np
import matplotlib.pyplot as plt

class controller:
    def __init__(self,controller_radius,pos_x,pos_y):
        self.controller_radius = controller_radius
        self.x_center = pos_x
        self.y_center = pos_y
        self.num_points = 50
        self.x = None
        self.y = None
        self.x_vel = None
        self.y_vel = none 

    def column_create(self):
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
        circumference = 2 * np.pi * self.controller_radius
        
        # 如果未指定点数，根据周长自动计算（保持与翼型相似的点密度）
        if self.num_points is None:
            self.num_points = int(circumference / 0.8)  # 与翼型生成保持一致的点间距
        
        # 生成均匀分布的角度
        theta = np.linspace(0, 2 * np.pi, self.num_points, endpoint=False)
        
        # 计算圆周上的点坐标
        x_coords = self.x_center + self.controller_radius * np.cos(theta) - self.controller_radius
        y_coords = self.y_center + self.controller_radius * np.sin(theta)
        self.x = x_coords
        self.y = y_coords

        #返回点的 n*2 numpy数组
        return np.column_stack((x_coords, y_coords))
        

    def colume_num(self):

        return self.num_points


    def control(self,choice):
        w = 0.1 *  choice  / 10 / self.controller_radius
        theta = np.linspace(0, 2 * np.pi, self.num_points, endpoint=False)
        self.x_vel =  - w * self.controller_radius * np.sin(theta)
        self.y_vel = w * self.controller_radius * np.cos(theta)

        #返回一组速度
        return np.column_stack((self.x_vel,self.y_vel))




if __name__ == '__main__':
    controller = controller(100,100,100)
    points = controller.column_create()
    plt.scatter(points[:, 0], points[:, 1], s=10)
    plt.axis('equal')
    plt.show()




