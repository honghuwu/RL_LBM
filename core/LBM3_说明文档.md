# LBM求解器 (lbm3.py) 详细说明文档

## 概述

本文档详细介绍了基于Taichi的格子玻尔兹曼方法(LBM)流体求解器的模块化版本。该求解器实现了D2Q9格子玻尔兹曼方法，结合浸没边界法(IBM)处理复杂几何边界，特别适用于翼型绕流的数值模拟。

## 目录

1. [技术架构](#技术架构)
2. [核心算法](#核心算法)
3. [类结构分析](#类结构分析)
4. [关键方法详解](#关键方法详解)
5. [物理模型](#物理模型)
6. [使用示例](#使用示例)
7. [性能优化](#性能优化)

## 技术架构

### 主要特点
- **GPU加速**: 使用Taichi框架实现GPU并行计算
- **模块化设计**: 可视化模块独立，提高代码可维护性
- **高性能**: 优化的LBM算法和IBM实现
- **实时可视化**: 基于Taichi GUI的实时流场显示

### 依赖模块
```python
import taichi as ti              # GPU加速计算框架
import numpy as np               # 数值计算

# import matplotlib.pyplot as plt  # 绘图工具

from show_tools import LBMVisualizer  # 可视化模块
from tools.obstacles_generate.naca_genarate import obstacles_generate  # 翼型生成
```

## 核心算法

### 1. D2Q9格子玻尔兹曼方法

**离散速度模型**:
```python
# 9个离散速度方向
e = [[0,0], [1,0], [0,1], [-1,0], [0,-1], [1,1], [-1,1], [-1,-1], [1,-1]]

# 对应权重
w = [4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36]
```

**平衡态分布函数**:
```
f_eq = w_i * ρ * (1 + e_i·u/c_s² + (e_i·u)²/(2c_s⁴) - u²/(2c_s²))
```

### 2. 浸没边界法(IBM)

**核心思想**: 通过在流体和固体边界之间施加力来模拟复杂几何边界

**实现步骤**:
1. **速度插值**: 从欧拉网格插值到拉格朗日边界点
2. **力计算**: 根据速度差计算边界力
3. **力扩散**: 将边界力扩散到周围欧拉网格点

### 3. 数值求解流程

```
初始化 → 速度插值 → 计算边界力 → 力扩散 → 速度更新 → 碰撞 → 迁移 → 宏观量更新 → 边界条件
```

## 类结构分析

### lbm_solver类

#### 初始化参数
```python
def __init__(self, nx, ny, Red, inlet_velocity, air_para, air_c, air_o, air_d=0, name='LBM Solver'):
```

| 参数 | 类型 | 说明 |
|------|------|------|
| nx, ny | int | 网格尺寸 400 * 200 |
| Red | float | 雷诺数 （可变？） |
| inlet_velocity | float | 入流速度   0.1 |
| air_para | list | 翼型参数 [m, p, t, α] |
| air_c | float | 翼型弦长 |
| air_o | list | 翼型中心位置 |
| air_d | float | 旋转中心距离 |

#### 核心数据结构

**欧拉场(流场)**:
```python
self.rho = ti.field(float, shape=(nx, ny))           # 密度场

self.vel = ti.Vector.field(2, float, shape=(nx, ny)) # 速度场

self.f_old = ti.Vector.field(9, float, shape=(nx, ny)) # 分布函数(旧)
self.f_new = ti.Vector.field(9, float, shape=(nx, ny)) # 分布函数(新)
self.euler_force = ti.Vector.field(2, float, shape=(nx, ny)) # 欧拉力场
```

**拉格朗日场(边界)**:
```python
self.boundary_pos = ti.Vector.field(2, float, shape=num_boundary_pts)    # 边界点位置
self.boundary_vel = ti.Vector.field(2, float, shape=num_boundary_pts)    # 边界点速度
self.boundary_force = ti.Vector.field(2, float, shape=num_boundary_pts)  # 边界点力
self.interp_vel = ti.Vector.field(2, float, shape=num_boundary_pts)      # 插值速度
```

## 关键方法详解

### 1. 平衡态分布函数计算
```python
@ti.func
def f_eq(self, i, j):
    """计算平衡态分布函数"""
    vel_ij = self.vel[i, j]
    rho_ij = self.rho[i, j]
    uv_sq = vel_ij.dot(vel_ij)
    
    result = ti.Vector.zero(float, 9)
    for k in ti.static(range(9)):
        e_k = tm.vec2(self.e[k, 0], self.e[k, 1])
        eu = e_k.dot(vel_ij)
        result[k] = self.w[k] * rho_ij * (1.0 + eu / self.cs2 + 
                   0.5 * (eu / self.cs2)**2 - 0.5 * uv_sq / self.cs2)
    return result
```

### 2. IBM速度插值
```python
@ti.kernel
def interpolate_velocity(self):
    """从欧拉网格插值速度到拉格朗日边界点"""
    for k in self.boundary_pos:
        lag_pos = self.boundary_pos[k]
        ix_base, iy_base = int(lag_pos.x), int(lag_pos.y)
        interp_v = tm.vec2(0.0, 0.0)
        
        # 4点插值
        for i_offset in range(-1, 3):
            for j_offset in range(-1, 3):
                i, j = ix_base + i_offset, iy_base + j_offset
                if 0 <= i < self.nx and 0 <= j < self.ny:
                    euler_pos = tm.vec2(float(i), float(j))
                    weight = self.discrete_delta(lag_pos - euler_pos)
                    interp_v += self.vel[i, j] * weight
        
        self.interp_vel[k] = interp_v
```

### 3. 离散Delta函数
```python
@ti.func
def _phi(self, x):
    """4点离散Delta函数"""
    r = abs(x)
    if r < 1.0:
        return (3.0 - 2.0 * r + ti.sqrt(1.0 + 4.0 * r - 4.0 * r ** 2)) * 0.125
    elif r < 2.0:
        return (5.0 - 2.0 * r - ti.sqrt(-7.0 + 12.0 * r - 4.0 * r ** 2)) * 0.125
    else:
        return 0.0

@ti.func
def discrete_delta(self, r_vec):
    """二维离散Delta函数"""
    return self._phi(r_vec.x) * self._phi(r_vec.y)
```

### 4. 边界力计算
```python
@ti.kernel
def calculate_boundary_force(self):
    """计算边界点上的力"""
    for k in self.boundary_pos:
        rho_k = 1.0  # 参考密度
        # 使用直接力方法
        force = 2 * rho_k * (self.boundary_vel[k] - self.interp_vel[k]) / self.dt
        self.boundary_force[k] = force
```

### 5. 力扩散
```python
@ti.kernel
def spread_force(self):
    """将拉格朗日力扩散到欧拉网格"""
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
```

### 6. LBM碰撞步
```python
@ti.kernel
def collision(self):
    """LBM碰撞步骤"""
    for i, j in ti.ndrange(self.nx, self.ny):
        feq_ij = self.f_eq(i, j)
        vol_f = self.vol_force(i, j)  # 体积力项
        
        for k in ti.static(range(9)):
            self.f_new[i,j][k] = (self.f_old[i,j][k] - 
                                self.inv_tau * (self.f_old[i,j][k] - feq_ij[k]) + vol_f[k])
```

### 7. LBM迁移步
```python
@ti.kernel
def streaming(self):
    """LBM迁移步骤 (Pull Scheme)"""
    for i, j in ti.ndrange(self.nx, self.ny):
        for k in ti.static(range(9)):
            # 计算源网格点坐标
            ip = i - ti.cast(self.e[k, 0], ti.i32)
            jp = j - ti.cast(self.e[k, 1], ti.i32)

            # 周期性边界条件
            if ip < 0: ip = self.nx - 1
            if ip > self.nx - 1: ip = 0
            if jp < 0: jp = self.ny - 1
            if jp > self.ny - 1: jp = 0
            
            # 从源网格点拉取数据
            self.f_old[i, j][k] = self.f_new[ip, jp][k]
```

## 物理模型

### 1. 无量纲参数

**雷诺数**:
```
Re = ρUL/μ = UL/ν
```

**松弛时间**:
```
τ = 3ν + 0.5
```

其中：
- U: 特征速度
- L: 特征长度(弦长)
- ν: 运动粘度

### 2. 升阻力计算

```python
@ti.kernel
def calculate_drag_lift(self) -> tm.vec4:
    """计算升力、阻力及其系数"""
    drag = 0.0
    lift = 0.0

    for k in self.boundary_pos:
        force = -self.boundary_force[k]
        drag += force.dot(tm.vec2(1.0, 0.0))  # x方向分量
        lift += force.dot(tm.vec2(0.0, 1.0))  # y方向分量

    # 计算无量纲系数
    dynamic_pressure = 0.5 * rho_ref * U² * L
    cd = drag / dynamic_pressure
    cl = lift / dynamic_pressure

    return tm.vec4(drag, lift, cd, cl)
```

### 3. 边界条件

**入流边界** (左边界):
```python
self.vel[0, j] = tm.vec2(inlet_velocity, 0.0)
self.rho[0, j] = self.rho[1, j]
```

**出流边界** (右边界):
```python
self.vel[nx-1, j] = self.vel[nx-2, j]
self.rho[nx-1, j] = self.rho[nx-2, j]
```

**周期性边界** (上下边界):
自动处理，无需特殊设置

## 使用示例

### 基本使用
```python
# 创建求解器实例
lbm = lbm_solver(
    nx=400,                    # x方向网格数
    ny=200,                    # y方向网格数
    Red=1500,                  # 雷诺数
    inlet_velocity=0.1,        # 入流速度
    air_c=100,                 # 翼型弦长
    air_para=[0, 0, 12.0, -20.0],  # NACA翼型参数 [m, p, t, α]
    air_o=[100.0, 100.0],      # 翼型中心位置
)

# 实时可视化
lbm.show()

# 或者批量计算
lbm.solver(steps=10000)
```

### 翼型参数说明
```python
air_para = [m, p, t, alpha]
```
- m: 最大弯度 (%)
- p: 最大弯度位置 (弦长的%)
- t: 最大厚度 (%)
- alpha: 攻角 (度)

## 性能优化

### 1. GPU内存管理
```python
ti.init(arch=ti.gpu, device_memory_fraction=0.8)
```

### 2. 并行计算优化
- 使用`@ti.kernel`装饰器实现GPU并行
- 避免在kernel中使用Python对象
- 使用`ti.static`优化循环

### 3. 数据访问优化
- 合理安排数据布局
- 减少全局内存访问
- 使用局部变量缓存

## 扩展功能

### 1. 可视化模块
```python
from show_tools import LBMVisualizer

visualizer = LBMVisualizer(nx, ny)
visualizer.update_visualization(vel_field)
gui.set_image(visualizer.get_combined_image())
```

### 2. 力历史记录
```python
self.drag_history = []
self.lift_history = []
self.time_history = []
```

### 3. 多翼型支持
通过修改`obstacles_generate`函数可以支持多个翼型或其他几何形状。

## 注意事项

1. **数值稳定性**: 确保CFL条件满足，避免数值不稳定
2. **边界处理**: IBM方法需要足够的边界点密度
3. **内存使用**: 大网格时注意GPU内存限制
4. **收敛性**: 监控升阻力系数的收敛性

## 总结

本LBM求解器实现了高效的翼型绕流模拟，具有以下优势：

- **高性能**: GPU加速，适合大规模计算
- **模块化**: 代码结构清晰，易于维护和扩展
- **实时可视化**: 直观的流场显示
- **物理准确**: 基于成熟的LBM-IBM耦合方法

该求解器可广泛应用于空气动力学研究、教学演示和工程设计等领域。