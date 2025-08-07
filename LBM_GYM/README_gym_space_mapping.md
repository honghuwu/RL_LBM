# LBM Gym环境空间映射说明文档

## 概述

本文档详细说明了LBM强化学习环境中观察空间（Observation Space）和动作空间（Action Space）与相关工具模块的对应关系。

## 文件结构

- **主环境文件**: `lbm_env_structure_modified2.py` - 定义了Gym环境
- **观察空间工具**: `tools/output_tool/output.py` - 负责生成观察数据
- **动作空间工具**: `tools/control_tools/movable_column.py` - 负责执行控制动作

---

## 1. 观察空间 (Observation Space) 对应关系

### 1.1 Gym环境中的定义
```python
# 在 lbm_env_structure_modified2.py 中
self.observation_space = spaces.Box(
    low=-np.inf, high=np.inf, shape=(642, 5), dtype=np.float32
)
```

### 1.2 对应的工具模块
**文件**: `tools/output_tool/output.py`  
**核心类**: `LBMOutputProcessor`

### 1.3 详细对应关系

#### 观察数据生成流程：
1. **采样点选择** (`select_sampling_points()`)
   - 在翼型周围和后方选择约642个采样点
   - 避开翼型内部区域
   - 包含三层外围点（距离翼型3、6、9个单位）

2. **流场数据提取** (`output()`)
   - 对每个采样点提取5个物理量：
     - `x, y`: 采样点坐标 (2个值)
     - `u, v`: 速度分量 (2个值)  
     - `vorticity`: 涡度 (1个值)

3. **数据格式**
   ```python
   # 返回形状为 (642, 5) 的numpy数组
   output_data = [
       [x1, y1, u1, v1, vort1],
       [x2, y2, u2, v2, vort2],
       ...
       [x642, y642, u642, v642, vort642]
   ]
   ```

#### 关键方法说明：
- `calculate_vorticity()`: 计算整个流场的涡度
- `is_inside_airfoil()`: 判断点是否在翼型内部
- `generate_outer_points_by_normal()`: 生成翼型外围采样点
- `select_sampling_points()`: 综合选择所有采样点

---

## 2. 动作空间 (Action Space) 对应关系

### 2.1 Gym环境中的定义
```python
# 在 lbm_env_structure_modified2.py 中
self.action_space = spaces.Box(
    low=np.array([-15.0], dtype=np.float32),
    high=np.array([15.0], dtype=np.float32),
    dtype=np.float32
)
```

### 2.2 对应的工具模块
**文件**: `tools/control_tools/movable_column.py`  
**核心类**: `controller`

### 2.3 详细对应关系

#### 动作执行流程：
1. **动作接收** (在gym环境的`step()`方法中)
   ```python
   control_val = float(np.clip(action[0], -15.0, 15.0))
   self.lbm.control(np.array([control_val, 0.0], dtype=np.float32))
   ```

2. **控制器响应** (`movable_column.py`中的`control()`方法)
   ```python
   def control(self, choice):
       w = 0.1 * (choice - 9) / 10 / self.controller_radius
       # 计算圆柱表面各点的切向速度
       self.x_vel = -w * self.controller_radius * np.sin(theta)
       self.y_vel = w * self.controller_radius * np.cos(theta)
   ```

#### 动作含义：
- **输入范围**: [-15.0, 15.0]
- **物理含义**: 控制圆柱的旋转速度
  - 正值：顺时针旋转
  - 负值：逆时针旋转
  - 数值大小：旋转速度强度
  - choice的范围可以是-15 - +15 （没那么死，可以看着办）

#### 控制机制：
1. **圆柱几何**: 由`column_create()`生成圆柱边界点
2. **速度计算**: 根据动作值计算圆柱表面各点的切向速度
3. **流场影响**: 通过边界条件影响周围流场

---

## 3. 完整的强化学习循环

### 3.1 环境初始化
```python
env = LBMEnv(config={"render_mode": "human"})
obs, info = env.reset()  # obs来自output.py的数据处理
```

### 3.2 动作-观察循环
```python
for step in range(episodes):
    action = agent.select_action(obs)  # 智能体选择动作 [-15.0, 15.0]
    
    # 执行动作 (使用movable_column.py的控制逻辑)
    obs, reward, done, info = env.step(action)
    
    # 新观察 (来自output.py的642×5数据)
    # obs包含642个采样点的[x,y,u,v,vorticity]信息
```

### 3.3 奖励计算
```python
cd, cl = self.lbm.calculate_drag_lift()
reward = cl - 0.5 * cd  # 最大化升力，最小化阻力
```

---

## 4. 数据流向图

```
动作空间 [-15.0, 15.0]
        ↓
movable_column.py (控制圆柱旋转)
        ↓
LBM求解器 (流场计算)
        ↓
output.py (采样642个点的流场数据)
        ↓
观察空间 (642, 5) [x,y,u,v,vorticity]
        ↓
强化学习智能体
        ↓
新的动作选择
```

---

## 5. 关键参数对应

| 参数 | Gym环境 | output.py | movable_column.py |
|------|---------|-----------|-------------------|
| 网格尺寸 | nx=400, ny=200 | 用于采样点选择 | 用于控制器定位 |
| 采样点数 | shape=(642, 5) | ~642个有效点 | - |
| 控制范围 | [-15.0, 15.0] | - | choice参数 |
| 物理量 | 5个特征 | [x,y,u,v,vorticity] | [x_vel, y_vel] |

---

## 6. 使用建议

1. **调试观察空间**: 使用`output.py`中的`show_valid_points()`可视化采样点分布
2. **调试动作空间**: 检查`movable_column.py`中的速度计算是否合理
3. **性能优化**: 采样点数量可根据需要调整，影响观察空间维度
4. **控制精度**: 动作范围可根据具体应用调整

---

## 总结

这个LBM强化学习环境实现了一个完整的流体控制系统：
- **观察**: 通过`output.py`提取流场关键信息
- **动作**: 通过`movable_column.py`控制圆柱旋转
- **目标**: 优化升阻比，实现流动控制

三个文件协同工作，构成了一个用于流体力学主动控制研究的强化学习平台。