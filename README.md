# LBM Project - Lattice Boltzmann Method Implementation

## 项目简介

这是一个基于格子玻尔兹曼方法（Lattice Boltzmann Method, LBM）的流体力学仿真项目，使用 Python 和 Taichi 实现高性能的流体仿真计算。

## 项目结构

```
MyCode/
├── core/                    # 核心LBM算法实现
│   ├── lbm.py              # 基础LBM求解器
│   ├── lbm2.py             # LBM求解器版本2
│   ├── lbm3.py             # LBM求解器版本3
│   ├── lbm4.py             # LBM求解器版本4
│   └── LBM3_说明文档.md     # LBM3版本说明文档
├── tools/                   # 工具模块
│   ├── obstacles_generate/  # 障碍物生成工具
│   │   ├── column_generate.py    # 圆柱体生成
│   │   └── naca_genarate.py      # NACA翼型生成
│   ├── output_tool/        # 输出处理工具
│   │   └── output.py       # 结果输出和可视化
│   └── show_tools/         # 可视化工具
│       └── visualizer.py   # 可视化器
├── test/                   # 测试和示例
│   ├── Guo.py             # Guo边界条件测试
│   ├── Dupuis.py          # Dupuis边界条件测试
│   ├── authentic.py       # 验证测试
│   └── momentum_exchange.py # 动量交换测试
├── LBM_GYM/               # 强化学习环境
│   ├── lbm_env_structure.py # 环境结构
│   └── lbm_gym.py         # Gym环境接口
├── config/                # 配置文件
└── LBM.py                 # 主程序入口
```

## 主要功能

- **高性能LBM求解器**: 使用Taichi加速的格子玻尔兹曼方法实现
- **多种边界条件**: 支持Guo、Dupuis等多种边界条件处理方法
- **障碍物生成**: 支持圆柱体、NACA翼型等多种几何形状
- **结果可视化**: 提供流场、涡量、压力等多种可视化功能
- **科学绘图**: 符合科学出版物标准的图表生成
- **强化学习接口**: 提供Gym兼容的环境接口

## 依赖环境

- Python 3.8+
- Taichi >= 1.0.0
- NumPy
- Matplotlib
- SciPy

## 安装和使用

1. 克隆仓库：
```bash
git clone <repository-url>
cd MyCode
```

2. 安装依赖：
```bash
pip install taichi numpy matplotlib scipy
```

3. 运行示例：
```bash
python LBM.py
```

## 示例

### 圆柱绕流仿真
```python
from core.lbm3 import LBMSolver
from tools.obstacles_generate.column_generate import obstacles_generate_cylinder

# 创建求解器
solver = LBMSolver(nx=400, ny=200)

# 生成圆柱障碍物
obstacles_generate_cylinder(solver, center_x=100, center_y=100, radius=20)

# 运行仿真
solver.run(steps=10000)

# 输出结果
solver.output()
```

### NACA翼型仿真
```python
from core.lbm3 import LBMSolver
from tools.obstacles_generate.naca_genarate import obstacles_generate

# 创建求解器
solver = LBMSolver(nx=400, ny=200)

# 生成NACA翼型
obstacles_generate(solver, airfoil_type="0012", chord_length=80)

# 运行仿真
solver.run(steps=10000)

# 输出结果
solver.output()
```

## 贡献

欢迎提交Issue和Pull Request来改进这个项目。

## 许可证

本项目采用MIT许可证。

## 联系方式

如有问题或建议，请通过GitHub Issues联系。




U-net Resnet GoogleNet
LSTM

RL算法   PPO  PPO—plus cl cd cd/cl  