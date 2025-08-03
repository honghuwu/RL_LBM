# LBM输出模块重构说明

## 概述

本模块将 `lbm3 copy.py` 中第437-626行的输出相关功能拆分到独立的模块中，实现了代码的模块化和解耦。

## 重构内容

### 拆分的功能

从 `lbm3 copy.py` 中拆分出以下功能到 `output.py`：

1. **涡度计算** (`calculate_vorticity`)
   - 使用Taichi kernel计算流场涡度
   - 处理边界条件
   - 计算速度梯度

2. **机翼内部检测** (`is_inside_airfoil`)
   - 使用射线法判断点是否在翼型内部
   - 改进的算法，避免采样点落在机翼内部
   - 提高检测精度

3. **采样点选择** (`select_sampling_points`)
   - 分层密度采样策略
   - 尾流区域重点采样
   - 边界层点捕捉
   - 智能去重和数量控制

4. **数据输出** (`output`)
   - 采样点流场数据提取
   - 数据格式化和展平
   - 统计信息输出

### 新的模块结构

```
tools/output/
├── __init__.py          # 模块初始化
├── output.py            # 核心输出处理器
└── README.md           # 说明文档
```

### 核心类

#### `LBMOutputProcessor`

输出处理器类，负责所有输出相关功能：

```python
class LBMOutputProcessor:
    def __init__(self, solver)
    def calculate_vorticity(self, vorticity)
    def is_inside_airfoil(self, x, y)
    def select_sampling_points(self)
    def output(self, step)
```

#### 工厂函数

```python
def create_output_processor(solver):
    """创建输出处理器实例"""
    return LBMOutputProcessor(solver)
```

## 使用方法

### 在主求解器中使用

```python
from tools.output.output import create_output_processor

class lbm_solver:
    def __init__(self, ...):
        # ... 其他初始化代码
        self.output_processor = None
    
    def initialize_output_processor(self):
        if self.output_processor is None:
            self.output_processor = create_output_processor(self)
    
    def output(self, step):
        self.initialize_output_processor()
        return self.output_processor.output(step)
```

### 直接使用输出处理器

```python
from tools.output.output import LBMOutputProcessor

# 创建处理器
processor = LBMOutputProcessor(solver)

# 获取输出数据
output_data = processor.output(step=100)
```

## 优势

1. **模块化设计**
   - 输出功能独立，便于维护和测试
   - 降低主求解器代码复杂度
   - 提高代码复用性

2. **接口兼容性**
   - 保持原有的 `output()` 方法接口
   - 无需修改现有调用代码
   - 平滑迁移

3. **功能增强**
   - 改进的机翼内部检测算法
   - 更精确的采样点选择
   - 更好的错误处理

4. **可扩展性**
   - 易于添加新的输出格式
   - 支持不同的采样策略
   - 便于集成其他分析工具

## 测试

使用 `test_refactored_output.py` 脚本验证重构后的功能：

```bash
python test_refactored_output.py
```

## 依赖

- `numpy`: 数值计算
- `taichi`: GPU加速计算
- 主LBM求解器实例

## 版本信息

- 重构版本: v1.0
- 基于: lbm3 copy.py (第437-626行)
- 创建日期: 2024
- 兼容性: 完全向后兼容