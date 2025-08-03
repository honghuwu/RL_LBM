# LBM可视化模块使用说明

## 概述

我已经将 `lbm2 copy.py` 文件中的可视化相关方法封装到了独立的 `show_tools` 模块中，实现了代码的模块化和可重用性。



## 主要改进

### 1. 模块化设计
- 将所有可视化相关的方法提取到独立的 `LBMVisualizer` 类中
- 保持了原有的功能，但代码结构更清晰
- 便于维护和扩展

### 2. 封装的可视化方法

#### 核心计算方法：
- `compute_vorticity()` - 计算涡度场
- `compute_velocity_magnitude()` - 计算速度大小
- `normalize_field()` - 场数据归一化

#### 颜色映射方法：
- `apply_colormap_vorticity()` - 涡度场颜色映射
- `apply_colormap_velocity()` - 速度场颜色映射
- `combine_images_vertical()` - 垂直组合两个图像

#### 高级接口方法：
- `update_visualization()` - 一键更新所有可视化数据
- `get_combined_image()` - 获取组合后的图像
- `draw_boundary_points()` - 绘制边界点
- `draw_info_text()` - 绘制信息文本

### 3. 使用方式

#### 在LBM求解器中的使用：
```python
from show_tools import LBMVisualizer

# 在show()方法中
def show(self):
    self.init()
    
    # 创建可视化器
    visualizer = LBMVisualizer(self.nx, self.ny)
    
    # 创建GUI窗口
    gui = ti.GUI(self.name, res=(self.nx, visualizer.combined_height))
    
    while gui.running:
        # 执行LBM步骤
        for _ in range(5):
            self.step()
            self.it[None] += 1
        
        # 更新可视化（一行代码完成所有可视化更新）
        visualizer.update_visualization(self.vel)
        
        # 显示图像
        gui.set_image(visualizer.get_combined_image())
        
        # 绘制边界点和信息
        visualizer.draw_boundary_points(gui, self.boundary_pos)
        visualizer.draw_info_text(gui, self.it[None], self.Red, cd, cl)
        
        gui.show()
```

## 优势

### 1. 代码复用性
- 可视化模块可以被其他LBM求解器重用
- 不同的求解器可以使用相同的可视化功能

### 2. 维护性
- 可视化逻辑集中在一个模块中
- 修改可视化效果只需要修改一个文件
- 主求解器代码更简洁，专注于物理计算

### 3. 扩展性
- 容易添加新的可视化功能
- 可以轻松支持不同的颜色映射方案
- 支持自定义可视化参数

### 4. 性能
- 保持了原有的Taichi GPU加速
- 所有计算核心仍然使用 `@ti.kernel` 装饰器
- 没有性能损失

## 自定义选项

### 可视化范围调整：
```python
# 自定义涡度和速度的显示范围
visualizer.update_visualization(
    self.vel, 
    vorticity_range=(-0.05, 0.05),  # 涡度范围
    velocity_range=(0.0, 0.6)       # 速度范围
)
```

### 单独获取图像：
```python
# 只获取涡度场图像
vorticity_img = visualizer.get_vorticity_image()

# 只获取速度场图像
velocity_img = visualizer.get_velocity_image()
```

## 总结

通过这次重构，我们实现了：
1. **代码模块化** - 可视化功能独立封装
2. **接口简化** - 使用更简单的API
3. **功能保持** - 所有原有功能都得到保留
4. **性能维持** - 保持了GPU加速的性能优势

这种设计使得代码更加专业和可维护，符合软件工程的最佳实践。