# 训练好的模型使用指南

## 概述

本指南详细说明如何使用通过 `train_ppo.py` 训练好的PPO模型。

## 模型文件位置

训练完成后，模型文件会保存在以下位置：
- **最终模型**: `./models/final_model.zip`
- **最佳模型**: `./models/best_model/best_model.zip`

## 基本使用方法

### 1. 快速测试（推荐）

运行简化的测试脚本：
```bash
python simple_model_usage.py
```

### 2. 完整功能测试

运行完整的模型使用脚本：
```bash
python use_trained_model.py
```

## 代码示例

### 基本模型加载和使用

```python
from stable_baselines3 import PPO
from gymnasium.wrappers import TimeLimit
from env_lbm import LBMEnv

# 1. 创建环境（与训练时相同）
base_env = LBMEnv(config={"max_episode_steps": 200})
env = TimeLimit(base_env, max_episode_steps=200)

# 2. 加载模型
model = PPO.load("./models/final_model.zip", env=env)

# 3. 使用模型
obs, info = env.reset()
action, _states = model.predict(obs, deterministic=True)
obs, reward, terminated, truncated, info = env.step(action)
```

### 运行完整episode

```python
def run_episode(model, env):
    obs, info = env.reset()
    total_reward = 0
    step_count = 0
    
    while True:
        # 预测动作
        action, _states = model.predict(obs, deterministic=True)
        
        # 执行动作
        obs, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward
        step_count += 1
        
        if terminated or truncated:
            break
    
    return total_reward, step_count
```

## 模型输出说明

### 动作空间
- **类型**: 连续动作空间
- **范围**: [-1, 1]
- **含义**: 控制参数（具体含义取决于环境配置）

### 观测空间
- **类型**: 多维数组
- **内容**: 流场状态信息

### 奖励信息
- **reward**: 当前步的奖励值
- **CD**: 阻力系数
- **CL**: 升力系数
- **CL/CD**: 升阻比

## 性能评估

### 单episode测试
```python
total_reward, steps = run_episode(model, env)
average_reward = total_reward / steps
print(f"总奖励: {total_reward:.4f}")
print(f"平均奖励: {average_reward:.4f}")
```

### 多episode评估
```python
def evaluate_model(model, env, num_episodes=10):
    rewards = []
    for i in range(num_episodes):
        total_reward, _ = run_episode(model, env)
        rewards.append(total_reward)
    
    return {
        'mean_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'min_reward': np.min(rewards),
        'max_reward': np.max(rewards)
    }
```

## 模型部署

### 1. 生产环境使用
- 确保环境配置与训练时一致
- 使用 `deterministic=True` 获得稳定的预测结果
- 考虑批量处理以提高效率

### 2. 实时控制
```python
# 实时控制循环
while True:
    obs = get_current_observation()  # 获取当前观测
    action, _ = model.predict(obs, deterministic=True)
    apply_action(action)  # 应用动作
    time.sleep(control_interval)  # 控制频率
```

## 故障排除

### 常见问题

1. **模型文件不存在**
   - 检查训练是否完成
   - 确认文件路径正确

2. **环境配置不匹配**
   - 确保环境参数与训练时相同
   - 检查 `max_episode_steps` 设置

3. **性能不佳**
   - 检查训练是否充分
   - 考虑调整环境参数
   - 尝试不同的模型检查点

### 调试技巧

```python
# 打印模型信息
print(f"模型策略: {model.policy}")
print(f"观测空间: {env.observation_space}")
print(f"动作空间: {env.action_space}")

# 检查预测结果
action, _states = model.predict(obs, deterministic=True)
print(f"预测动作: {action}")
print(f"动作范围: [{env.action_space.low}, {env.action_space.high}]")
```

## 相关文件

- `simple_model_usage.py`: 简化的使用示例
- `use_trained_model.py`: 完整的使用脚本
- `train_ppo.py`: 训练脚本
- `env_lbm.py`: 环境定义

## 下一步

1. 运行 `simple_model_usage.py` 快速测试模型
2. 根据需要修改测试参数
3. 集成到你的应用中
4. 考虑进一步的模型优化