#!/usr/bin/env python3
"""
可视化训练好的PPO模型运行过程
实时渲染环境状态
"""

import numpy as np
import time
import os
import sys
from stable_baselines3 import PPO
from gymnasium.wrappers import TimeLimit

# 添加PPO目录到Python路径以导入env_lbm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'PPO')))
from env_lbm import LBMEnv

def visualize_model_performance(num_episodes=3, render_delay=0.1):
    """
    可视化模型性能
    
    Args:
        num_episodes: 可视化的episode数量
        render_delay: 渲染间隔时间（秒）
    """
    # 模型路径
    # 修正路径：从test目录指向PPO目录下的models文件夹
    model_paths = [
        "../models/final_model.zip",
        "../PPO/models/final_model.zip",
        "../PPO/models/best_model/best_model.zip",
        "../PPO/models/ppo_lbm_model.zip",
        "../PPO/final_model.zip",
        "../PPO/best_model.zip"
    ]
    
    # 选择可用的模型
    model_path = None
    for path in model_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    if model_path is None:
        print("❌ 未找到训练好的模型文件！")
        print("💡 请确保以下文件之一存在：")
        for path in model_paths:
            print(f"   - {path}")
        print("\n🔧 将使用随机动作进行可视化演示...")
        use_model = False
    else:
        print(f"🔄 加载模型: {model_path}")
        use_model = True
    
    # 创建环境
    base_env = LBMEnv(config={"max_episode_steps": 200})
    env = TimeLimit(base_env, max_episode_steps=200)
    
    # 加载模型（如果可用）
    if use_model:
        model = PPO.load(model_path, env=env)
        print("✅ 模型加载成功")
    else:
        model = None
        print("⚠️ 使用随机动作进行演示")
    
    print(f"\n🎬 开始可视化演示 ({num_episodes} episodes)...")
    print("💡 提示: 关闭渲染窗口可以停止演示")
    
    try:
        for episode in range(num_episodes):
            print(f"\n📊 Episode {episode + 1}/{num_episodes}")
            
            obs, info = env.reset()
            episode_reward = 0
            step_count = 0
            
            print(f"   🎯 开始新的episode...")
            
            while True:
                # 预测动作
                if use_model:
                    action, _states = model.predict(obs, deterministic=True)
                else:
                    action = env.action_space.sample()
                
                # 执行动作
                obs, reward, terminated, truncated, info = env.step(action)
                
                episode_reward += reward
                step_count += 1
                
                # 渲染环境
                env.render()
                
                # 打印实时信息
                if step_count % 10 == 0:
                    cd = info.get('CD', 0)
                    cl = info.get('CL', 0)
                    action_val = action[0] if hasattr(action, '__len__') else action
                    print(f"     步数: {step_count:3d} | 动作: {action_val:6.3f} | "
                          f"奖励: {reward:7.4f} | CD: {cd:6.4f} | CL: {cl:6.4f}")
                
                # 控制渲染速度
                time.sleep(render_delay)
                
                # 检查episode结束
                if terminated or truncated:
                    end_reason = "自然结束" if terminated else "达到最大步数"
                    print(f"   ✅ Episode结束: {end_reason}")
                    print(f"   📈 总步数: {step_count}")
                    print(f"   🏆 总奖励: {episode_reward:.4f}")
                    print(f"   📊 平均步奖励: {episode_reward/step_count:.4f}")
                    break
            
            # episode间暂停
            if episode < num_episodes - 1:
                print("\n⏸️  3秒后开始下一个episode...")
                time.sleep(3)
    
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断了可视化演示")
    except Exception as e:
        print(f"\n❌ 可视化过程中出错: {e}")
    finally:
        # 关闭环境
        env.close()
        print("\n🔚 可视化演示结束")

def interactive_visualization():
    """
    交互式可视化选项
    """
    print("\n" + "=" * 60)
    print("🎬 LBM环境可视化演示")
    print("=" * 60)
    print("选项:")
    print("1. 快速演示 (1 episode, 快速渲染)")
    print("2. 详细演示 (3 episodes, 正常渲染)")
    print("3. 慢速演示 (1 episode, 慢速渲染)")
    print("4. 自定义设置")
    print("=" * 60)
    
    choice = input("请选择演示模式 (1-4): ").strip()
    
    if choice == "1":
        visualize_model_performance(num_episodes=1, render_delay=0.02)
    elif choice == "2":
        visualize_model_performance(num_episodes=3, render_delay=0.1)
    elif choice == "3":
        visualize_model_performance(num_episodes=1, render_delay=0.3)
    elif choice == "4":
        try:
            episodes = int(input("请输入episode数量 (1-10): "))
            delay = float(input("请输入渲染延迟时间（秒，0.01-1.0): "))
            episodes = max(1, min(10, episodes))
            delay = max(0.01, min(1.0, delay))
            visualize_model_performance(num_episodes=episodes, render_delay=delay)
        except ValueError:
            print("❌ 输入无效，使用默认设置")
            visualize_model_performance(num_episodes=1, render_delay=0.1)
    else:
        print("❌ 无效选择，使用默认设置")
        visualize_model_performance(num_episodes=1, render_delay=0.1)

if __name__ == "__main__":
    try:
        interactive_visualization()
    except Exception as e:
        print(f"❌ 程序出错: {e}")
        import traceback
        traceback.print_exc()