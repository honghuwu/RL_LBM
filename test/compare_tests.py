#!/usr/bin/env python3
"""
对比测试：使用模型 vs 不使用模型
同时运行两个测试并对比结果
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

from stable_baselines3 import PPO
from gymnasium.wrappers import TimeLimit

# 添加PPO目录到Python路径以导入env_lbm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'PPO')))
from env_lbm import LBMEnv

def run_comparison_test(total_steps=5000):
    """
    运行对比测试
    
    Args:
        total_steps: 每个测试的总步数
    """

    model_paths = [
        "../../models/final_model.zip",
        "../models/final_model.zip",
        "./models/final_model.zip",
    ]
    
    model_path = None

    for path in model_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    if model_path is None:
        print("未找到训练好的模型文件！")
        return
    
    # 测试1: 使用模型
    print("\n测试1: 使用训练好的模型")
    print("-" * 40)
    
    base_env1 = LBMEnv(config={"max_episode_steps": 200})
    env1 = TimeLimit(base_env1, max_episode_steps=200)
    model = PPO.load(model_path, env=env1)
    
    model_rewards = []
    obs1, info1 = env1.reset()
    current_step1 = 0
    
    while current_step1 < total_steps:
        action, _states = model.predict(obs1, deterministic=True)
        obs1, reward, terminated, truncated, info1 = env1.step(action)
        model_rewards.append(reward)
        current_step1 += 1
        
        # 实时进度条显示
        if current_step1 % 100 == 0 or current_step1 == total_steps:
            progress = current_step1 / total_steps
            bar_length = 50
            filled_length = int(bar_length * progress)
            bar = '█' * filled_length + '░' * (bar_length - filled_length)
            recent_avg = np.mean(model_rewards[-1000:]) if len(model_rewards) >= 1000 else np.mean(model_rewards)
            print(f"\r模型测试 [{bar}] {current_step1}/{total_steps} ({progress:.1%}) | 平均奖励: {recent_avg:.4f}", end='', flush=True)
        
        if terminated or truncated:
            obs1, info1 = env1.reset()
    
    print()  # 换行，确保进度条显示完整
    env1.close()
    
    # 测试2: 无动作
    print("\n⏸ 测试2: 无动作")
    print("-" * 40)
    
    base_env2 = LBMEnv(config={"max_episode_steps": 200})
    env2 = TimeLimit(base_env2, max_episode_steps=200)
    
    no_action_rewards = []
    obs2, info2 = env2.reset()
    current_step2 = 0
    
    # 无动作（动作空间中心值，通常为0）
    no_action = np.array([0])
    
    while current_step2 < total_steps:
        obs2, reward, terminated, truncated, info2 = env2.step(no_action)
        no_action_rewards.append(reward)
        current_step2 += 1
        
        # 实时进度条显示
        if current_step2 % 100 == 0 or current_step2 == total_steps:
            progress = current_step2 / total_steps
            bar_length = 50
            filled_length = int(bar_length * progress)
            bar = '█' * filled_length + '░' * (bar_length - filled_length)
            recent_avg = np.mean(no_action_rewards[-1000:]) if len(no_action_rewards) >= 1000 else np.mean(no_action_rewards)
            print(f"\r无动作测试 [{bar}] {current_step2}/{total_steps} ({progress:.1%}) | 平均奖励: {recent_avg:.4f}", end='', flush=True)
        
        if terminated or truncated:
            obs2, info2 = env2.reset()
    
    print()  # 换行，确保进度条显示完整
    env2.close()
    
    # 计算统计结果
    model_mean = np.mean(model_rewards)
    model_std = np.std(model_rewards)
    no_action_mean = np.mean(no_action_rewards)
    no_action_std = np.std(no_action_rewards)
    
    improvement = ((model_mean - no_action_mean) / abs(no_action_mean)) * 100 if no_action_mean != 0 else 0
    
    # 打印对比结果
    print("\n" + "=" * 80)
    print("📊 对比测试结果")
    print("=" * 80)
    print(f"🤖 使用模型:")
    print(f"   🏆 平均奖励: {model_mean:.6f} ± {model_std:.6f}")
    print(f"   📈 最大奖励: {np.max(model_rewards):.6f}")
    print(f"   📉 最小奖励: {np.min(model_rewards):.6f}")
    print()
    print(f"⏸️ 无动作:")
    print(f"   🏆 平均奖励: {no_action_mean:.6f} ± {no_action_std:.6f}")
    print(f"   📈 最大奖励: {np.max(no_action_rewards):.6f}")
    print(f"   📉 最小奖励: {np.min(no_action_rewards):.6f}")
    print()
    print(f"📈 性能提升: {improvement:+.2f}%")
    if improvement > 0:
        print("✅ 模型表现优于无动作")
    elif improvement < 0:
        print("⚠️ 模型表现不如无动作")
    else:
        print("➖ 模型表现与无动作相当")
    print("=" * 80)
    
    # 绘制对比图
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('模型 vs 无动作 - 性能对比', fontsize=16)
    
    # 奖励曲线对比
    axes[0, 0].plot(model_rewards, alpha=0.7, linewidth=0.5, label='使用模型', color='blue')
    axes[0, 0].plot(no_action_rewards, alpha=0.7, linewidth=0.5, label='无动作', color='orange')
    axes[0, 0].set_title('步奖励曲线对比')
    axes[0, 0].set_xlabel('步数')
    axes[0, 0].set_ylabel('奖励')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 滑动平均对比
    window_size = 100
    if len(model_rewards) >= window_size:
        model_moving_avg = np.convolve(model_rewards, np.ones(window_size)/window_size, mode='valid')
        no_action_moving_avg = np.convolve(no_action_rewards, np.ones(window_size)/window_size, mode='valid')
        
        axes[0, 1].plot(range(window_size-1, len(model_rewards)), model_moving_avg, 
                       'b-', linewidth=2, label='使用模型')
        axes[0, 1].plot(range(window_size-1, len(no_action_rewards)), no_action_moving_avg, 
                       'orange', linewidth=2, label='无动作')
        axes[0, 1].set_title(f'{window_size}步滑动平均对比')
    else:
        axes[0, 1].plot(model_rewards, 'b-', linewidth=2, label='使用模型')
        axes[0, 1].plot(no_action_rewards, 'orange', linewidth=2, label='无动作')
        axes[0, 1].set_title('奖励曲线对比')
    
    axes[0, 1].set_xlabel('步数')
    axes[0, 1].set_ylabel('平均奖励')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 奖励分布直方图
    axes[1, 0].hist(model_rewards, bins=50, alpha=0.7, label='使用模型', color='blue', density=True)
    axes[1, 0].hist(no_action_rewards, bins=50, alpha=0.7, label='无动作', color='orange', density=True)
    axes[1, 0].set_title('奖励分布对比')
    axes[1, 0].set_xlabel('奖励值')
    axes[1, 0].set_ylabel('密度')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 统计对比柱状图
    categories = ['平均奖励', '标准差', '最大奖励', '最小奖励']
    model_stats = [model_mean, model_std, np.max(model_rewards), np.min(model_rewards)]
    no_action_stats = [no_action_mean, no_action_std, np.max(no_action_rewards), np.min(no_action_rewards)]
    
    x = np.arange(len(categories))
    width = 0.35
    
    axes[1, 1].bar(x - width/2, model_stats, width, label='使用模型', color='blue', alpha=0.7)
    axes[1, 1].bar(x + width/2, no_action_stats, width, label='无动作', color='orange', alpha=0.7)
    axes[1, 1].set_title('统计指标对比')
    axes[1, 1].set_xlabel('统计指标')
    axes[1, 1].set_ylabel('数值')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(categories, rotation=45)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('./models/comparison_results.png', dpi=300, bbox_inches='tight')
    print("📈 对比结果图已保存至 ./models/comparison_results.png")
    plt.show()
    
    return {
        'model_results': {
            'mean_reward': model_mean,
            'std_reward': model_std,
            'step_rewards': model_rewards
        },
        'no_action_results': {
            'mean_reward': no_action_mean,
            'std_reward': no_action_std,
            'step_rewards': no_action_rewards
        },
        'improvement_percentage': improvement
    }

if __name__ == "__main__":
    try:
        print("LBM环境性能对比测试")
        print("比较训练模型与无动作的性能差异")
        
        steps = 5000
        print(f"使用步数: {steps}")
        
        results = run_comparison_test(total_steps=steps)
        
        if results:
            print(f"\n对比测试完成！")
            print(f"模型性能提升: {results['improvement_percentage']:+.2f}%")
        
    except Exception as e:
        print(f"对比测试出错: {e}")
        import traceback
        traceback.print_exc()