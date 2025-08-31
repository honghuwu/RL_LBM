#!/usr/bin/env python3
"""
不使用模型进行5000步测试（随机动作）
输出平均奖励和奖励曲线图
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from gymnasium.wrappers import TimeLimit

# 添加PPO目录到Python路径以导入env_lbm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'PPO')))
from env_lbm import LBMEnv

def test_without_model(total_steps=5000):
    """
    不使用模型进行测试，使用随机动作
    
    Args:
        total_steps: 总测试步数
    """
    print(f"🧪 开始不使用模型进行 {total_steps} 步测试（随机动作）...")
    
    # 创建环境
    base_env = LBMEnv(config={"max_episode_steps": 200})
    env = TimeLimit(base_env, max_episode_steps=200)
    
    # 存储测试数据
    step_rewards = []
    current_step = 0
    
    obs, info = env.reset()
    
    while current_step < total_steps:
        # 使用随机动作
        action = env.action_space.sample()
        
        # 执行动作
        obs, reward, terminated, truncated, info = env.step(action)
        
        # 记录奖励
        step_rewards.append(reward)
        current_step += 1
        
        # 打印进度
        if current_step % 1000 == 0:
            recent_avg = np.mean(step_rewards[-1000:]) if len(step_rewards) >= 1000 else np.mean(step_rewards)
            print(f"   📊 步数: {current_step}/{total_steps} | 最近1000步平均奖励: {recent_avg:.4f}")
        
        # 如果episode结束，重置环境
        if terminated or truncated:
            obs, info = env.reset()
    
    # 计算统计结果
    mean_reward = np.mean(step_rewards)
    std_reward = np.std(step_rewards)
    
    print("\n" + "=" * 60)
    print("📊 不使用模型测试结果（随机动作）:")
    print(f"   🏆 平均奖励: {mean_reward:.6f} ± {std_reward:.6f}")
    print(f"   📏 总步数: {len(step_rewards)}")
    print(f"   📈 最大奖励: {np.max(step_rewards):.6f}")
    print(f"   📉 最小奖励: {np.min(step_rewards):.6f}")
    print("=" * 60)
    
    # 绘制奖励曲线
    plt.figure(figsize=(12, 6))
    
    # 原始奖励曲线
    plt.subplot(1, 2, 1)
    plt.plot(step_rewards, alpha=0.7, linewidth=0.5, color='orange')
    plt.title('不使用模型（随机动作） - 步奖励曲线')
    plt.xlabel('步数')
    plt.ylabel('奖励')
    plt.grid(True, alpha=0.3)
    
    # 滑动平均奖励曲线
    plt.subplot(1, 2, 2)
    window_size = 100
    if len(step_rewards) >= window_size:
        moving_avg = np.convolve(step_rewards, np.ones(window_size)/window_size, mode='valid')
        plt.plot(range(window_size-1, len(step_rewards)), moving_avg, 'orange', linewidth=2)
        plt.title(f'不使用模型（随机动作） - {window_size}步滑动平均奖励')
    else:
        plt.plot(step_rewards, 'orange', linewidth=2)
        plt.title('不使用模型（随机动作） - 奖励曲线')
    plt.xlabel('步数')
    plt.ylabel('平均奖励')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('./models/test_without_model_results.png', dpi=300, bbox_inches='tight')
    print("📈 测试结果图已保存至 ./models/test_without_model_results.png")
    plt.show()
    
    # 关闭环境
    env.close()
    
    return {
        'mean_reward': mean_reward,
        'std_reward': std_reward,
        'step_rewards': step_rewards,
        'total_steps': len(step_rewards)
    }

if __name__ == "__main__":
    try:
        results = test_without_model(total_steps=5000)
        print(f"\n✅ 测试完成！平均奖励: {results['mean_reward']:.6f}")
    except Exception as e:
        print(f"❌ 测试出错: {e}")
        import traceback
        traceback.print_exc()