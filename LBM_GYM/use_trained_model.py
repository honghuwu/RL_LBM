#!/usr/bin/env python3
"""
使用训练好的PPO模型进行预测和评估
展示如何加载模型、进行推理、评估性能和可视化结果
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from stable_baselines3 import PPO
from gymnasium.wrappers import TimeLimit
from env_lbm import LBMEnv

class ModelEvaluator:
    """训练好的模型评估器"""
    
    def __init__(self, model_path, env_config=None):
        """
        初始化模型评估器
        
        Args:
            model_path: 训练好的模型路径
            env_config: 环境配置
        """
        self.model_path = model_path
        self.env_config = env_config or {"max_episode_steps": 200}
        
        # 加载模型和环境
        self.load_model_and_env()
        
        # 存储评估结果
        self.evaluation_results = {
            'rewards': [],
            'episode_lengths': [],
            'cd_values': [],
            'cl_values': [],
            'actions': [],
            'observations': []
        }
    
    def load_model_and_env(self):
        """加载训练好的模型和环境"""
        try:
            print("🔄 加载训练好的模型...")
            
            # 创建环境
            base_env = LBMEnv(config=self.env_config)
            self.env = TimeLimit(base_env, max_episode_steps=self.env_config["max_episode_steps"])
            
            # 加载模型
            self.model = PPO.load(self.model_path, env=self.env)
            
            print(f"✅ 模型加载成功: {self.model_path}")
            print(f"📋 环境信息:")
            print(f"   - 动作空间: {self.env.action_space}")
            print(f"   - 观测空间: {self.env.observation_space.shape}")
            print(f"   - 最大episode长度: {self.env_config['max_episode_steps']}")
            
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            raise
    
    def predict_single_action(self, observation, deterministic=True):
        """
        对单个观测进行预测
        
        Args:
            observation: 环境观测
            deterministic: 是否使用确定性策略
            
        Returns:
            action: 预测的动作
            action_prob: 动作概率（如果可用）
        """
        action, _states = self.model.predict(observation, deterministic=deterministic)
        return action
    
    def run_single_episode(self, render=False, deterministic=True, verbose=True):
        """
        运行单个episode
        
        Args:
            render: 是否渲染
            deterministic: 是否使用确定性策略
            verbose: 是否打印详细信息
            
        Returns:
            episode_data: episode的详细数据
        """
        obs, info = self.env.reset()
        
        episode_data = {
            'observations': [obs.copy()],
            'actions': [],
            'rewards': [],
            'infos': [info.copy()],
            'total_reward': 0,
            'episode_length': 0
        }
        
        if verbose:
            print(f"\n🎯 开始新的episode...")
        
        step_count = 0
        while True:
            # 预测动作
            action = self.predict_single_action(obs, deterministic=deterministic)
            
            # 执行动作
            obs, reward, terminated, truncated, info = self.env.step(action)
            
            # 记录数据
            episode_data['actions'].append(action.copy())
            episode_data['rewards'].append(reward)
            episode_data['observations'].append(obs.copy())
            episode_data['infos'].append(info.copy())
            episode_data['total_reward'] += reward
            episode_data['episode_length'] += 1
            
            step_count += 1
            
            # 打印进度
            if verbose and step_count % 20 == 0:
                cd = info.get('CD', 0)
                cl = info.get('CL', 0)
                print(f"   📊 步数: {step_count} | 动作: {action[0]:.3f} | "
                      f"奖励: {reward:.4f} | CD: {cd:.4f} | CL: {cl:.4f}")
            
            # 渲染
            if render:
                self.env.render()
                time.sleep(0.05)
            
            # 检查episode结束
            if terminated or truncated:
                end_reason = "自然结束" if terminated else "达到最大步数"
                if verbose:
                    print(f"   ✅ Episode结束: {end_reason}")
                    print(f"   📈 总步数: {episode_data['episode_length']}")
                    print(f"   🏆 总奖励: {episode_data['total_reward']:.4f}")
                break
        
        return episode_data
    
    def evaluate_model(self, num_episodes=10, deterministic=True, verbose=True):
        """
        评估模型性能
        
        Args:
            num_episodes: 评估的episode数量
            deterministic: 是否使用确定性策略
            verbose: 是否打印详细信息
        """
        print(f"\n🧪 开始模型评估 ({num_episodes} episodes)...")
        print("=" * 60)
        
        all_rewards = []
        all_lengths = []
        all_cd_values = []
        all_cl_values = []
        
        for episode in range(num_episodes):
            if verbose:
                print(f"\n📊 Episode {episode + 1}/{num_episodes}")
            
            # 运行episode
            episode_data = self.run_single_episode(
                render=False, 
                deterministic=deterministic, 
                verbose=verbose
            )
            
            # 收集统计数据
            all_rewards.append(episode_data['total_reward'])
            all_lengths.append(episode_data['episode_length'])
            
            # 收集物理参数
            cd_values = [info.get('CD', 0) for info in episode_data['infos']]
            cl_values = [info.get('CL', 0) for info in episode_data['infos']]
            
            if cd_values:
                all_cd_values.extend(cd_values)
                all_cl_values.extend(cl_values)
        
        # 计算统计结果
        results = {
            'mean_reward': np.mean(all_rewards),
            'std_reward': np.std(all_rewards),
            'mean_length': np.mean(all_lengths),
            'std_length': np.std(all_lengths),
            'mean_cd': np.mean(all_cd_values) if all_cd_values else 0,
            'mean_cl': np.mean(all_cl_values) if all_cl_values else 0,
            'all_rewards': all_rewards,
            'all_lengths': all_lengths,
            'all_cd_values': all_cd_values,
            'all_cl_values': all_cl_values
        }
        
        # 打印评估结果
        print("\n" + "=" * 60)
        print("📊 评估结果总结:")
        print(f"   🏆 平均奖励: {results['mean_reward']:.4f} ± {results['std_reward']:.4f}")
        print(f"   📏 平均长度: {results['mean_length']:.1f} ± {results['std_length']:.1f}")
        print(f"   🌪️  平均阻力系数(CD): {results['mean_cd']:.4f}")
        print(f"   ⬆️  平均升力系数(CL): {results['mean_cl']:.4f}")
        if results['mean_cd'] > 0:
            print(f"   ⚡ 平均升阻比(CL/CD): {results['mean_cl']/results['mean_cd']:.4f}")
        print("=" * 60)
        
        return results
    
    def visualize_episode(self, render=True, save_data=True):
        """
        可视化单个episode的运行过程
        
        Args:
            render: 是否实时渲染
            save_data: 是否保存数据用于后续分析
        """
        print("\n🎬 开始可视化episode...")
        
        episode_data = self.run_single_episode(render=render, verbose=True)
        
        if save_data:
            # 绘制episode数据
            self.plot_episode_data(episode_data)
        
        return episode_data
    
    def plot_episode_data(self, episode_data):
        """绘制episode数据图表"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Episode 运行数据分析', fontsize=16)
        
        steps = range(len(episode_data['rewards']))
        
        # 奖励曲线
        axes[0, 0].plot(steps, episode_data['rewards'])
        axes[0, 0].set_title('Step Rewards')
        axes[0, 0].set_xlabel('Steps')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].grid(True)
        
        # 动作曲线
        actions = [action[0] for action in episode_data['actions']]
        axes[0, 1].plot(steps[:-1], actions)  # actions比observations少一个
        axes[0, 1].set_title('Actions')
        axes[0, 1].set_xlabel('Steps')
        axes[0, 1].set_ylabel('Action Value')
        axes[0, 1].grid(True)
        
        # CD曲线
        cd_values = [info.get('CD', 0) for info in episode_data['infos']]
        axes[1, 0].plot(steps, cd_values)
        axes[1, 0].set_title('Drag Coefficient (CD)')
        axes[1, 0].set_xlabel('Steps')
        axes[1, 0].set_ylabel('CD')
        axes[1, 0].grid(True)
        
        # CL曲线
        cl_values = [info.get('CL', 0) for info in episode_data['infos']]
        axes[1, 1].plot(steps, cl_values)
        axes[1, 1].set_title('Lift Coefficient (CL)')
        axes[1, 1].set_xlabel('Steps')
        axes[1, 1].set_ylabel('CL')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig('./models/episode_analysis.png', dpi=300, bbox_inches='tight')
        print("📈 Episode分析图已保存至 ./models/episode_analysis.png")
        plt.show()
    
    def close(self):
        """关闭环境"""
        if hasattr(self, 'env'):
            self.env.close()

def main():
    """主函数：演示如何使用训练好的模型"""
    
    # 模型路径（根据实际情况修改）
    model_paths = [
        "./models/final_model.zip",      # 最终模型
        "./models/best_model/best_model.zip"  # 最佳模型
    ]
    
    # 选择可用的模型
    model_path = None
    for path in model_paths:
        try:
            import os
            if os.path.exists(path):
                model_path = path
                break
        except:
            continue
    
    if model_path is None:
        print("❌ 未找到训练好的模型文件！")
        print("💡 请确保以下文件之一存在：")
        for path in model_paths:
            print(f"   - {path}")
        print("\n🔧 如果还没有训练模型，请先运行: python train_ppo.py")
        return
    
    try:
        # 创建模型评估器
        evaluator = ModelEvaluator(model_path)
        
        print("\n" + "=" * 60)
        print("🎯 模型使用选项:")
        print("1. 快速评估 (5 episodes)")
        print("2. 详细评估 (10 episodes)")
        print("3. 可视化运行 (1 episode with rendering)")
        print("4. 单步预测演示")
        print("=" * 60)
        
        choice = input("请选择操作 (1-4): ").strip()
        
        if choice == "1":
            # 快速评估
            results = evaluator.evaluate_model(num_episodes=5, verbose=False)
            
        elif choice == "2":
            # 详细评估
            results = evaluator.evaluate_model(num_episodes=10, verbose=True)
            
        elif choice == "3":
            # 可视化运行
            episode_data = evaluator.visualize_episode(render=True, save_data=True)
            
        elif choice == "4":
            # 单步预测演示
            print("\n🔍 单步预测演示...")
            obs, info = evaluator.env.reset()
            print(f"📊 初始观测形状: {obs.shape}")
            
            for i in range(5):
                action = evaluator.predict_single_action(obs)
                print(f"   步骤 {i+1}: 预测动作 = {action[0]:.4f}")
                obs, reward, terminated, truncated, info = evaluator.env.step(action)
                print(f"           奖励 = {reward:.4f}, CD = {info.get('CD', 0):.4f}, CL = {info.get('CL', 0):.4f}")
                if terminated or truncated:
                    break
        
        else:
            print("❌ 无效选择！")
        
        # 关闭环境
        evaluator.close()
        
    except Exception as e:
        print(f"❌ 运行出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()