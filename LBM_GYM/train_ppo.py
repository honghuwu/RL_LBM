import gymnasium as gym
import numpy as np
import time
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.env_util import make_vec_env
from gymnasium.wrappers import TimeLimit
from env_lbm import LBMEnv

"""
训练脚本：使用 PPO 强化学习算法对 lbm_solver 控制器速度进行优化
目标：学会控制 (vx, vy) 提升升阻比 cl / (cd + ε)
"""

class ProgressCallback(BaseCallback):
    """
    自定义回调函数，用于实时显示训练进度和性能指标
    """
    def __init__(self, check_freq: int = 1000, verbose=1):
        super(ProgressCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.start_time = None
        self.episode_rewards = []
        self.episode_lengths = []
        self.drag_coefficients = []
        self.lift_coefficients = []
        
    def _on_training_start(self) -> None:
        self.start_time = time.time()
        print("🚀 开始训练...")
        print("=" * 60)
        
    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # 计算训练进度
            progress = (self.n_calls / self.locals.get('total_timesteps', 100000)) * 100
            elapsed_time = time.time() - self.start_time
            
            # 获取最近的训练统计信息
            if len(self.model.ep_info_buffer) > 0:
                # 获取最近几个episode的平均reward
                recent_episodes = list(self.model.ep_info_buffer)[-10:]  # 最近10个episode
                recent_rewards = [ep.get('r', 0) for ep in recent_episodes]
                recent_lengths = [ep.get('l', 0) for ep in recent_episodes]
                
                avg_reward = np.mean(recent_rewards) if recent_rewards else 0
                avg_length = np.mean(recent_lengths) if recent_lengths else 0
                latest_reward = recent_rewards[-1] if recent_rewards else 0
                
                self.episode_rewards.append(latest_reward)
                self.episode_lengths.append(recent_lengths[-1] if recent_lengths else 0)
                
                # 尝试获取环境信息
                cd, cl = 0, 0
                try:
                    if hasattr(self.training_env, 'get_attr'):
                        env_info = self.training_env.get_attr('last_info')[0]
                        if env_info and isinstance(env_info, dict):
                            cd = env_info.get('CD', 0)
                            cl = env_info.get('CL', 0)
                            self.drag_coefficients.append(cd)
                            self.lift_coefficients.append(cl)
                except:
                    pass
                
                # 打印详细的训练信息
                print(f"🎯 【第 {self.n_calls:,} 步】")
                print(f"   📈 进度: {progress:.1f}% | ⏱️  用时: {elapsed_time:.0f}s")
                print(f"   🏆 最新奖励: {latest_reward:.6f}")
                print(f"   📊 平均奖励(最近10轮): {avg_reward:.6f}")
                print(f"   📏 平均长度: {avg_length:.1f} 步")
                if cd != 0 or cl != 0:
                    print(f"   🌪️  阻力系数(CD): {cd:.6f} | 升力系数(CL): {cl:.6f}")
                    print(f"   ⚡ 升阻比(CL/CD): {cl/cd:.4f}" if cd > 0 else "   ⚡ 升阻比: N/A")
                print("-" * 50)
            else:
                print(f"📊 步数: {self.n_calls:,} | 进度: {progress:.1f}% | "
                      f"时间: {elapsed_time:.0f}s | 等待episode完成...")
                
        return True
    
    def _on_training_end(self) -> None:
        total_time = time.time() - self.start_time
        print("=" * 60)
        print(f"✅ 训练完成！总用时: {total_time:.0f}秒 ({total_time/60:.1f}分钟)")
        
        # 绘制训练曲线
        if len(self.episode_rewards) > 0:
            self.plot_training_curves()
    
    def plot_training_curves(self):
        """绘制训练曲线"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('PPO 训练过程监控', fontsize=16)
        
        # 奖励曲线
        if len(self.episode_rewards) > 0:
            axes[0, 0].plot(self.episode_rewards)
            axes[0, 0].set_title('Episode Rewards')
            axes[0, 0].set_xlabel('Episodes')
            axes[0, 0].set_ylabel('Reward')
            axes[0, 0].grid(True)
        
        # Episode 长度
        if len(self.episode_lengths) > 0:
            axes[0, 1].plot(self.episode_lengths)
            axes[0, 1].set_title('Episode Lengths')
            axes[0, 1].set_xlabel('Episodes')
            axes[0, 1].set_ylabel('Steps')
            axes[0, 1].grid(True)
        
        # 阻力系数
        if len(self.drag_coefficients) > 0:
            axes[1, 0].plot(self.drag_coefficients)
            axes[1, 0].set_title('Drag Coefficient (CD)')
            axes[1, 0].set_xlabel('Episodes')
            axes[1, 0].set_ylabel('CD')
            axes[1, 0].grid(True)
        
        # 升力系数
        if len(self.lift_coefficients) > 0:
            axes[1, 1].plot(self.lift_coefficients)
            axes[1, 1].set_title('Lift Coefficient (CL)')
            axes[1, 1].set_xlabel('Episodes')
            axes[1, 1].set_ylabel('CL')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig('./models/training_curves.png', dpi=300, bbox_inches='tight')
        print("📈 训练曲线已保存至 ./models/training_curves.png")
        plt.show()

def main():
    print("🔧 初始化 LBM 强化学习训练环境...")
    
    # 创建环境实例，设置episode最大长度
    base_env = LBMEnv(config={"max_episode_steps": 200})
    env = TimeLimit(base_env, max_episode_steps=200)  # 确保episode在200步后结束
    print("✅ LBM 环境创建成功 (最大episode长度: 200步)")

    # 配置日志记录器
    new_logger = configure("./models/logs/", ["stdout", "csv", "tensorboard"])
    
    # 创建 PPO 模型
    model = PPO(
        policy="MlpPolicy",          # 使用多层感知机策略网络
        env=env,                     # 使用自定义的LBM环境
        verbose=1,                   # 输出训练信息
        tensorboard_log="./ppo_lbm_tensorboard/",  # Tensorboard 日志目录
        learning_rate=3e-4,          # 学习率
        n_steps=2048,                # 每次更新的步数
        batch_size=64,               # 批次大小
        n_epochs=10,                 # 每次更新的轮数
        gamma=0.99,                  # 折扣因子
        gae_lambda=0.95,             # GAE lambda
        clip_range=0.2,              # PPO clip range
        ent_coef=0.01,               # 熵系数
    )
    
    # 设置自定义日志记录器
    model.set_logger(new_logger)
    print("✅ PPO 模型创建成功")

    # 创建回调函数
    progress_callback = ProgressCallback(check_freq=1000, verbose=1)
    
    eval_callback = EvalCallback(
        env,
        best_model_save_path="./models/best_model/",
        log_path="./models/eval_log/",
        eval_freq=5000,
        deterministic=True,
        render=False,
        verbose=1
    )
    
    # 组合回调函数
    callbacks = [progress_callback, eval_callback]
    
    print("🎯 开始训练...")
    print(f"📋 训练参数:")
    print(f"   - 总步数: 100,000")
    print(f"   - 学习率: 3e-4")
    print(f"   - 批次大小: 64")
    print(f"   - 评估频率: 每 5,000 步")
    print(f"   - 进度更新: 每 1,000 步")
    print("=" * 60)

    # 训练智能体
    model.learn(
        total_timesteps=50000,      # 总训练步数
        callback=callbacks,          # 回调函数列表
        progress_bar=True           # 显示进度条
    )

    # 保存最终模型
    model.save("./models/final_model")
    print("\n✅ 模型训练完成，已保存至 ./models/final_model")
    
    # 启动 TensorBoard 提示
    print("\n📊 查看详细训练日志:")
    print("   1. TensorBoard: tensorboard --logdir=./ppo_lbm_tensorboard")
    print("   2. 训练曲线: ./models/training_curves.png")
    print("   3. CSV 日志: ./models/logs/progress.csv")

if __name__ == '__main__':
    main()
