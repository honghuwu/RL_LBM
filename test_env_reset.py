#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单测试env_lbm.py中reset函数的功能
"""

import numpy as np
import sys
import os

# 添加路径以导入模块
sys.path.append(os.path.join(os.path.dirname(__file__), 'LBM_GYM'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'core'))

from env_lbm import LBMEnv

def test_reset_function():
    """
    测试LBM环境的reset函数
    """
    print("="*60)
    print("测试LBM环境reset函数")
    print("="*60)
    
    # 1. 创建环境
    print("\n1. 创建LBM环境...")
    env = LBMEnv()
    print(f"环境创建成功")
    
    # 1.5. 执行一些初始步骤
    print("\n1.5. 执行一些初始步骤...")
    obs_init, info_init = env.reset()
    print(f"初始化后观测形状: {obs_init.shape}")
    
    for i in range(2):
        action = env.action_space.sample()
        obs_step, reward, done, truncated, info = env.step(action)
        print(f"  初始步骤 {i+1}: reward={reward:.6f}, step_counter={env.step_counter}")
    
    print(f"初始步骤后step_counter: {env.step_counter}")
    
    # 2. 初始重置
    print("\n2. 执行初始重置...")
    obs1, info1 = env.reset()
    print(f"初始观测形状: {obs1.shape}")
    print(f"初始观测范围: [{obs1.min():.6f}, {obs1.max():.6f}]")
    print(f"初始step_counter: {env.step_counter}")
    
    # 3. 运行几步改变状态
    print("\n3. 运行环境几步...")
    for i in range(3):
        action = env.action_space.sample()  # 随机动作
        obs, reward, done, truncated, info = env.step(action)
        print(f"  步骤 {i+1}: reward={reward:.6f}, step_counter={env.step_counter}")
    
    print(f"运行后step_counter: {env.step_counter}")
    print(f"运行后观测范围: [{obs.min():.6f}, {obs.max():.6f}]")
    
    # 4. 再次重置
    print("\n4. 再次重置环境...")
    obs2, info2 = env.reset()
    print(f"重置后观测形状: {obs2.shape}")
    print(f"重置后观测范围: [{obs2.min():.6f}, {obs2.max():.6f}]")
    print(f"重置后step_counter: {env.step_counter}")
    
    # 5. 验证重置效果
    print("\n5. 验证重置效果...")
    
    # 检查step_counter是否重置
    step_counter_reset = env.step_counter == 0
    print(f"  step_counter重置: {'✓' if step_counter_reset else '✗'}")
    
    # 检查观测形状是否一致
    shape_consistent = obs1.shape == obs2.shape
    print(f"  观测形状一致: {'✓' if shape_consistent else '✗'}")
    
    # 检查观测数值是否合理
    obs_valid = np.isfinite(obs2).all() and not np.isnan(obs2).any()
    print(f"  观测数值有效: {'✓' if obs_valid else '✗'}")
    
    # 6. 测试多次重置的一致性
    print("\n6. 测试多次重置一致性...")
    obs_list = []
    for i in range(3):
        obs_reset, _ = env.reset()
        obs_list.append(obs_reset)
        print(f"  重置 {i+1}: 观测均值={obs_reset.mean():.6f}")
    
    # 检查多次重置的一致性
    consistent = True
    for i in range(1, len(obs_list)):
        if not np.allclose(obs_list[0], obs_list[i], rtol=1e-5):
            consistent = False
            break
    
    print(f"  多次重置一致性: {'✓' if consistent else '✗'}")
    
    # 7. 总结
    print("\n" + "="*60)
    all_tests_passed = all([
        step_counter_reset,
        shape_consistent, 
        obs_valid,
        consistent
    ])
    
    if all_tests_passed:
        print("🎉 所有测试通过！reset函数工作正常。")
    else:
        print("⚠️  部分测试失败，请检查reset函数实现。")
    print("="*60)
    
    return all_tests_passed

if __name__ == "__main__":
    try:
        success = test_reset_function()
        exit_code = 0 if success else 1
        exit(exit_code)
    except Exception as e:
        print(f"\n❌ 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        exit(1)