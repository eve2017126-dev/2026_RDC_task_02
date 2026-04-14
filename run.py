#!/usr/bin/env python3
"""
笔记本电脑价格预测项目 - 一键运行入口
"""

import os
import sys
import argparse
from src.train import train_laptop_price_model, train_simple_linear_model
from src.predict import predict_laptop_prices

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='笔记本电脑价格预测项目')
    parser.add_argument('--mode', choices=['train', 'predict', 'full'], 
                       default='full', help='运行模式: train(训练), predict(预测), full(完整流程)')
    parser.add_argument('--simple', action='store_true', 
                       help='使用简单的线性回归模型（快速测试）')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("💻 笔记本电脑价格预测项目")
    print("=" * 60)
    
    if args.mode in ['train', 'full']:
        print("\n🚀 开始训练模型...")
        if args.simple:
            train_simple_linear_model()
        else:
            train_laptop_price_model()
    
    if args.mode in ['predict', 'full']:
        print("\n🔮 开始预测...")
        submission_file = predict_laptop_prices()
        
        if submission_file:
            print(f"\n🎉 项目完成！")
            print(f"提交文件: {submission_file}")
            print("\n📋 下一步操作:")
            print("1. 将提交文件上传到Kaggle平台")
            print("2. 查看模型评分和排名")
            print("3. 根据结果优化特征工程")
        else:
            print("❌ 预测失败")
    
    print("\n✅ 程序执行完成！")

if __name__ == "__main__":
    main()