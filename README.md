# 💻 笔记本电脑价格预测项目

## 📋 项目概述

本项目使用机器学习算法预测笔记本电脑价格，重点考察特征工程能力。项目基于Kaggle竞赛数据集，使用sklearn等机器学习库实现线性回归模型。

### 🎯 任务要求
- **调库代码实现线性回归模型**（分值20%）
- **重点考察特征工程**，而非算法实现
- **Kaggle平台打榜**，评价指标为R²

### 📊 数据集特征
- **品牌**：MSI、Razer、Dell、HP、Apple等
- **处理器**：Intel i3/i5/i7/i9、AMD Ryzen 3/5/7/9
- **内存**：4GB、8GB、16GB、32GB、64GB
- **存储**：SSD/HDD，容量从256GB到1TB
- **显卡**：集成显卡、Nvidia、AMD独立显卡
- **屏幕**：尺寸、分辨率
- **电池**：续航时间（4-12小时）
- **重量**：1.2-3.5kg
- **操作系统**：Windows、macOS、Linux、FreeDOS
- **目标变量**：价格（美元）

## 🚀 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 运行完整流程
```bash
python run.py
```

### 3. 单独运行训练或预测
```bash
# 只训练模型
python run.py --mode train

# 只进行预测
python run.py --mode predict

# 使用简单线性回归（快速测试）
python run.py --simple
```

## 📁 项目结构

```
laptop_price_project/
│
├── data/                     # 数据目录
│   ├── raw/                  # 原始数据
│   │   ├── laptop_prices_train.csv
│   │   └── laptop_prices_test.csv
│   ├── processed/            # 处理后的数据
│   └── submission/           # 提交文件
│       └── submission.csv
│
├── notebooks/                # 探索分析
│   └── eda.ipynb
│
├── src/                      # 核心代码
│   ├── data_preprocessing.py # 数据清洗 + 特征提取
│   ├── feature_engineering.py# 高级特征构造
│   ├── model.py              # 模型定义和评估
│   ├── train.py              # 训练流程
│   ├── predict.py            # 预测流程
│   └── utils.py              # 工具函数
│
├── models/                   # 训练好的模型
│   ├── best_model.pkl
│   ├── preprocessor.pkl
│   └── feature_importance.csv
│
├── config/                   # 配置文件
│   └── config.yaml
│
├── run.py                    # 一键运行入口
└── README.md                 # 项目说明
```

## 🔧 技术实现

### 特征工程亮点

#### 1. 基础特征提取
- **数值提取**：从文本中提取内存容量、存储容量、分辨率等
- **分类编码**：品牌频率编码、处理器等级编码、GPU类型编码

#### 2. 高级特征构造
- **性能评分**：综合处理器、内存、显卡、存储的性能评分
- **便携性指数**：电池续航与重量的比值
- **屏幕质量**：像素密度计算
- **品牌溢价**：识别高端品牌

#### 3. 交互特征
- **RAM存储比**：内存与存储容量的比例
- **电池重量比**：续航时间与重量的比例
- **屏幕像素密度**：分辨率与屏幕尺寸的比例

### 模型选择
项目实现了多种回归模型进行比较：
- **线性回归**（基础模型）
- **Ridge回归**（L2正则化）
- **Lasso回归**（L1正则化）
- **随机森林**
- **梯度提升**
- **支持向量回归**

## 📊 模型评估

### 评估指标
- **R²决定系数**：主要评估指标
- **RMSE均方根误差**：误差大小
- **交叉验证**：模型稳定性

### 预期性能
基于特征工程的复杂度，预期R²得分在0.8-0.9之间。

## 🎯 Kaggle打榜指南

### 1. 加入比赛
访问 [Kaggle比赛链接](https://www.kaggle.com/t/75b77ad11a424afc8c222be23e6b34f7)，点击"Join Competition"加入比赛。

### 2. 提交预测
1. 运行项目生成`data/submission/submission.csv`
2. 在Kaggle页面点击"Submit Predictions"
3. 上传生成的CSV文件
4. 等待系统自动评分

### 3. 优化策略
- **特征工程**：尝试不同的特征组合
- **模型调参**：调整超参数优化性能
- **集成学习**：组合多个模型的预测结果

## 🎓 答辩准备

### 技术亮点展示
1. **特征工程深度**：展示了从原始数据到高级特征的完整流程
2. **模型比较**：实现了多种模型的性能对比
3. **工程化实现**：完整的pipeline和模块化设计

### 业务理解
- **价格影响因素**：处理器、显卡、品牌是主要影响因素
- **市场细分**：识别不同价格区间的产品特征
- **实用价值**：为消费者购买和厂商定价提供参考

### 改进方向
1. **更复杂的特征工程**：深度学习特征、时间序列特征
2. **模型集成**：Stacking、Blending等集成方法
3. **自动化调参**：使用Optuna等自动化调参工具

## 🤝 贡献指南

欢迎提交Issue和Pull Request来改进项目！