import pandas as pd
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_preprocessing import preprocess_data
from src.feature_engineering import prepare_features
from src.model import load_model

def predict_laptop_prices():
    """预测笔记本电脑价格并生成提交文件"""
    print("开始预测笔记本电脑价格...")
    
    # 1. 加载测试数据
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    train_path = os.path.join(project_root, "data", "raw", "laptop_prices_train.csv")
    test_path = os.path.join(project_root, "data", "raw", "laptop_prices_test.csv")
    
    print("加载数据...")
    train_data, test_data = preprocess_data(train_path, test_path)
    
    # 2. 特征工程
    print("特征工程...")
    X_train, y_train, X_test, train_features, test_features = prepare_features(train_data, test_data)
    
    # 3. 加载模型和预处理器
    print("加载模型...")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    model_dir = os.path.join(project_root, "models")
    
    # 尝试加载最佳模型，如果不存在则加载线性模型
    best_model_path = os.path.join(model_dir, "best_model.pkl")
    linear_model_path = os.path.join(model_dir, "linear_model.pkl")
    
    if os.path.exists(best_model_path):
        model = load_model(best_model_path)
        preprocessor = load_model(os.path.join(model_dir, "preprocessor.pkl"))
        print("加载最佳模型")
    elif os.path.exists(linear_model_path):
        model = load_model(linear_model_path)
        preprocessor = load_model(os.path.join(model_dir, "linear_preprocessor.pkl"))
        print("加载线性回归模型")
    else:
        print("未找到训练好的模型，请先运行训练脚本")
        return None
    
    # 4. 应用预处理
    print("⚙️ 数据预处理...")
    X_test_processed = preprocessor.transform(X_test)
    
    # 5. 预测价格
    print("🔮 进行预测...")
    predictions = model.predict(X_test_processed)
    
    # 6. 确保预测值为正数（价格不能为负）
    predictions = np.maximum(predictions, 0)
    
    print(f"预测价格统计:")
    print(f"最小值: ${predictions.min():.2f}")
    print(f"最大值: ${predictions.max():.2f}")
    print(f"平均值: ${predictions.mean():.2f}")
    print(f"中位数: ${np.median(predictions):.2f}")
    
    # 7. 生成符合Kaggle要求的提交文件
    print("📄 生成符合Kaggle要求的提交文件...")
    submission_dir = os.path.join(project_root, "data", "submission")
    os.makedirs(submission_dir, exist_ok=True)
    
    # 确保行数匹配
    test_data_original = pd.read_csv(test_path)
    expected_rows = len(test_data_original)
    actual_rows = len(predictions)
    
    if expected_rows != actual_rows:
        print(f"⚠️  警告: 预测结果行数({actual_rows})与测试集行数({expected_rows})不匹配!")
        print("正在调整预测结果以匹配测试集行数...")
        
        # 如果预测结果少于测试集，用中位数填充
        if actual_rows < expected_rows:
            median_price = np.median(predictions)
            predictions = np.append(predictions, [median_price] * (expected_rows - actual_rows))
        # 如果预测结果多于测试集，截断
        elif actual_rows > expected_rows:
            predictions = predictions[:expected_rows]
    
    # 创建符合Kaggle要求的提交文件
    submission_df = pd.DataFrame({
        'Id': range(1, len(predictions) + 1),  # 从1开始的连续ID
        'Price': predictions  # 注意：列名必须是'Price'，不能有美元符号
    })
    
    submission_path = os.path.join(submission_dir, "submission.csv")
    submission_df.to_csv(submission_path, index=False)  # 确保index=False
    
    print(f"✅ 提交文件已生成: {submission_path}")
    print(f"📊 提交文件包含 {len(predictions)} 条预测记录")
    print(f"🔍 列名检查: {list(submission_df.columns)}")
    print(f"🔢 ID范围: {submission_df['Id'].min()} - {submission_df['Id'].max()}")
    print(f"💰 价格范围: ${submission_df['Price'].min():.2f} - ${submission_df['Price'].max():.2f}")
    
    # 8. 显示前几条预测结果
    print("\n前10条预测结果:")
    print(submission_df.head(10).to_string(index=False))
    
    return submission_path

def predict_with_custom_model(model_path, preprocessor_path):
    """使用自定义模型进行预测"""
    print(f"使用自定义模型进行预测: {model_path}")
    
    # 加载数据
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    train_path = os.path.join(project_root, "data", "raw", "laptop_prices_train.csv")
    test_path = os.path.join(project_root, "data", "raw", "laptop_prices_test.csv")
    
    train_data, test_data = preprocess_data(train_path, test_path)
    X_train, y_train, X_test, train_features, test_features = prepare_features(train_data, test_data)
    
    # 加载模型
    model = load_model(model_path)
    preprocessor = load_model(preprocessor_path)
    
    # 预测
    X_test_processed = preprocessor.transform(X_test)
    predictions = model.predict(X_test_processed)
    predictions = np.maximum(predictions, 0)
    
    # 生成符合Kaggle要求的提交文件
    submission_dir = os.path.join(project_root, "data", "submission")
    os.makedirs(submission_dir, exist_ok=True)
    
    # 确保行数匹配
    test_data_original = pd.read_csv(test_path)
    expected_rows = len(test_data_original)
    actual_rows = len(predictions)
    
    if expected_rows != actual_rows:
        print(f"⚠️  警告: 预测结果行数({actual_rows})与测试集行数({expected_rows})不匹配!")
        # 调整预测结果以匹配测试集行数
        if actual_rows < expected_rows:
            median_price = np.median(predictions)
            predictions = np.append(predictions, [median_price] * (expected_rows - actual_rows))
        elif actual_rows > expected_rows:
            predictions = predictions[:expected_rows]
    
    submission_df = pd.DataFrame({
        'Id': range(1, len(predictions) + 1),
        'Price': predictions  # 修正列名
    })
    
    custom_submission_path = os.path.join(submission_dir, "custom_submission.csv")
    submission_df.to_csv(custom_submission_path, index=False)
    
    print(f"自定义模型预测完成: {custom_submission_path}")
    
    return custom_submission_path

if __name__ == "__main__":
    # 执行预测
    submission_file = predict_laptop_prices()
    
    if submission_file:
        print(f"\n预测完成！请将文件上传到Kaggle平台:")
        print(f"文件路径: {submission_file}")
        print("\n上传步骤:")
        print("1. 访问: https://www.kaggle.com/t/75b77ad11a424afc8c222be23e6b34f7")
        print("2. 点击右上角'Join Competition'加入比赛")
        print("3. 点击'Submit Predictions'提交文件")
        print("4. 选择刚才生成的submission.csv文件")
        print("5. 等待系统自动评分")
    else:
        print("预测失败，请检查错误信息")