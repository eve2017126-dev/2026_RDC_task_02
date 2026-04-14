import pandas as pd
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_preprocessing import preprocess_data
from src.feature_engineering import prepare_features, create_feature_pipeline
from src.model import compare_models, save_model, evaluate_model, create_linear_model
from sklearn.model_selection import train_test_split

def train_laptop_price_model():
    """训练笔记本电脑价格预测模型"""
    print("🚀 开始训练笔记本电脑价格预测模型")
    
    # 1. 数据预处理
    # 使用绝对路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    train_path = os.path.join(project_root, "data", "raw", "laptop_prices_train.csv")
    test_path = os.path.join(project_root, "data", "raw", "laptop_prices_test.csv")
    
    train_data, test_data = preprocess_data(train_path, test_path)
    
    # 2. 特征工程
    X_train, y_train, X_test, train_features, test_features = prepare_features(train_data, test_data)
    
    # 由于测试集没有价格标签，我们需要从训练集分割验证集
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    print(f"训练集大小: {X_train_split.shape}")
    print(f"验证集大小: {X_val.shape}")
    
    # 3. 创建特征预处理pipeline
    preprocessor = create_feature_pipeline()
    
    # 4. 应用预处理
    X_train_processed = preprocessor.fit_transform(X_train_split)
    X_val_processed = preprocessor.transform(X_val)
    
    print(f"预处理后训练集维度: {X_train_processed.shape}")
    print(f"预处理后验证集维度: {X_val_processed.shape}")
    
    # 5. 模型训练和比较
    print("\n📊 开始模型比较...")
    best_model, results = compare_models(X_train_processed, y_train_split, X_val_processed, y_val)
    
    # 6. 使用最佳模型在整个训练集上训练
    print("\n🎯 使用最佳模型在整个训练集上训练...")
    X_full_processed = preprocessor.fit_transform(X_train)
    best_model.fit(X_full_processed, y_train)
    
    # 7. 最终评估
    y_pred_full = best_model.predict(X_full_processed)
    final_r2 = np.corrcoef(y_train, y_pred_full)[0, 1] ** 2
    final_rmse = np.sqrt(np.mean((y_train - y_pred_full) ** 2))
    
    print(f"\n🎉 最终模型性能:")
    print(f"R²: {final_r2:.4f}")
    print(f"RMSE: {final_rmse:.2f}")
    
    # 8. 保存模型和预处理器
    model_dir = os.path.join(project_root, "models")
    os.makedirs(model_dir, exist_ok=True)
    
    save_model(best_model, os.path.join(model_dir, "best_model.pkl"))
    save_model(preprocessor, os.path.join(model_dir, "preprocessor.pkl"))
    
    # 9. 保存特征重要性
    if hasattr(best_model, 'feature_importances_') or hasattr(best_model, 'coef_'):
        feature_names = preprocessor.get_feature_names_out()
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': best_model.feature_importances_ if hasattr(best_model, 'feature_importances_') 
                         else abs(best_model.coef_)
        }).sort_values('importance', ascending=False)
        
        importance_path = os.path.join(model_dir, "feature_importance.csv")
        importance_df.to_csv(importance_path, index=False)
        print(f"特征重要性已保存到: {importance_path}")
    
    print("\n✅ 模型训练完成!")
    
    return best_model, preprocessor, results

def train_simple_linear_model():
    """训练简单的线性回归模型（用于快速测试）"""
    print("🔧 训练简单线性回归模型...")
    
    # 数据预处理
    train_path = "../data/raw/laptop_prices_train.csv"
    test_path = "../data/raw/laptop_prices_test.csv"
    
    train_data, test_data = preprocess_data(train_path, test_path)
    
    # 特征工程
    X_train, y_train, X_test, train_features, test_features = prepare_features(train_data, test_data)
    
    # 创建预处理pipeline
    preprocessor = create_feature_pipeline()
    
    # 应用预处理
    X_processed = preprocessor.fit_transform(X_train)
    
    # 训练线性回归模型
    model = create_linear_model()
    model.fit(X_processed, y_train)
    
    # 评估
    y_pred = model.predict(X_processed)
    r2 = np.corrcoef(y_train, y_pred)[0, 1] ** 2
    
    print(f"简单线性回归模型 R²: {r2:.4f}")
    
    # 保存模型
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    model_dir = os.path.join(project_root, "models")
    os.makedirs(model_dir, exist_ok=True)
    
    save_model(model, os.path.join(model_dir, "linear_model.pkl"))
    save_model(preprocessor, os.path.join(model_dir, "linear_preprocessor.pkl"))
    
    return model, preprocessor

if __name__ == "__main__":
    # 训练模型
    train_laptop_price_model()