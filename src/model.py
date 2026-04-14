import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import joblib

def create_linear_model():
    """创建线性回归模型"""
    return LinearRegression()

def create_ridge_model():
    """创建Ridge回归模型（L2正则化）"""
    return Ridge(alpha=1.0)

def create_lasso_model():
    """创建Lasso回归模型（L1正则化）"""
    return Lasso(alpha=0.1)

def create_random_forest():
    """创建随机森林模型"""
    return RandomForestRegressor(n_estimators=100, random_state=42)

def create_gradient_boosting():
    """创建梯度提升模型"""
    return GradientBoostingRegressor(n_estimators=100, random_state=42)

def create_svr_model():
    """创建支持向量回归模型"""
    return SVR(kernel='rbf', C=1.0)

def evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    """评估模型性能"""
    # 训练模型
    model.fit(X_train, y_train)
    
    # 预测
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # 计算指标
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    
    # 交叉验证
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    
    print(f"\n=== {model_name} 模型评估 ===")
    print(f"训练集 R²: {train_r2:.4f}")
    print(f"测试集 R²: {test_r2:.4f}")
    print(f"训练集 RMSE: {train_rmse:.2f}")
    print(f"测试集 RMSE: {test_rmse:.2f}")
    print(f"交叉验证 R²: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    return {
        'model': model,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std()
    }

def hyperparameter_tuning(model, param_grid, X_train, y_train):
    """超参数调优"""
    grid_search = GridSearchCV(
        model, param_grid, cv=5, scoring='r2', n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    
    print(f"最佳参数: {grid_search.best_params_}")
    print(f"最佳分数: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

def save_model(model, filepath):
    """保存训练好的模型"""
    joblib.dump(model, filepath)
    print(f"模型已保存到: {filepath}")

def load_model(filepath):
    """加载训练好的模型"""
    model = joblib.load(filepath)
    print(f"模型已从 {filepath} 加载")
    return model

def compare_models(X_train, y_train, X_test, y_test):
    """比较多个模型性能"""
    models = {
        'Linear Regression': create_linear_model(),
        'Ridge Regression': create_ridge_model(),
        'Lasso Regression': create_lasso_model(),
        'Random Forest': create_random_forest(),
        'Gradient Boosting': create_gradient_boosting(),
        'SVR': create_svr_model()
    }
    
    results = {}
    
    for name, model in models.items():
        results[name] = evaluate_model(model, X_train, y_train, X_test, y_test, name)
    
    # 选择最佳模型
    best_model_name = max(results.keys(), key=lambda x: results[x]['test_r2'])
    best_model = results[best_model_name]['model']
    
    print(f"\n🎯 最佳模型: {best_model_name}")
    print(f"测试集 R²: {results[best_model_name]['test_r2']:.4f}")
    
    return best_model, results

def get_feature_importance(model, feature_names):
    """获取特征重要性"""
    if hasattr(model, 'coef_'):
        # 线性模型
        importance = pd.DataFrame({
            'feature': feature_names,
            'importance': abs(model.coef_)
        }).sort_values('importance', ascending=False)
    elif hasattr(model, 'feature_importances_'):
        # 树模型
        importance = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
    else:
        importance = None
    
    return importance

if __name__ == "__main__":
    # 测试模型模块
    print("模型模块测试完成")