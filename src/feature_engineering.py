import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def create_advanced_features(data):
    """创建高级特征工程"""
    data_advanced = data.copy()
    
    # 品牌溢价特征
    premium_brands = ['Apple', 'Razer', 'Microsoft', 'Dell', 'HP']
    data_advanced['Premium_Brand'] = data_advanced['Brand'].isin(premium_brands).astype(int)
    
    # 处理器性能分级
    def get_processor_tier(processor):
        if 'i9' in processor or 'Ryzen 9' in processor:
            return 4  # 旗舰级
        elif 'i7' in processor or 'Ryzen 7' in processor:
            return 3  # 高性能
        elif 'i5' in processor or 'Ryzen 5' in processor:
            return 2  # 主流级
        elif 'i3' in processor or 'Ryzen 3' in processor:
            return 1  # 入门级
        else:
            return 0  # 其他
    
    data_advanced['Processor_Tier'] = data_advanced['Processor'].apply(get_processor_tier)
    
    # GPU性能分级
    def get_gpu_tier(gpu):
        gpu = str(gpu).lower()
        if 'rtx 3080' in gpu or 'rx 6800' in gpu:
            return 4  # 旗舰级
        elif 'rtx 3060' in gpu or 'rx 6600' in gpu:
            return 3  # 高性能
        elif 'gtx' in gpu:
            return 2  # 主流级
        elif 'integrated' in gpu:
            return 1  # 集成显卡
        else:
            return 0  # 其他
    
    data_advanced['GPU_Tier'] = data_advanced['GPU'].apply(get_gpu_tier)
    
    # 存储类型编码
    storage_type_mapping = {'SSD': 2, 'HDD': 1, None: 0}
    data_advanced['Storage_Type_Encoded'] = data_advanced['Storage_Type'].map(storage_type_mapping).fillna(0)
    
    # 屏幕质量特征
    data_advanced['Screen_Quality'] = data_advanced['Resolution_Pixels'] / (data_advanced['Screen Size (inch)'] ** 2)
    
    # 便携性指数
    data_advanced['Portability_Index'] = (
        data_advanced['Battery Life (hours)'] / data_advanced['Weight (kg)']
    )
    
    # 性能综合评分
    data_advanced['Performance_Score'] = (
        data_advanced['Processor_Tier'] * 0.3 +
        data_advanced['RAM_GB'] * 0.1 +
        data_advanced['GPU_Tier'] * 0.3 +
        data_advanced['Storage_Type_Encoded'] * 0.2 +
        data_advanced['Screen_Quality'] * 0.1
    )
    
    # 价格分段特征
    def get_price_segment(price):
        if price < 1000:
            return 'Budget'
        elif price < 2000:
            return 'Mid-range'
        elif price < 3000:
            return 'Premium'
        else:
            return 'Luxury'
    
    if 'Price ($)' in data_advanced.columns:
        data_advanced['Price_Segment'] = data_advanced['Price ($)'].apply(get_price_segment)
    
    return data_advanced

def get_feature_columns():
    """定义最终使用的特征列"""
    numeric_features = [
        'RAM_GB', 'Storage_Capacity', 'Screen Size (inch)', 
        'Battery Life (hours)', 'Weight (kg)', 'Resolution_Width',
        'Resolution_Height', 'Resolution_Pixels', 'Brand_Frequency',
        'Processor_Rank', 'GPU_Type', 'OS_Encoded', 'RAM_Storage_Ratio',
        'Screen_Pixel_Density', 'Battery_Weight_Ratio', 'Premium_Brand',
        'Processor_Tier', 'GPU_Tier', 'Storage_Type_Encoded', 'Screen_Quality',
        'Portability_Index', 'Performance_Score'
    ]
    
    categorical_features = ['Brand', 'Processor', 'Operating System']
    
    return numeric_features, categorical_features

def create_feature_pipeline():
    """创建特征工程pipeline"""
    numeric_features, categorical_features = get_feature_columns()
    
    # 数值特征预处理
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    # 分类特征预处理
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # 组合预处理步骤
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    return preprocessor

def prepare_features(train_data, test_data):
    """准备训练和测试特征"""
    print("=== 开始特征工程 ===")
    
    # 应用高级特征工程
    train_features = create_advanced_features(train_data)
    test_features = create_advanced_features(test_data)
    
    # 获取特征列
    numeric_features, categorical_features = get_feature_columns()
    
    # 准备X和y
    if 'Price ($)' in train_features.columns:
        X_train = train_features[numeric_features + categorical_features]
        y_train = train_features['Price ($)']
        X_test = test_features[numeric_features + categorical_features]
        y_test = None  # 测试集没有价格标签
    else:
        X_train = None
        y_train = None
        X_test = test_features[numeric_features + categorical_features]
    
    print(f"训练特征维度: {X_train.shape if X_train is not None else 'N/A'}")
    print(f"测试特征维度: {X_test.shape}")
    print("=== 特征工程完成 ===\n")
    
    return X_train, y_train, X_test, train_features, test_features

if __name__ == "__main__":
    # 测试特征工程
    import sys
    sys.path.append('..')
    from data_preprocessing import preprocess_data
    
    train_path = "../data/raw/laptop_prices_train.csv"
    test_path = "../data/raw/laptop_prices_test.csv"
    
    train_data, test_data = preprocess_data(train_path, test_path)
    X_train, y_train, X_test, train_features, test_features = prepare_features(train_data, test_data)
    
    print("特征工程后的列名:")
    print(train_features.columns.tolist())