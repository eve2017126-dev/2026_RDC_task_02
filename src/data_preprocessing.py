import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

def load_data(train_path, test_path):
    """加载训练集和测试集数据"""
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    return train_data, test_data

def basic_cleaning(data):
    """基础数据清洗"""
    # 检查缺失值
    print("缺失值统计:")
    print(data.isnull().sum())
    
    # 检查数据类型
    print("\n数据类型:")
    print(data.dtypes)
    
    # 删除重复行
    data = data.drop_duplicates()
    print(f"删除重复行后数据量: {len(data)}")
    
    return data

def extract_numeric_features(data):
    """提取数值特征"""
    data_clean = data.copy()
    
    # RAM已经是数值类型，直接使用
    data_clean['RAM_GB'] = data_clean['RAM (GB)'].astype(float)
    
    # 从存储中提取容量和类型
    storage_info = data_clean['Storage'].str.extract('(\d+)(GB|TB)\s*(SSD|HDD)')
    data_clean['Storage_Capacity'] = storage_info[0].astype(float)
    data_clean['Storage_Type'] = storage_info[2]
    
    # 如果单位是TB，转换为GB
    tb_mask = data_clean['Storage'].str.contains('TB')
    data_clean.loc[tb_mask, 'Storage_Capacity'] *= 1024
    
    # 从分辨率中提取宽度和高度
    resolution_info = data_clean['Resolution'].str.extract('(\d+)x(\d+)')
    data_clean['Resolution_Width'] = resolution_info[0].astype(float)
    data_clean['Resolution_Height'] = resolution_info[1].astype(float)
    data_clean['Resolution_Pixels'] = data_clean['Resolution_Width'] * data_clean['Resolution_Height']
    
    return data_clean

def encode_categorical_features(data):
    """编码分类特征"""
    data_encoded = data.copy()
    
    # 品牌编码（使用目标编码或频率编码）
    brand_freq = data_encoded['Brand'].value_counts()
    data_encoded['Brand_Frequency'] = data_encoded['Brand'].map(brand_freq)
    
    # 处理器等级编码
    processor_rank = {
        'Intel i3': 1, 'Intel i5': 2, 'Intel i7': 3, 'Intel i9': 4,
        'AMD Ryzen 3': 1, 'AMD Ryzen 5': 2, 'AMD Ryzen 7': 3, 'AMD Ryzen 9': 4
    }
    data_encoded['Processor_Rank'] = data_encoded['Processor'].map(processor_rank).fillna(0)
    
    # GPU类型编码（集成/独立）
    data_encoded['GPU_Type'] = data_encoded['GPU'].apply(
        lambda x: 0 if 'Integrated' in str(x) else 1
    )
    
    # 操作系统编码
    os_encoding = {
        'Windows': 0, 'macOS': 1, 'Linux': 2, 'FreeDOS': 3
    }
    data_encoded['OS_Encoded'] = data_encoded['Operating System'].map(os_encoding)
    
    # 存储类型编码（SSD=1, HDD=0）
    data_encoded['Storage_Type_SSD'] = data_encoded['Storage_Type'].apply(
        lambda x: 1 if x == 'SSD' else 0
    )
    
    return data_encoded

def create_interaction_features(data):
    """创建交互特征"""
    data_interaction = data.copy()
    
    # 性能相关交互特征
    data_interaction['RAM_Storage_Ratio'] = data_interaction['RAM_GB'] / data_interaction['Storage_Capacity']
    data_interaction['Screen_Pixel_Density'] = data_interaction['Resolution_Pixels'] / data_interaction['Screen Size (inch)']
    
    # 便携性特征
    data_interaction['Battery_Weight_Ratio'] = data_interaction['Battery Life (hours)'] / data_interaction['Weight (kg)']
    
    # 性能价格比特征
    data_interaction['Performance_Score'] = (
        data_interaction['Processor_Rank'] * 0.3 +
        data_interaction['RAM_GB'] * 0.2 +
        data_interaction['GPU_Type'] * 0.3 +
        data_interaction['Storage_Type_SSD'] * 0.2
    )
    
    return data_interaction

def preprocess_data(train_path, test_path):
    """完整的数据预处理流程"""
    print("=== 开始数据预处理 ===")
    
    # 1. 加载数据
    train_data, test_data = load_data(train_path, test_path)
    print(f"训练集大小: {train_data.shape}")
    print(f"测试集大小: {test_data.shape}")
    
    # 2. 基础清洗
    train_data = basic_cleaning(train_data)
    test_data = basic_cleaning(test_data)
    
    # 3. 提取数值特征
    train_data = extract_numeric_features(train_data)
    test_data = extract_numeric_features(test_data)
    
    # 4. 编码分类特征
    train_data = encode_categorical_features(train_data)
    test_data = encode_categorical_features(test_data)
    
    # 5. 创建交互特征
    train_data = create_interaction_features(train_data)
    test_data = create_interaction_features(test_data)
    
    print("=== 数据预处理完成 ===\n")
    
    return train_data, test_data

if __name__ == "__main__":
    # 测试预处理流程
    train_path = "../data/raw/laptop_prices_train.csv"
    test_path = "../data/raw/laptop_prices_test.csv"
    train_data, test_data = preprocess_data(train_path, test_path)
    print("预处理后的特征列:")
    print(train_data.columns.tolist())AttributeError: Can only use .str accessor with string values!
AttributeError: Can only use .str accessor with string values!
AttributeError: Can only use .str accessor with string values!
