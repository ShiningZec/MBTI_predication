import os
import zipfile
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi

# 1. 初始化并认证 Kaggle API
api = KaggleApi()
api.authenticate()

# 2. 定义数据集和保存路径
dataset = "zeyadkhalid/mbti-personality-types-500-dataset"
data_dir = "data"
os.makedirs(data_dir, exist_ok=True)

# 3. 下载数据集（会自动下载一个 zip 包）
print(f"正在从 Kaggle 下载 {dataset}...")
api.dataset_download_files(dataset, path=data_dir, unzip=True)
print("下载并解压完成！")

# 4. 读取并重命名 CSV 文件（如果下载的压缩包内文件名不是 'MBTI 500.csv'，请根据实际情况修改）
csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
if csv_files:
    # 假设压缩包内只有一个 CSV 文件
    original_name = csv_files[0]
    target_name = "MBTI 500.csv"  # 你项目里用的名字
    os.rename(os.path.join(data_dir, original_name), os.path.join(data_dir, target_name))
    print(f"文件已重命名为: {target_name}")
    
    # 可选：快速验证 CSV 是否可用
    df = pd.read_csv(os.path.join(data_dir, target_name))
    print(f"CSV 文件加载成功！形状: {df.shape}")
else:
    print("未在下载目录中找到 CSV 文件，请检查。")