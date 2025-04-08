import numpy as np
import pandas as pd
import os

def generate_sample_data(n_samples=100):
    """
    サンプルデータを生成する関数
    """
    X = np.random.uniform(-5, 5, (n_samples, 6))
    
    # 目的変数は外部データから読み込むため、ここではダミーデータを生成
    f1 = np.zeros(n_samples)
    f2 = np.zeros(n_samples)
    
    return X, f1, f2

def save_sample_data(X, f1, f2, filename='sample_data.csv'):
    """
    サンプルデータをCSVファイルに保存する関数
    """
    # 結果ディレクトリが存在しない場合は作成
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # DataFrameの作成
    df = pd.DataFrame({
        'x1': X[:, 0],
        'x2': X[:, 1],
        'x3': X[:, 2],
        'x4': X[:, 3],
        'x5': X[:, 4],
        'x6': X[:, 5],
        'f1': f1,
        'f2': f2
    })
    
    # CSVファイルに保存
    df.to_csv(filename, index=False, encoding='utf-8')
    print(f"サンプルデータを {filename} に保存しました。")

def load_sample_data(filename='sample_data.csv'):
    """
    CSVファイルからサンプルデータを読み込む関数
    """
    # CSVファイルの読み込み
    df = pd.read_csv(filename, encoding='utf-8')
    
    # 説明変数と目的変数の抽出
    X = df[['x1', 'x2', 'x3', 'x4', 'x5', 'x6']].values
    f1 = df['f1'].values
    f2 = df['f2'].values
    
    return X, f1, f2 