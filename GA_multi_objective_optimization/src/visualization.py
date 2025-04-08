import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .objective_functions import load_sample_data

# 日本語フォントの設定
plt.rcParams['font.family'] = 'MS Gothic'  # Windowsの場合
# plt.rcParams['font.family'] = 'IPAGothic'  # Macの場合

def plot_pareto_front(pareto_front, data_file=None):
    """
    パレート最適解の可視化
    """
    if data_file:
        # データファイルから目的変数を読み込む
        _, f1_all, f2_all = load_sample_data(data_file)
        
        # パレート最適解に対応する目的変数を取得
        f1_values = []
        f2_values = []
        for x in pareto_front:
            # 最も近いデータポイントを見つける
            distances = np.sum((np.array([x]) - pareto_front)**2, axis=1)
            closest_idx = np.argmin(distances)
            f1_values.append(f1_all[closest_idx])
            f2_values.append(f2_all[closest_idx])
        
        f1_values = np.array(f1_values)
        f2_values = np.array(f2_values)
    else:
        # 目的変数が直接指定されている場合
        f1_values = np.array([x[0] for x in pareto_front])
        f2_values = np.array([x[1] for x in pareto_front])
    
    plt.figure(figsize=(10, 6))
    plt.scatter(f1_values, f2_values, c='blue', label='パレート最適解')
    plt.xlabel('目的関数1 (f1)')
    plt.ylabel('目的関数2 (f2)')
    plt.title('パレート最適解の分布')
    plt.grid(True)
    plt.legend()
    plt.show()

def save_pareto_front(pareto_front, filename='pareto_front.csv', data_file=None):
    """
    パレート最適解をCSVファイルに保存
    """
    if data_file:
        # データファイルから目的変数を読み込む
        _, f1_all, f2_all = load_sample_data(data_file)
        
        # パレート最適解に対応する目的変数を取得
        f1_values = []
        f2_values = []
        for x in pareto_front:
            # 最も近いデータポイントを見つける
            distances = np.sum((np.array([x]) - pareto_front)**2, axis=1)
            closest_idx = np.argmin(distances)
            f1_values.append(f1_all[closest_idx])
            f2_values.append(f2_all[closest_idx])
        
        f1_values = np.array(f1_values)
        f2_values = np.array(f2_values)
    else:
        # 目的変数が直接指定されている場合
        f1_values = np.array([x[0] for x in pareto_front])
        f2_values = np.array([x[1] for x in pareto_front])
    
    # DataFrameの作成
    df = pd.DataFrame({
        'x1': pareto_front[:, 0],
        'x2': pareto_front[:, 1],
        'x3': pareto_front[:, 2],
        'x4': pareto_front[:, 3],
        'x5': pareto_front[:, 4],
        'x6': pareto_front[:, 5],
        'f1': f1_values,
        'f2': f2_values
    })
    
    # CSVファイルに保存
    df.to_csv(filename, index=False, encoding='utf-8')
    print(f"パレート最適解を {filename} に保存しました。")

def plot_search_space():
    """
    探索空間の可視化
    """
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    
    Z1 = X**2 + Y**2  # 目的関数1
    Z2 = (X-2)**2 + (Y-2)**2  # 目的関数2
    
    fig = plt.figure(figsize=(15, 5))
    
    ax1 = fig.add_subplot(121, projection='3d')
    surf1 = ax1.plot_surface(X, Y, Z1, cmap='viridis')
    ax1.set_title('目的関数1: f1(x) = x^2')
    fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)
    
    ax2 = fig.add_subplot(122, projection='3d')
    surf2 = ax2.plot_surface(X, Y, Z2, cmap='viridis')
    ax2.set_title('目的関数2: f2(x) = (x-2)^2')
    fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=5)
    
    plt.tight_layout()
    plt.show() 