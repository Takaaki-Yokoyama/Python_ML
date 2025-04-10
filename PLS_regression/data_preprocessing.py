import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_data_from_csv(file_path, target_column, feature_columns=None):
    """
    CSVファイルからデータを読み込む関数
    
    Parameters:
    -----------
    file_path : str
        CSVファイルのパス
    target_column : str
        目的変数の列名
    feature_columns : list of str, optional
        説明変数の列名のリスト。Noneの場合は、target_column以外のすべての列を使用
    
    Returns:
    --------
    X : array-like
        説明変数
    y : array-like
        目的変数
    """
    # CSVファイルの読み込み
    df = pd.read_csv(file_path)
    
    # 説明変数の列を決定
    if feature_columns is None:
        feature_columns = [col for col in df.columns if col != target_column]
    
    # データの抽出
    X = df[feature_columns].values
    y = df[target_column].values
    
    return X, y

def load_and_preprocess_data(X, y, test_size=0.2, random_state=42):
    """
    データの読み込みと前処理を行う関数
    
    Parameters:
    -----------
    X : array-like
        説明変数
    y : array-like
        目的変数
    test_size : float
        テストデータの割合
    random_state : int
        乱数シード
    
    Returns:
    --------
    X_train, X_test, y_train, y_test : 前処理済みの訓練データとテストデータ
    scaler_X, scaler_y : スケーラー
    """
    # データの分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # スケーリング
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1))
    
    return (X_train_scaled, X_test_scaled, 
            y_train_scaled, y_test_scaled,
            scaler_X, scaler_y) 