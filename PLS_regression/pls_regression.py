import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

class PLSRegressionModel:
    def __init__(self, n_components=2):
        """
        PLS回帰モデルの初期化
        
        Parameters:
        -----------
        n_components : int
            使用する主成分の数
        """
        self.model = PLSRegression(n_components=n_components)
        self.n_components = n_components
        
    def fit(self, X, y):
        """
        モデルの学習
        
        Parameters:
        -----------
        X : array-like
            説明変数
        y : array-like
            目的変数
        """
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        """
        予測を行う
        
        Parameters:
        -----------
        X : array-like
            説明変数
            
        Returns:
        --------
        array-like
            予測値
        """
        return self.model.predict(X)
    
    def evaluate(self, X, y_true):
        """
        モデルの評価を行う
        
        Parameters:
        -----------
        X : array-like
            説明変数
        y_true : array-like
            実際の値
            
        Returns:
        --------
        dict
            評価指標
        """
        y_pred = self.predict(X)
        return {
            'R2': r2_score(y_true, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred))
        }
    
    def plot_components(self, X, y):
        """
        主成分の可視化
        
        Parameters:
        -----------
        X : array-like
            説明変数
        y : array-like
            目的変数
        """
        # スコアの計算
        scores = self.model.transform(X)
        
        # プロット
        plt.figure(figsize=(10, 6))
        plt.scatter(scores[:, 0], scores[:, 1], c=y, cmap='viridis')
        plt.xlabel('First Component')
        plt.ylabel('Second Component')
        plt.title('PLS Components')
        plt.colorbar(label='Target Value')
        plt.show()
        
    def plot_loading(self):
        """
        ローディングの可視化
        """
        loadings = self.model.x_loadings_
        
        plt.figure(figsize=(12, 6))
        sns.heatmap(loadings, annot=True, cmap='RdBu_r', center=0)
        plt.title('PLS Loadings')
        plt.xlabel('Component')
        plt.ylabel('Variable')
        plt.show()
        
    def find_optimal_candidates(self, X_range, n_candidates=10, n_samples=1000, random_state=42):
        """
        ランダムに生成したXの候補から、最もyが良くなるトップn個を選定する
        
        Parameters:
        -----------
        X_range : list of tuple
            各特徴量の範囲を指定するリスト。各要素は(min, max)のタプル
        n_candidates : int
            選定する候補の数
        n_samples : int
            生成するランダムサンプルの数
        random_state : int
            乱数シード
            
        Returns:
        --------
        X_candidates : array-like
            選定されたXの候補
        y_candidates : array-like
            選定されたXに対応する予測値
        """
        np.random.seed(random_state)
        
        # ランダムなXを生成
        X_random = np.zeros((n_samples, len(X_range)))
        for i, (min_val, max_val) in enumerate(X_range):
            X_random[:, i] = np.random.uniform(min_val, max_val, n_samples)
        
        # 予測値を計算
        y_pred = self.predict(X_random)
        
        # 予測値でソート（昇順）
        sorted_indices = np.argsort(y_pred.flatten())
        
        # トップn個を選定
        top_indices = sorted_indices[:n_candidates]
        X_candidates = X_random[top_indices]
        y_candidates = y_pred[top_indices]
        
        return X_candidates, y_candidates
    
    def plot_optimal_candidates(self, X_candidates, y_candidates, feature_names=None):
        """
        最適な候補を可視化する
        
        Parameters:
        -----------
        X_candidates : array-like
            選定されたXの候補
        y_candidates : array-like
            選定されたXに対応する予測値
        feature_names : list of str, optional
            特徴量の名前のリスト
        """
        n_features = X_candidates.shape[1]
        if feature_names is None:
            feature_names = [f'Feature {i+1}' for i in range(n_features)]
        
        # 特徴量ごとのプロット
        plt.figure(figsize=(15, 5))
        for i in range(n_features):
            plt.subplot(1, n_features, i+1)
            plt.scatter(X_candidates[:, i], y_candidates, c='blue', alpha=0.7)
            plt.xlabel(feature_names[i])
            plt.ylabel('Predicted Y')
            plt.title(f'{feature_names[i]} vs Y')
        plt.tight_layout()
        plt.show()
        
        # 予測値の分布
        plt.figure(figsize=(10, 5))
        plt.hist(y_candidates, bins=10, alpha=0.7, color='green')
        plt.xlabel('Predicted Y')
        plt.ylabel('Frequency')
        plt.title('Distribution of Predicted Y for Optimal Candidates')
        plt.show()
        
    def save_candidates_to_csv(self, X_candidates, y_candidates, feature_names=None, output_path='optimal_candidates.csv'):
        """
        最適な候補をCSVファイルに保存する
        
        Parameters:
        -----------
        X_candidates : array-like
            選定されたXの候補
        y_candidates : array-like
            選定されたXに対応する予測値
        feature_names : list of str, optional
            特徴量の名前のリスト
        output_path : str
            出力するCSVファイルのパス
        """
        n_features = X_candidates.shape[1]
        if feature_names is None:
            feature_names = [f'Feature {i+1}' for i in range(n_features)]
        
        # DataFrameを作成
        data = {}
        for i, name in enumerate(feature_names):
            data[name] = X_candidates[:, i]
        
        # 予測値を追加
        data['Predicted_Y'] = y_candidates.flatten()
        
        # インデックスを追加
        data['Candidate_Index'] = np.arange(1, len(X_candidates) + 1)
        
        # DataFrameに変換
        df = pd.DataFrame(data)
        
        # 列の順序を調整（インデックスを先頭に）
        cols = ['Candidate_Index'] + feature_names + ['Predicted_Y']
        df = df[cols]
        
        # CSVファイルに保存
        df.to_csv(output_path, index=False)
        print(f"最適な候補を {output_path} に保存しました。") 