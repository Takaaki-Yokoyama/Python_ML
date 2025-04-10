import numpy as np
import argparse
import json
from data_preprocessing import load_and_preprocess_data, load_data_from_csv
from pls_regression import PLSRegressionModel

def generate_sample_data(n_samples=100, n_features=10):
    """
    サンプルデータの生成
    
    Parameters:
    -----------
    n_samples : int
        サンプル数
    n_features : int
        特徴量の数
        
    Returns:
    --------
    X : array-like
        説明変数
    y : array-like
        目的変数
    """
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features)
    # 目的変数は説明変数の線形結合にノイズを加えたもの
    y = np.sum(X[:, :3], axis=1) + 0.1 * np.random.randn(n_samples)
    return X, y

def main():
    # コマンドライン引数の設定
    parser = argparse.ArgumentParser(description='PLS回帰分析を実行します')
    parser.add_argument('--csv', type=str, help='CSVファイルのパス')
    parser.add_argument('--target', type=str, help='目的変数の列名')
    parser.add_argument('--features', type=str, nargs='+', help='説明変数の列名（省略可）')
    parser.add_argument('--n-components', type=int, default=2, help='使用する主成分の数')
    parser.add_argument('--find-optimal', action='store_true', help='最適な候補を選定する')
    parser.add_argument('--n-candidates', type=int, default=10, help='選定する候補の数')
    parser.add_argument('--n-samples', type=int, default=1000, help='生成するランダムサンプルの数')
    parser.add_argument('--x-range', type=str, help='各特徴量の範囲を指定するJSONファイルのパス')
    parser.add_argument('--output', type=str, help='結果を保存するJSONファイルのパス')
    parser.add_argument('--csv-output', type=str, help='最適な候補を保存するCSVファイルのパス')
    args = parser.parse_args()

    # データの準備
    if args.csv:
        # CSVファイルからデータを読み込む
        if not args.target:
            raise ValueError("CSVファイルを使用する場合は、--targetオプションで目的変数の列名を指定してください")
        X, y = load_data_from_csv(args.csv, args.target, args.features)
        feature_names = args.features if args.features else [f'Feature {i+1}' for i in range(X.shape[1])]
    else:
        # サンプルデータを生成
        X, y = generate_sample_data()
        feature_names = [f'Feature {i+1}' for i in range(X.shape[1])]
    
    # データの前処理
    X_train, X_test, y_train, y_test, scaler_X, scaler_y = load_and_preprocess_data(X, y)
    
    # モデルの学習
    pls_model = PLSRegressionModel(n_components=args.n_components)
    pls_model.fit(X_train, y_train)
    
    # モデルの評価
    train_metrics = pls_model.evaluate(X_train, y_train)
    test_metrics = pls_model.evaluate(X_test, y_test)
    
    print("訓練データの評価指標:")
    print(f"R2: {train_metrics['R2']:.4f}")
    print(f"RMSE: {train_metrics['RMSE']:.4f}")
    
    print("\nテストデータの評価指標:")
    print(f"R2: {test_metrics['R2']:.4f}")
    print(f"RMSE: {test_metrics['RMSE']:.4f}")
    
    # 可視化
    pls_model.plot_components(X_train, y_train)
    pls_model.plot_loading()
    
    # 最適な候補の選定
    if args.find_optimal:
        # Xの範囲を設定
        if args.x_range:
            # JSONファイルから範囲を読み込む
            with open(args.x_range, 'r') as f:
                x_range_data = json.load(f)
            X_range = [(item['min'], item['max']) for item in x_range_data]
        else:
            # デフォルトの範囲を設定（各特徴量の最小値と最大値から少し広めに）
            X_min = np.min(X, axis=0)
            X_max = np.max(X, axis=0)
            X_range = [(min_val - 0.5, max_val + 0.5) for min_val, max_val in zip(X_min, X_max)]
        
        # 最適な候補を選定
        X_candidates, y_candidates = pls_model.find_optimal_candidates(
            X_range, 
            n_candidates=args.n_candidates, 
            n_samples=args.n_samples
        )
        
        # 結果の表示
        print("\n最適な候補:")
        for i, (x, y) in enumerate(zip(X_candidates, y_candidates)):
            print(f"候補 {i+1}:")
            for j, (name, value) in enumerate(zip(feature_names, x)):
                print(f"  {name}: {value:.4f}")
            print(f"  予測値: {y[0]:.4f}")
        
        # 可視化
        pls_model.plot_optimal_candidates(X_candidates, y_candidates, feature_names)
        
        # 結果の保存（JSON）
        if args.output:
            result = {
                'candidates': [
                    {
                        'index': i,
                        'features': {name: float(value) for name, value in zip(feature_names, x)},
                        'predicted_value': float(y[0])
                    }
                    for i, (x, y) in enumerate(zip(X_candidates, y_candidates))
                ]
            }
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"\n結果を {args.output} に保存しました。")
            
        # 結果の保存（CSV）
        csv_output_path = args.csv_output if args.csv_output else 'optimal_candidates.csv'
        pls_model.save_candidates_to_csv(X_candidates, y_candidates, feature_names, csv_output_path)

if __name__ == "__main__":
    main() 