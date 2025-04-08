import os
import argparse
from src.genetic_algorithm import MultiObjectiveGA
from src.visualization import plot_pareto_front, save_pareto_front
from src.objective_functions import generate_sample_data, save_sample_data

def main():
    # コマンドライン引数の解析
    parser = argparse.ArgumentParser(description='遺伝的アルゴリズムによる多目的最適化')
    parser.add_argument('--generate', action='store_true', help='サンプルデータを生成して保存する')
    parser.add_argument('--data-file', type=str, default='results/sample_data.csv', help='データファイルのパス')
    parser.add_argument('--pop-size', type=int, default=100, help='個体数')
    parser.add_argument('--n-generations', type=int, default=100, help='世代数')
    parser.add_argument('--mutation-rate', type=float, default=0.1, help='突然変異率')
    args = parser.parse_args()
    
    # 結果ディレクトリの作成
    os.makedirs('results', exist_ok=True)
    
    # サンプルデータの生成と保存
    if args.generate:
        print("サンプルデータを生成しています...")
        X, f1, f2 = generate_sample_data(n_samples=1000)
        save_sample_data(X, f1, f2, args.data_file)
        print("サンプルデータの生成が完了しました。")
        print("注意: 生成されたサンプルデータの目的変数はダミーデータです。")
        print("実際のデータセットを使用する場合は、目的変数を適切な値に置き換えてください。")
        return
    
    # 遺伝的アルゴリズムの実行
    print("遺伝的アルゴリズムを実行しています...")
    ga = MultiObjectiveGA(
        pop_size=args.pop_size,
        n_generations=args.n_generations,
        mutation_rate=args.mutation_rate,
        data_file=args.data_file
    )
    pareto_front = ga.optimize()
    
    # 結果の可視化
    print("パレート最適解を可視化しています...")
    plot_pareto_front(pareto_front, data_file=args.data_file)
    
    # パレート最適解をCSVファイルに保存
    save_pareto_front(pareto_front, 'results/pareto_front.csv', data_file=args.data_file)
    
    # 結果の出力
    print("\nパレート最適解:")
    for i, solution in enumerate(pareto_front):
        print(f"解 {i+1}: x = {solution}")

if __name__ == "__main__":
    main() 