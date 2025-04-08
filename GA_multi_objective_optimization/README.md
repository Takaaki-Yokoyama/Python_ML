# 遺伝的アルゴリズムによる多目的最適化

このプロジェクトは、遺伝的アルゴリズム（GA）を使用して多目的最適化問題を解くPythonプログラムです。

## 機能
- 6次元の説明変数と2次元の目的変数による多目的最適化
- CSVファイルからのデータ読み込み
- パレート最適解の探索
- パレート最適解の可視化
- 結果のCSVファイル出力

## セットアップ
```bash
pip install -r requirements.txt
```

## 使用方法
### サンプルデータの生成（説明変数のみ）
```bash
python main.py --generate
```
これにより、`results/sample_data.csv`に説明変数のサンプルデータが生成されます。
**注意**: 生成されたサンプルデータの目的変数はダミーデータです。実際のデータセットを使用する場合は、目的変数を適切な値に置き換えてください。

### 遺伝的アルゴリズムの実行
```bash
python main.py --data-file results/sample_data.csv
```

### オプション
- `--generate`: サンプルデータを生成して保存する
- `--data-file`: データファイルのパス（デフォルト: results/sample_data.csv）
- `--pop-size`: 個体数（デフォルト: 100）
- `--n-generations`: 世代数（デフォルト: 100）
- `--mutation-rate`: 突然変異率（デフォルト: 0.1）

## プロジェクト構造
```
GA_multi_objective_optimization/
├── requirements.txt          # 必要なパッケージの一覧
├── README.md                # プロジェクトの説明
├── src/                     # ソースコードディレクトリ
│   ├── __init__.py
│   ├── genetic_algorithm.py # 遺伝的アルゴリズムの実装
│   ├── objective_functions.py # データの読み込みと保存
│   └── visualization.py     # 可視化機能
├── results/                 # 結果出力ディレクトリ
│   ├── sample_data.csv     # サンプルデータ
│   └── pareto_front.csv    # パレート最適解のCSVファイル
└── main.py                 # メイン実行ファイル
```

## 入力ファイル
- `results/sample_data.csv`: サンプルデータ
  - x1~x6: 6次元の説明変数の値
  - f1: 第1目的変数の値
  - f2: 第2目的変数の値

## 出力ファイル
- `results/pareto_front.csv`: パレート最適解のデータ
  - x1~x6: 6次元の説明変数の値
  - f1: 第1目的変数の値
  - f2: 第2目的変数の値 