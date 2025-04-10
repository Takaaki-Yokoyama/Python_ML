# PLS回帰分析

このプロジェクトは、部分最小二乗回帰（PLS回帰）を実装したPythonコードを提供します。

## 機能

- データの前処理（スケーリング、訓練データとテストデータの分割）
- PLS回帰モデルの実装
- モデルの評価（R2スコア、RMSE）
- 可視化機能（主成分のプロット、ローディングのヒートマップ）
- CSVファイルからのデータ読み込み
- 最適なXの候補の選定
- 最適な候補のCSVファイルへの保存

## 必要条件

必要なパッケージは`requirements.txt`に記載されています。以下のコマンドでインストールできます：

```bash
pip install -r requirements.txt
```

## 使用方法

1. 必要なパッケージをインストール
2. `main.py`を実行

### サンプルデータを使用する場合：
```bash
python main.py --n-components 2
```

### CSVファイルを使用する場合：
```bash
python main.py --csv データファイル.csv --target 目的変数の列名 --features 説明変数1 説明変数2 ... --n-components 2
```

### 最適な候補を選定する場合：
```bash
python main.py --find-optimal --n-candidates 10 --n-samples 1000
```

### 特徴量の範囲を指定して最適な候補を選定する場合：
```bash
python main.py --find-optimal --x-range 範囲設定.json --output 結果.json
```

### 最適な候補をCSVファイルに保存する場合：
```bash
python main.py --find-optimal --csv-output 最適候補.csv
```

#### コマンドラインオプション：
- `--csv`: 使用するCSVファイルのパス
- `--target`: 目的変数の列名
- `--features`: 説明変数の列名（省略可能。省略時は目的変数以外のすべての列を使用）
- `--n-components`: 使用する主成分の数（デフォルト: 2）
- `--find-optimal`: 最適な候補を選定する
- `--n-candidates`: 選定する候補の数（デフォルト: 10）
- `--n-samples`: 生成するランダムサンプルの数（デフォルト: 1000）
- `--x-range`: 各特徴量の範囲を指定するJSONファイルのパス
- `--output`: 結果を保存するJSONファイルのパス
- `--csv-output`: 最適な候補を保存するCSVファイルのパス（デフォルト: optimal_candidates.csv）

## 特徴量の範囲を指定するJSONファイルの形式

```json
[
  {"min": 0, "max": 10},
  {"min": -5, "max": 5},
  {"min": 1, "max": 100}
]
```

## CSVファイルの形式

最適な候補を保存するCSVファイルには以下の列が含まれます：
- `Candidate_Index`: 候補のインデックス（1から始まる連番）
- 各特徴量の列（特徴量の名前が列名）
- `Predicted_Y`: 予測値

## ファイル構成

- `main.py`: メイン実行ファイル
- `data_preprocessing.py`: データの前処理を行うモジュール
- `pls_regression.py`: PLS回帰モデルの実装
- `requirements.txt`: 必要なパッケージの一覧

## カスタマイズ

- CSVファイルを使用する場合は、データの形式に合わせて列名を指定してください
- `PLSRegressionModel`クラスの`n_components`パラメータを変更することで、使用する主成分の数を調整できます
- 最適な候補を選定する際は、`--n-candidates`と`--n-samples`パラメータを調整して、候補の数と探索範囲を変更できます 