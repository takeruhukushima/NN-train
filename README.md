# Fashion MNIST 機械学習・深層学習モデル比較プロジェクト

## 概要

このプロジェクトは、[Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist) データセットを用いて、様々な機械学習および深層学習モデルを実装・比較するためのリポジトリです。

古典的な機械学習手法から、NumPyを使ったスクラッチ実装、そしてTensorFlow/KerasとPyTorchによる深層学習モデルまで、幅広いアプローチを通じて画像分類タスクへの理解を深めることを目的としています。

## 🎯 目的

*   **ベースラインの確立:** Scikit-learnを用いた古典的手法での性能評価。
*   **アルゴリズムの理解:** NumPyを使い、主要な機械学習アルゴリズムをスクラッチで実装。
*   **フレームワークの比較:** TensorFlow/KerasとPyTorch、2つの主要な深層学習フレームワークの構文と設計思想の違いを実践的に学習。
*   **モデルの進化:** MLP（多層パーセプトロン）からCNN（畳み込みニューラルネットワーク）へとモデルを進化させ、性能向上を体験。

## 📂 ディレクトリ構成

```
.
├── data/              # Fashion MNISTデータセットと提出用CSV
├── keras/             # TensorFlow/Kerasによるモデル実装
├── numpy/             # NumPyによるアルゴリズムのスクラッチ実装
├── pytorch/           # PyTorchによるモデル実装
├── sklearn/           # Scikit-learnによる古典的モデル実装
├── agenda.md          # プロジェクトの学習計画
├── pyproject.toml     # プロジェクトの依存関係定義
└── README.md          # このファイル
```

*   **`sklearn/`**: ロジスティック回帰、SVM、k-NNなどの古典的モデル。
*   **`numpy/`**: ライブラリに頼らず、k-NN、ロジスティック回帰、SVM、MLPをNumPyで実装。
*   **`keras/`**: Keras (TensorFlowバックエンド) を用いたMLP、CNNモデル。
*   **`pytorch/`**: PyTorchを用いたMLPモデル。
*   **`data/`**: このディレクトリには、モデルの学習と評価に使用するデータが格納して実行します。
    *   `data/classification/x_train.npy`: 学習用の画像データ。
    *   `data/classification/y_train.npy`: 学習用のラベルデータ。
    *   `data/classification/x_test.npy`: テスト（評価）用の画像データ。
    *   各モデルを実行すると、予測結果のCSVファイルもこのディレクトリ以下に保存されます。

## 🛠️ 使用技術・ライブラリ

プロジェクトの依存関係は `pyproject.toml` に記載されています。

*   **深層学習:**
    *   `tensorflow`
    *   `torch`
    *   `torchvision`
*   **古典的機械学習:**
    *   `scikit-learn`
*   **数値計算・データ操作:**
    *   `numpy`
    *   `pandas`
    *   `scipy`
*   **その他:**
    *   `tqdm`

## 🚀 実行方法

1.  **リポジトリをクローン:**
    ```bash
    git clone <repository-url>
    cd NN-training
    ```

2.  **仮想環境の作成と依存関係のインストール:**
    このプロジェクトでは `uv` をパッケージマネージャとして使用します。`uv`がインストールされていない場合は、まずインストールしてください。
    ```bash
    # 仮想環境を作成し、pyproject.tomlに基づいて依存関係を同期
    uv sync
    ```
    これにより、`pyproject.toml` と `uv.lock` に基づいた正確な環境が構築されます。
3.  **各モデルのスクリプトを実行:**
    各ディレクトリ内のPythonスクリプトを実行して、モデルの学習と予測を行います。
    ```bash
    # 例: Scikit-learnのロジスティック回帰を実行
    uv run sklearn/logistic.py

    # 例: KerasのCNNモデルを実行
    uv run keras/keras_cnn.py
    ```
    実行後、`data/` ディレクトリに提出用のCSVファイルが生成されます。
