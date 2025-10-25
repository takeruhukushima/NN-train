#kerasを用いたMLPモデルの実装
# --- モデル評価 (検証データ) ---
# Keras .evaluate() の結果:
#   検証データの損失 (Loss): 0.3596
#   検証データの正解率 (Accuracy): 87.09 %

# --- 分類レポート (検証データ) ---
# 375/375 ━━━━━━━━━━━━━━━━━━━━ 0s 189us/step
#               precision    recall  f1-score   support

#  T-shirt/top       0.91      0.67      0.77      1182
#      Trouser       0.99      0.98      0.98      1222
#     Pullover       0.81      0.83      0.82      1210
#        Dress       0.86      0.92      0.89      1153
#         Coat       0.87      0.65      0.74      1180
#       Sandal       0.98      0.93      0.96      1215
#        Shirt       0.59      0.84      0.69      1239
#      Sneaker       0.91      0.98      0.94      1228
#          Bag       0.97      0.96      0.97      1185
#   Ankle boot       0.96      0.95      0.95      1186

#     accuracy                           0.87     12000
#    macro avg       0.89      0.87      0.87     12000
# weighted avg       0.89      0.87      0.87     12000
#めちゃくちゃ速かった。k-NN（線形モデル）よりも似ている画像における精度が高い。（shirtが気になる）

import numpy as np
import pandas as pd
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# ★ Keras (TensorFlow) のライブラリをインポート
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ★ tf のログレベルを調整 (INFOログを非表示に)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
tf.get_logger().setLevel('WARNING')

class KerasMLPModel:
    """
    Keras (TensorFlow) で MLP を構築・評価するクラス。
    """
    
    def __init__(self, hidden_dim=128, epochs=10, batch_size=32, random_state=42):
        """
        ハイパーパラメータとモデルのアーキテクチャを定義
        """
        np.random.seed(random_state)
        tf.random.set_seed(random_state)
        
        self.epochs = epochs
        self.batch_size = batch_size
        
        # ★ ここで Keras のモデルを定義
        self.model = keras.Sequential([
            # 1. 入力層 (Flattenは _preprocess で実行済みなので、入力次元 784 を指定)
            layers.Input(shape=(784,), name="input_layer"),
            
            # 2. 隠れ層 (これが非線形性を生み出す)
            layers.Dense(hidden_dim, activation='relu', name="hidden_layer_1"),
            
            # 3. 出力層 (ソフトマックス回帰と同じ)
            layers.Dense(10, activation='softmax', name="output_layer")
        ])
        
        # ★ モデルの「コンパイル」 (オプティマイザと損失関数を設定)
        self.model.compile(
            optimizer='adam',  # ★ オプティマイザは Adam
            loss='sparse_categorical_crossentropy', # y が [1, 9, 0] の形式の場合
            metrics=['accuracy'] # 訓練中に精度も表示
        )
        
        print("Keras モデルの定義完了:")
        self.model.summary() # モデルの構造を表示

        # --- (他の変数は sklearn 版と同様) ---
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_train_processed = None
        self.X_test_processed = None
        self.y_train_orig = None

    # --- load_data, _preprocess, split_data は sklearn 版と全く同じ ---

    def load_data(self, x_train_path, y_train_path, x_test_path):
        try:
            X_train_orig = np.load(x_train_path)
            self.y_train_orig = np.load(y_train_path)
            X_test_orig = np.load(x_test_path)
            
            print("データの読み込み完了")
            self.X_train_processed = self._preprocess(X_train_orig)
            self.X_test_processed = self._preprocess(X_test_orig)
            print("データの前処理 (Flatten, Normalization) 完了")
        except FileNotFoundError as e:
            print(f"エラー: ファイルが見つかりません - {e}", file=sys.stderr)
            sys.exit(1)

    def _preprocess(self, data):
        if data.ndim == 3:
            data = data.reshape(data.shape[0], -1) # (N, 28, 28) -> (N, 784)
        data = data.astype('float32') / 255.0
        return data

    def split_data(self, validation_size=0.2):
        if self.X_train_processed is None or self.y_train_orig is None:
            print("エラー: 訓練データが読み込まれていません", file=sys.stderr)
            return
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            self.X_train_processed,
            self.y_train_orig,
            test_size=validation_size,
            random_state=42
        )
        print(f"訓練データを分割しました (検証用: {validation_size*100}%)")

    # --- train メソッドを Keras 用に書き換え ---

    def train(self):
        """
        Keras の .fit() を使ってモデルを訓練する
        """
        if self.X_train is None or self.y_train is None:
            print("エラー: 訓練データがありません", file=sys.stderr)
            return

        print(f"Keras モデルの訓練を開始します (Epochs: {self.epochs}, Batch Size: {self.batch_size})...")
        
        # ★ .fit() を呼び出すだけ
        # (訓練の進捗 (loss, accuracy) が自動で表示される)
        self.history = self.model.fit(
            self.X_train,
            self.y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            # ★ 検証用データを渡すと、エポック毎に評価もしてくれる
            validation_data=(self.X_val, self.y_val), 
            verbose=1 # ログを表示
        )
        print("モデルの訓練が完了しました。")

    # --- evaluate メソッドを Keras 用に書き換え ---

    def evaluate(self):
        """
        Keras の .evaluate() を使ってモデルを評価する
        """
        if self.model is None or self.X_val is None or self.y_val is None:
            print("エラー: モデルまたは検証データがありません", file=sys.stderr)
            return

        print("\n--- モデル評価 (検証データ) ---")
        
        # 1. Keras の .evaluate() で損失と精度を計算
        loss, accuracy = self.model.evaluate(self.X_val, self.y_val, verbose=0)
        print(f"Keras .evaluate() の結果:")
        print(f"  検証データの損失 (Loss): {loss:.4f}")
        print(f"  検証データの正解率 (Accuracy): {accuracy * 100:.2f} %")

        # 2. sklearn の分類レポートも表示 (予測が必要)
        print("\n--- 分類レポート (検証データ) ---")
        y_pred_proba = self.model.predict(self.X_val)
        y_pred_val = np.argmax(y_pred_proba, axis=1) # 確率からラベルに変換
        
        class_names = [
            "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
            "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
        ]
        try:
            report = classification_report(self.y_val, y_pred_val, target_names=class_names, zero_division=0)
            print(report)
        except Exception as e:
            print(f"分類レポートの生成に失敗: {e}")
        return accuracy

    # --- predict_for_submission メソッドを Keras 用に書き換え ---

    def predict_for_submission(self, output_csv_path):
        """
        x_test.npy に対する最終的な予測（ラベル）をCSVに保存する
        """
        if self.model is None or self.X_test_processed is None:
            print("エラー: モデルまたはテストデータがありません", file=sys.stderr)
            return
            
        print(f"\n最終的なテストデータ (x_test.npy) に対する予測を実行します...")
        y_pred_proba = self.model.predict(self.X_test_processed)
        y_pred_test = np.argmax(y_pred_proba, axis=1) # 確率からラベルに変換
        
        submission_df = pd.DataFrame({
            "ID": np.arange(1, len(y_pred_test) + 1),
            "Label": y_pred_test
        })

        try:
            submission_df.to_csv(output_csv_path, index=False)
            print(f"予測結果を {output_csv_path} に保存しました。")
        except Exception as e:
            print(f"エラー: CSVファイルの保存に失敗しました - {e}", file=sys.stderr)


if __name__ == "__main__":
    X_TRAIN_PATH = 'data/classification/x_train.npy'
    Y_TRAIN_PATH = 'data/classification/y_train.npy'
    X_TEST_PATH = 'data/classification/x_test.npy'
    OUTPUT_CSV_PATH = 'data/classification/submission_keras_mlp.csv'

    # 1. モデルのインスタンス化 (10エポック実行)
    mlp_model = KerasMLPModel(epochs=10, batch_size=32)
    
    mlp_model.load_data(X_TRAIN_PATH, Y_TRAIN_PATH, X_TEST_PATH)
    mlp_model.split_data(validation_size=0.2)
    mlp_model.train()
    mlp_model.evaluate()
    mlp_model.predict_for_submission(OUTPUT_CSV_PATH)