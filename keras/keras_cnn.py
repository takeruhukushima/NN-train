#kerasを用いたCNNモデルの構築・評価クラス
# --- モデル評価 (検証データ) ---
# Keras .evaluate() の結果:
#   検証データの損失 (Loss): 0.3581
#   検証データの正解率 (Accuracy): 89.44 %

# --- 分類レポート (検証データ) ---
# 375/375 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step 
#               precision    recall  f1-score   support

#  T-shirt/top       0.91      0.75      0.82      1182
#      Trouser       0.98      0.99      0.98      1222
#     Pullover       0.87      0.84      0.86      1210
#        Dress       0.89      0.89      0.89      1153
#         Coat       0.88      0.74      0.80      1180
#       Sandal       0.98      0.98      0.98      1215
#        Shirt       0.63      0.84      0.72      1239
#      Sneaker       0.95      0.98      0.96      1228
#          Bag       0.97      0.98      0.98      1185
#   Ankle boot       0.98      0.95      0.97      1186

#     accuracy                           0.89     12000
#    macro avg       0.90      0.89      0.90     12000
# weighted avg       0.90      0.89      0.90     12000
#過去最高の結果になった
import numpy as np
import pandas as pd
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Keras (TensorFlow) のライブラリ
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# tf のログレベル調整
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
tf.get_logger().setLevel('WARNING')

class KerasCNNModel:
    """
    Keras (TensorFlow) で CNN を構築・評価するクラス。
    """

    def __init__(self, epochs=10, batch_size=32, random_state=42):
        np.random.seed(random_state)
        tf.random.set_seed(random_state)

        self.epochs = epochs
        self.batch_size = batch_size

        # ★ CNN モデルの定義 ★
        self.model = keras.Sequential([
            # 1. 入力層 (28x28 のグレースケール画像, チャネル数は1)
            layers.Input(shape=(28, 28, 1), name="input_layer"),

            # === CNN ブロック 1 ===
            # 畳み込み層 1: 32個の 3x3 フィルターを使用
            layers.Conv2D(32, kernel_size=(3, 3), activation='relu', name="conv2d_1"),
            # プーリング層 1: 2x2 領域で最大値を取る
            layers.MaxPooling2D(pool_size=(2, 2), name="maxpool_1"),

            # === CNN ブロック 2 ===
            # 畳み込み層 2: 64個の 3x3 フィルターを使用
            layers.Conv2D(64, kernel_size=(3, 3), activation='relu', name="conv2d_2"),
            # プーリング層 2: 2x2 領域で最大値を取る
            layers.MaxPooling2D(pool_size=(2, 2), name="maxpool_2"),

            # === 分類のための全結合層 ===
            # 平坦化 (Flatten): CNNで抽出した特徴マップを1次元ベクトルに変換
            layers.Flatten(name="flatten"),
            # 全結合層 (MLPと同じ): ReLU活性化
            layers.Dense(128, activation='relu', name="dense_1"),
            # 出力層: Softmax活性化で10クラスに分類
            layers.Dense(10, activation='softmax', name="output_layer")
        ])

        # モデルのコンパイル (MLPと同じ)
        self.model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        print("Keras CNN モデルの定義完了:")
        self.model.summary() # モデルの構造を表示

        # --- (他の変数は以前のクラスと同様) ---
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_train_processed = None
        self.X_test_processed = None
        self.y_train_orig = None

    # --- load_data は変更なし ---
    def load_data(self, x_train_path, y_train_path, x_test_path):
        try:
            X_train_orig = np.load(x_train_path)
            self.y_train_orig = np.load(y_train_path)
            X_test_orig = np.load(x_test_path)

            print("データの読み込み完了")
            # ★ _preprocess を呼び出すのは split_data の後に行う ★
            #    CNNは 4D Tensor (N, H, W, C) を必要とするため
            self.X_train_orig = X_train_orig
            self.X_test_orig = X_test_orig

        except FileNotFoundError as e:
            print(f"エラー: ファイルが見つかりません - {e}", file=sys.stderr)
            sys.exit(1)

    # --- ★ _preprocess を CNN 用に変更 ★ ---
    def _preprocess(self, data):
        """
        データを CNN が扱える形式に前処理する
        (Normalization と チャネル次元の追加)
        """
        # 1. 正規化 (Normalization)
        data = data.astype('float32') / 255.0
        # 2. ★ チャネル次元の追加 (N, 28, 28) -> (N, 28, 28, 1)
        #    Keras (TensorFlowバックエンド) はチャネルラスト形式 (H, W, C) を期待
        data = np.expand_dims(data, axis=-1)
        return data

    # --- ★ split_data で _preprocess を呼び出すように変更 ---
    def split_data(self, validation_size=0.2):
        if self.X_train_orig is None or self.y_train_orig is None:
             print("エラー: 元データが読み込まれていません", file=sys.stderr)
             return

        # 1. まず NumPy 配列のまま分割
        X_train_split, X_val_split, self.y_train, self.y_val = train_test_split(
            self.X_train_orig, # 元の (N, 28, 28) データ
            self.y_train_orig,
            test_size=validation_size,
            random_state=42
        )

        # 2. ★ 分割後にそれぞれを前処理 (正規化 + チャネル追加)
        self.X_train = self._preprocess(X_train_split)
        self.X_val = self._preprocess(X_val_split)
        # テストデータも同様に前処理
        self.X_test = self._preprocess(self.X_test_orig)

        print(f"訓練データを分割し、前処理しました (検証用: {validation_size*100}%)")
        print(f"  新 X_train shape: {self.X_train.shape}") # (48000, 28, 28, 1) になるはず
        print(f"  新 y_train shape: {self.y_train.shape}")
        print(f"  X_val shape: {self.X_val.shape}")       # (12000, 28, 28, 1) になるはず
        print(f"  y_val shape: {self.y_val.shape}")
        print(f"  X_test shape: {self.X_test.shape}")      # (10000, 28, 28, 1) になるはず


    # --- train, evaluate, predict_for_submission は MLP版とほぼ同じ ---
    # (入力データの形状が変わっただけ)

    def train(self):
        if self.X_train is None or self.y_train is None:
            print("エラー: 訓練データがありません", file=sys.stderr)
            return

        print(f"Keras CNN モデルの訓練を開始します (Epochs: {self.epochs}, Batch Size: {self.batch_size})...")

        self.history = self.model.fit(
            self.X_train, # (N, 28, 28, 1) 形式のデータ
            self.y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_data=(self.X_val, self.y_val), # (N, 28, 28, 1) 形式のデータ
            verbose=1
        )
        print("モデルの訓練が完了しました。")

    def evaluate(self):
        if self.model is None or self.X_val is None or self.y_val is None:
            print("エラー: モデルまたは検証データがありません", file=sys.stderr)
            return

        print("\n--- モデル評価 (検証データ) ---")
        loss, accuracy = self.model.evaluate(self.X_val, self.y_val, verbose=0)
        print(f"Keras .evaluate() の結果:")
        print(f"  検証データの損失 (Loss): {loss:.4f}")
        print(f"  検証データの正解率 (Accuracy): {accuracy * 100:.2f} %")

        print("\n--- 分類レポート (検証データ) ---")
        y_pred_proba = self.model.predict(self.X_val)
        y_pred_val = np.argmax(y_pred_proba, axis=1)

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

    def predict_for_submission(self, output_csv_path):
        if self.model is None or self.X_test is None: # X_test_processed ではなく X_test
             print("エラー: モデルまたはテストデータがありません", file=sys.stderr)
             return

        print(f"\n最終的なテストデータ (x_test.npy) に対する予測を実行します...")
        y_pred_proba = self.model.predict(self.X_test) # (N, 28, 28, 1) 形式のデータ
        y_pred_test = np.argmax(y_pred_proba, axis=1)

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
    OUTPUT_CSV_PATH = 'data/classification/submission_keras_cnn.csv'

    # モデルのインスタンス化 (CNNはMLPより学習に時間がかかることがある)
    cnn_model = KerasCNNModel(epochs=10, batch_size=32)

    cnn_model.load_data(X_TRAIN_PATH, Y_TRAIN_PATH, X_TEST_PATH)
    cnn_model.split_data(validation_size=0.2)
    cnn_model.train()
    cnn_model.evaluate()
    cnn_model.predict_for_submission(OUTPUT_CSV_PATH)