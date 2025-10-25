# numpyを用いてk-NN (k-Nearest Neighbors) を実装するコード
#めちゃくちゃ遅かった。
# --- モデル評価 (検証データ) ---
# （k-NNの予測計算中です...）
# 検証データの正解率 (Accuracy): 85.16 %

# 分類レポート (検証データ):
#               precision    recall  f1-score   support

#  T-shirt/top       0.75      0.87      0.80      1182
#      Trouser       0.99      0.97      0.98      1222
#     Pullover       0.73      0.80      0.76      1210
#        Dress       0.90      0.86      0.88      1153
#         Coat       0.77      0.77      0.77      1180
#       Sandal       0.99      0.82      0.90      1215
#        Shirt       0.67      0.57      0.62      1239
#      Sneaker       0.87      0.95      0.91      1228
#          Bag       0.98      0.95      0.97      1185
#   Ankle boot       0.88      0.97      0.92      1186

#     accuracy                           0.85     12000
#    macro avg       0.85      0.85      0.85     12000
# weighted avg       0.85      0.85      0.85     12000
import numpy as np
import pandas as pd
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

class NumpyKNNModel:
    """
    k-NN (k-Nearest Neighbors) を
    NumPyのみで実装するクラス。
    """
    
    def __init__(self, n_neighbors=5):
        """
        ハイパーパラメータを初期化
        n_neighbors: 'k'の値
        """
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.y_train = None
        # --- self.model は NumPy版には不要 ---
        
        # --- NumPy版で必要な変数を __init__ で初期化 ---
        self.X_val = None
        self.y_val = None
        self.X_test_processed = None
        self.y_train_orig = None
        self.X_train_processed = None


    # --- load_data, _preprocess は 
    # --- NumpySVMModel と全く同じ (one_hot不要) ---
    
    def load_data(self, x_train_path, y_train_path, x_test_path):
        try:
            X_train_orig = np.load(x_train_path)
            self.y_train_orig = np.load(y_train_path) # 分割前のy
            X_test_orig = np.load(x_test_path)
            
            print("データの読み込み完了")
            print(f"  X_train (元): {X_train_orig.shape}")
            print(f"  y_train (元): {self.y_train_orig.shape}")
            print(f"  X_test (最終予測用): {X_test_orig.shape}")

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
            random_state=42 # sklearn版とseedを合わせる
        )
        print(f"訓練データを分割しました (検証用: {validation_size*100}%)")
        print(f"  新 X_train shape: {self.X_train.shape}")
        print(f"  新 y_train shape: {self.y_train.shape}")
        print(f"  X_val shape: {self.X_val.shape}")
        print(f"  y_val shape: {self.y_val.shape}")

    # --- train メソッドを k-NN (Lazy Learner) 用に書き換え ---

    def train(self):
        """
        k-NNの「訓練」は、データを丸暗記するだけ。
        (勾配降下法などの計算は行わない)
        """
        if self.X_train is None or self.y_train is None:
            print("エラー: 訓練データがありません (split_data() を先に実行してください)", file=sys.stderr)
            return

        print(f"k-NNモデルの訓練（データの記憶）を開始します (k={self.n_neighbors})...")
        # .fit() は単にデータをselfに保存するだけ (既に split_data で保存されている)
        print("モデルの訓練（データの記憶）が完了しました。")

    # --- predict メソッドを k-NN (NumPy) 用に書き換え ---

    def predict(self, X_test):
        """
        入力X_testの各点について、訓練データとの距離を計算し、
        k個の近傍の投票によって予測する。
        """
        if self.X_train is None or self.y_train is None:
            raise RuntimeError("モデルが訓練（記憶）されていません。train()を先に実行してください。")
        
        print("（k-NNの予測計算中です...）")
        
        # 予測結果を格納する配列
        y_pred = []
        
        # --- k-NNの核心: 予測データ1点ずつループ ---
        # 進捗がわかるように tqdm を使う (オプション)
        # try:
        #     from tqdm import tqdm
        #     iterator = tqdm(X_test, desc="Predicting")
        # except ImportError:
        iterator = X_test

        for x_test_point in iterator:
            
            # 1. 距離の計算 (NumPyブロードキャスト)
            # (x_test_point と 48,000件の X_train の全データとのユークリッド距離)
            # (X_train - x_test_point) は (48000, 784) - (784,) -> (48000, 784)
            distances = np.sqrt(np.sum((self.X_train - x_test_point)**2, axis=1))
            
            # 2. 距離でソートし、k個のインデックスを取得
            # (distances は (48000,) の配列)
            nearest_indices = np.argsort(distances)[:self.n_neighbors]
            
            # 3. k個の近傍のラベルを取得
            # (y_train は (48000,) の配列)
            nearest_labels = self.y_train[nearest_indices]
            
            # 4. 投票 (Vote)
            # k個のラベルの中で、最も出現回数が多いラベルを予測結果とする
            # np.bincount は各数値の出現回数をカウントする
            prediction = np.bincount(nearest_labels).argmax()
            
            y_pred.append(prediction)
            
        return np.array(y_pred)

    # --- evaluate, predict_for_submission を修正 ---

    def evaluate(self):
        """
        (分割して作成した) 検証用データでモデルの精度を評価する
        ★ k-NNの場合、predict()が呼ばれるため、非常に時間がかかる
        """
        # ★ 修正点: self.model のチェックを self.X_train/y_train に変更
        if self.X_train is None or self.y_train is None or self.X_val is None or self.y_val is None:
            print("エラー: モデルが訓練されていないか、検証データがありません", file=sys.stderr)
            return

        print("\n--- モデル評価 (検証データ) ---")
        y_pred_val = self.predict(self.X_val) # ★ NumPy k-NN の predict が呼ばれる
        
        accuracy = accuracy_score(self.y_val, y_pred_val)
        print(f"検証データの正解率 (Accuracy): {accuracy * 100:.2f} %")
        
        class_names = [
            "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
            "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
        ]
        try:
            report = classification_report(self.y_val, y_pred_val, target_names=class_names, zero_division=0)
            print("\n分類レポート (検証データ):")
            print(report)
        except Exception as e:
            print(f"分類レポートの生成に失敗: {e}")

        return accuracy

    def predict_for_submission(self, output_csv_path):
        """
        x_test.npy に対する最終的な予測をCSVに保存する
        ★ k-NNの場合、predict()が呼ばれるため、非常に時間がかかる
        """
        # ★ 修正点: self.model のチェックを self.X_train/y_train に変更
        if self.X_train is None or self.y_train is None or self.X_test_processed is None:
            print("エラー: モデルが訓練されていないか、テストデータがありません", file=sys.stderr)
            return
            
        print(f"\n最終的なテストデータ (x_test.npy) に対する予測を実行します...")
        y_pred_test = self.predict(self.X_test_processed) # ★ NumPy k-NN の predict が呼ばれる
        
        submission_df = pd.DataFrame({
            "ID": np.arange(1, len(y_pred_test) + 1),
            "Label": y_pred_test
        })

        try:
            submission_df.to_csv(output_csv_path, index=False)
            print(f"予測結果を {output_csv_path} に保存しました。")
            print("\n--- 保存されたCSV (先頭5件) ---")
            print(submission_df.head())
        except Exception as e:
            print(f"エラー: CSVファイルの保存に失敗しました - {e}", file=sys.stderr)


if __name__ == "__main__":
    # データパス
    X_TRAIN_PATH = 'data/classification/x_train.npy'
    Y_TRAIN_PATH = 'data/classification/y_train.npy'
    X_TEST_PATH = 'data/classification/x_test.npy'
    
    # ★ 提出用CSVの保存先パス
    OUTPUT_CSV_PATH = 'data/classification/submission_numpy_knn.csv'

    # 1. ★ モデルのインスタンス化
    numpy_model = NumpyKNNModel(n_neighbors=5)
    
    # 2. データの読み込みと前処理
    numpy_model.load_data(X_TRAIN_PATH, Y_TRAIN_PATH, X_TEST_PATH)
    
    # 3. 訓練データを「訓練用」と「検証用」に分割
    numpy_model.split_data(validation_size=0.2)
    
    # 4. ★ モデルの訓練 (データの記憶)
    numpy_model.train()
    
    # 5. ★ モデルの評価 (NumPy k-NNの核心部)
    # (sklearn版より遅い可能性が高いです)
    numpy_model.evaluate()
    
    # 6. 最終的な予測の実行
    numpy_model.predict_for_submission(OUTPUT_CSV_PATH)