

# NumPyのみでサポートベクターマシン (SVM) を実装するコード
# 線形カーネル、One-vs-Rest戦略、勾配降下法 (BGD) を用いる
# 二次元のピクセルデータ（N ,28,28）を平坦化して（N,784）を作り、線型写像して、(N,10)にする。スコアを「境界線からの距離」として扱い、「ヒンジ損失」が最小になる（＝マージンが最大になる）ように学習する。予測では10個の分類器が出した「生のスコア」のうち、最も高いスコアを出したクラスを最終的な予測結果とします。
# Accuracyは大体0.83くらいになる。
# --- モデル評価 (検証データ) ---
# 検証データの正解率 (Accuracy): 82.67 %

# 分類レポート (検証データ):
#               precision    recall  f1-score   support

#  T-shirt/top       0.77      0.82      0.80      1182
#      Trouser       0.96      0.95      0.95      1222
#     Pullover       0.70      0.71      0.71      1210
#        Dress       0.82      0.85      0.83      1153
#         Coat       0.68      0.80      0.74      1180
#       Sandal       0.92      0.89      0.91      1215
#        Shirt       0.67      0.49      0.57      1239
#      Sneaker       0.89      0.90      0.89      1228
#          Bag       0.94      0.94      0.94      1185
#   Ankle boot       0.89      0.93      0.91      1186

#     accuracy                           0.83     12000
#    macro avg       0.83      0.83      0.82     12000
# weighted avg       0.83      0.83      0.82     12000

import numpy as np
import pandas as pd
import sys
# 以下のライブラリは「モデルの実装」ではなく「データ準備」と「評価」のために使用します
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

class NumpySVMModel:
    """
    サポートベクターマシン (SVM) を
    NumPyのみで実装するクラス。
    (線形カーネル, One-vs-Rest, 勾配降下法)
    """
    
    def __init__(self, learning_rate=0.1, n_iterations=1000, alpha=0.01, random_state=42, verbose=True):
        """
        ハイパーパラメータを初期化
        learning_rate: 学習率 (BGD SVMは小さめが安定)
        n_iterations: 勾配降下法の反復回数
        alpha: 正則化の強さ (1/C に相当)
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.alpha = alpha # L2正則化の強さ
        self.random_state = random_state
        self.verbose = verbose
        self.weights = None # (特徴量, クラス数)
        self.bias = None    # (1, クラス数)
        self.n_classes = 10
        np.random.seed(self.random_state)

    # --- load_data, _preprocess は logistic_numpy と全く同じ ---
    
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

    # --- split_data も logistic_numpy とほぼ同じ (one_hot を削除) ---
    
    def split_data(self, validation_size=0.2):
        if self.X_train_processed is None or self.y_train_orig is None:
            print("エラー: 訓練データが読み込まれていません", file=sys.stderr)
            return

        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            self.X_train_processed,
            self.y_train_orig,
            test_size=validation_size,
            random_state=self.random_state
        )
        # SVM (OvR) は [1, 9, 0] の形式のまま扱うので one_hot 変換は不要
        
        print(f"訓練データを分割しました (検証用: {validation_size*100}%)")
        print(f"  新 X_train shape: {self.X_train.shape}")
        print(f"  新 y_train shape: {self.y_train.shape}")
        print(f"  X_val shape: {self.X_val.shape}")
        print(f"  y_val shape: {self.y_val.shape}")

    # --- _softmax は不要 ---
    
    # --- train メソッドを SVM (OvR, BGD) 用に書き換え ---

    def train(self):
        """
        NumPyによる勾配降下法でSVM (One-vs-Rest) を訓練する
        """
        if self.X_train is None or self.y_train is None:
            print("エラー: 訓練データがありません (split_data() を先に実行してください)", file=sys.stderr)
            return

        print(f"NumPy SVM (OvR) の訓練を開始します (lr={self.learning_rate}, iter={self.n_iterations}, alpha={self.alpha})...")
        n_samples, n_features = self.X_train.shape
        
        # 1. 重み(W)とバイアス(b)を初期化 (ロジスティック回帰と同じ)
        # W: (784 features, 10 classes)
        # b: (1, 10 classes)
        self.weights = np.random.randn(n_features, self.n_classes) * 0.01
        self.bias = np.zeros((1, self.n_classes))
        
        # 2. One-vs-Rest (OvR) 戦略: 10個の分類器を個別に訓練
        for k in range(self.n_classes):
            if self.verbose:
                print(f"  -- クラス {k} (vs Rest) の分類器を訓練中 --")
            
            # 2a. ラベルの準備 (クラスk = +1, それ以外 = -1)
            # SVMは +1 と -1 でラベルを扱うのが標準
            y_binary = np.where(self.y_train == k, 1, -1).reshape(-1, 1) # (N, 1)
            
            # このクラスk専用の重みとバイアス
            w_k = self.weights[:, k].reshape(-1, 1) # (784, 1)
            b_k = self.bias[:, k].reshape(-1, 1)   # (1, 1)

            # 2b. 勾配降下法のループ (このクラスk専用)
            for i in range(self.n_iterations):
                # 線形スコア (ロジスティック回帰と同じ)
                z = self.X_train.dot(w_k) + b_k # (N, 1)
                
                # 2c. ヒンジ損失 (Hinge Loss) の計算
                # 損失 = 1 - y_binary * z
                # マージン内 (損失 > 0) のデータ（= 不正解またはマージン内の正解）を見つける
                margin_mask = (1 - y_binary * z) > 0 # (N, 1) boolean
                
                # 2d. 損失(Loss)の計算 (verbose用)
                if self.verbose and (i % 100 == 0 or i == self.n_iterations - 1):
                    # 損失 = 正則化項 + ヒンジ損失項
                    reg_loss = self.alpha / 2 * np.sum(w_k * w_k)
                    hinge_loss = np.sum(1 - y_binary[margin_mask] * z[margin_mask]) / n_samples
                    total_loss = reg_loss + hinge_loss
                    print(f"    Epoch {i}, Loss: {total_loss:.4f} (Violators: {np.sum(margin_mask)})")

                # 2e. 勾配(Gradients)の計算
                # 勾配 = 正則化項の勾配 + ヒンジ損失項の勾配
                
                # 正則化項の勾配 (L2)
                dW_reg = self.alpha * w_k # (784, 1)
                
                # --- ここからが修正箇所 ---
                
                # マージン違反をしているサンプル数を取得
                M = np.sum(margin_mask)
                
                if M > 0:
                    # ★ バグ修正 1: y_binary[margin_mask.flatten()] を使い (M, 1) の 2D配列にする
                    y_violators = y_binary[margin_mask.flatten()].reshape(-1, 1) # (M, 1)
                    
                    # ★ バグ修正 2: X_train[margin_mask.flatten()] を使う
                    X_violators = self.X_train[margin_mask.flatten()] # (M, 784)
                    
                    # (784, M) .dot (M, 1) -> (784, 1)
                    dW_loss = - (X_violators.T.dot(y_violators)) / n_samples
                    
                    # ★ バグ修正 3: db_loss は .mean() ではなく .sum() / n_samples
                    db_loss = - np.sum(y_violators) / n_samples
                else:
                    # マージン違反がなければ勾配は 0
                    dW_loss = 0
                    db_loss = 0

                # --- 修正ここまで ---
                
                # 合計の勾配
                dW = dW_reg + dW_loss # (784, 1) + (784, 1) or scalar
                db = db_loss # scalar
                
                # 2f. パラメータの更新
                w_k -= self.learning_rate * dW
                b_k -= self.learning_rate * db
                
            # 訓練済みの w_k, b_k を保存
            self.weights[:, k] = w_k.flatten()
            self.bias[:, k] = b_k.flatten()
            
        print("モデルの訓練が完了しました。")

    # --- predict メソッドを SVM (OvR) 用に書き換え ---

    def predict(self, X):
        """
        入力Xに対する「予測ラベル」を計算する
        """
        if self.weights is None or self.bias is None:
            raise RuntimeError("モデルが訓練されていません。train()を先に実行してください。")
        
        # (N, 784) . (784, 10) + (1, 10) -> (N, 10)
        # 10個の分類器すべての「生スコア」を計算
        scores = X.dot(self.weights) + self.bias
        
        # One-vs-Rest のルール:
        # 最も高いスコアを出した分類器のクラスを予測結果とする
        return np.argmax(scores, axis=1)

    # --- evaluate, predict_for_submission は logistic_numpy と全く同じ ---
    # (内部で呼ぶ self.predict() がSVM版に切り替わっているため)

    def evaluate(self):
        """
        (分割して作成した) 検証用データでモデルの精度を評価する
        """
        if self.X_val is None or self.y_val is None:
            print("エラー: 検証データがありません", file=sys.stderr)
            return

        print("\n--- モデル評価 (検証データ) ---")
        y_pred_val = self.predict(self.X_val) # ★ NumPy SVM の predict が呼ばれる
        
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
        """
        if self.X_test_processed is None:
            print("エラー: テストデータがありません", file=sys.stderr)
            return
            
        print(f"\n最終的なテストデータ (x_test.npy) に対する予測を実行します...")
        y_pred_test = self.predict(self.X_test_processed) # ★ NumPy SVM の predict が呼ばれる
        
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
    OUTPUT_CSV_PATH = 'data/classification/submission_numpy_svm.csv'

    # 1. ★ モデルのインスタンス化
    # (ハイパーパラメータは調整が必要かもしれません)
    numpy_model = NumpySVMModel(learning_rate=0.1, n_iterations=1000, alpha=0.01, verbose=True)
    
    # 2. データの読み込みと前処理
    numpy_model.load_data(X_TRAIN_PATH, Y_TRAIN_PATH, X_TEST_PATH)
    
    # 3. 訓練データを「訓練用」と「検証用」に分割
    numpy_model.split_data(validation_size=0.2)
    
    # 4. ★ モデルの訓練 (SVMの核心部)
    numpy_model.train()
    
    # 5. モデルの評価
    numpy_model.evaluate()
    
    # 6. 最終的な予測の実行
    numpy_model.predict_for_submission(OUTPUT_CSV_PATH)