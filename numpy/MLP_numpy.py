#numpyを用いたMLPモデルの構築・評価クラス
#実行はしなかった。
import numpy as np
import pandas as pd
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

class NumpyMLPModel:
    """
    1隠れ層を持つMLPをNumPyのみで実装するクラス。
    """

    def __init__(self, input_dim=784, hidden_dim=128, output_dim=10,
                 learning_rate=0.1, n_iterations=1000, batch_size=None, # batch_size=None でBGD
                 random_state=42, verbose=True):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.batch_size = batch_size
        self.random_state = random_state
        self.verbose = verbose
        self.n_classes = output_dim # Fashion MNISTは10クラス

        np.random.seed(self.random_state)
        # ★ 重みとバイアスを2層分初期化
        self.W1 = np.random.randn(self.input_dim, self.hidden_dim) * 0.01 # (784, 128)
        self.b1 = np.zeros((1, self.hidden_dim))                         # (1, 128)
        self.W2 = np.random.randn(self.hidden_dim, self.output_dim) * 0.01 # (128, 10)
        self.b2 = np.zeros((1, self.output_dim))                          # (1, 10)

        # --- (データ格納用変数は同じ) ---
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_train_processed = None
        self.X_test_processed = None
        self.y_train_orig = None
        self.y_train_one_hot = None # OneHot形式の訓練ラベル

    # --- load_data, _preprocess, _one_hot, split_data は Logistic 版と同じ ---
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
            data = data.reshape(data.shape[0], -1)
        data = data.astype('float32') / 255.0
        return data

    def _one_hot(self, y):
        one_hot_y = np.zeros((y.shape[0], self.n_classes))
        one_hot_y[np.arange(y.shape[0]), y] = 1
        return one_hot_y

    def split_data(self, validation_size=0.2):
        if self.X_train_processed is None or self.y_train_orig is None:
            print("エラー: 訓練データが読み込まれていません", file=sys.stderr)
            return
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            self.X_train_processed, self.y_train_orig,
            test_size=validation_size, random_state=self.random_state
        )
        self.y_train_one_hot = self._one_hot(self.y_train)
        print(f"訓練データを分割しました (検証用: {validation_size*100}%)")

    # --- 活性化関数とその微分 ---
    def _relu(self, z):
        return np.maximum(0, z)

    def _relu_derivative(self, z):
        # ReLUの微分: z > 0 なら 1, それ以外は 0
        return (z > 0).astype(z.dtype)

    def _softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    # --- ★ train メソッドを MLP (BGD) 用に書き換え ---
    def train(self):
        if self.X_train is None or self.y_train_one_hot is None:
            print("エラー: 訓練データがありません", file=sys.stderr)
            return

        print(f"NumPy MLP モデルの訓練を開始します (lr={self.learning_rate}, iter={self.n_iterations})...")
        n_samples = self.X_train.shape[0]

        # 勾配降下法のループ (BGD: バッチ勾配降下法)
        for i in range(self.n_iterations):
            # === 1. フォワードプロパゲーション ===
            # (入力層 -> 隠れ層)
            z1 = self.X_train.dot(self.W1) + self.b1 # (N, 128)
            a1 = self._relu(z1)                     # (N, 128)
            # (隠れ層 -> 出力層)
            z2 = a1.dot(self.W2) + self.b2          # (N, 10)
            a2 = self._softmax(z2)                  # (N, 10) (最終予測確率)

            # === 2. 損失(Loss)の計算 ===
            if self.verbose and (i % 100 == 0 or i == self.n_iterations - 1):
                # クロスエントロピー損失
                loss = -np.mean(self.y_train_one_hot * np.log(a2 + 1e-9))
                print(f"  Epoch {i}, Loss: {loss:.4f}")

            # === 3. バックプロパゲーション (勾配計算) ===
            # (出力層の勾配) - ソフトマックス+クロスエントロピーの微分
            dz2 = a2 - self.y_train_one_hot         # (N, 10)

            # (出力層の重み W2 とバイアス b2 の勾配)
            dW2 = (1 / n_samples) * a1.T.dot(dz2)   # (128, 10)
            db2 = (1 / n_samples) * np.sum(dz2, axis=0, keepdims=True) # (1, 10)

            # (隠れ層の勾配) - 連鎖律 (Chain Rule)
            # da1: 出力層の勾配 dz2 が隠れ層の出力 a1 にどう影響するか
            da1 = dz2.dot(self.W2.T)                # (N, 128)
            # dz1: 隠れ層の活性化前 z1 にどう影響するか (ReLUの微分を掛ける)
            dz1 = da1 * self._relu_derivative(z1)   # (N, 128)

            # (隠れ層の重み W1 とバイアス b1 の勾配)
            dW1 = (1 / n_samples) * self.X_train.T.dot(dz1) # (784, 128)
            db1 = (1 / n_samples) * np.sum(dz1, axis=0, keepdims=True) # (1, 128)

            # === 4. パラメータの更新 ===
            self.W1 -= self.learning_rate * dW1
            self.b1 -= self.learning_rate * db1
            self.W2 -= self.learning_rate * dW2
            self.b2 -= self.learning_rate * db2

        print("モデルの訓練が完了しました。")

    # --- predict メソッドを MLP 用に書き換え ---
    def predict(self, X):
        if self.W1 is None:
            raise RuntimeError("モデルが訓練されていません。")
        # フォワードプロパゲーションを実行
        z1 = X.dot(self.W1) + self.b1
        a1 = self._relu(z1)
        z2 = a1.dot(self.W2) + self.b2
        a2 = self._softmax(z2) # 最終的な確率
        # 最も確率が高いクラスを返す
        return np.argmax(a2, axis=1)

    # --- evaluate, predict_for_submission は Logistic 版とほぼ同じ ---
    def evaluate(self):
        if self.X_val is None or self.y_val is None:
            print("エラー: 検証データがありません", file=sys.stderr)
            return
        print("\n--- モデル評価 (検証データ) ---")
        y_pred_val = self.predict(self.X_val) # ★ MLP の predict が呼ばれる
        accuracy = accuracy_score(self.y_val, y_pred_val)
        print(f"検証データの正解率 (Accuracy): {accuracy * 100:.2f} %")
        # (分類レポート表示は省略、Logistic版と同じ)
        return accuracy

    def predict_for_submission(self, output_csv_path):
        if self.X_test_processed is None:
             print("エラー: テストデータがありません", file=sys.stderr)
             return
        print(f"\n最終的なテストデータ (x_test.npy) に対する予測を実行します...")
        y_pred_test = self.predict(self.X_test_processed) # ★ MLP の predict が呼ばれる
        submission_df = pd.DataFrame({"ID": np.arange(1, len(y_pred_test) + 1), "Label": y_pred_test})
        try:
            submission_df.to_csv(output_csv_path, index=False)
            print(f"予測結果を {output_csv_path} に保存しました。")
        except Exception as e:
            print(f"エラー: CSVファイルの保存に失敗しました - {e}", file=sys.stderr)

# --- main 部分 ---
if __name__ == "__main__":
    X_TRAIN_PATH = 'data/classification/x_train.npy'
    Y_TRAIN_PATH = 'data/classification/y_train.npy'
    X_TEST_PATH = 'data/classification/x_test.npy'
    OUTPUT_CSV_PATH = 'data/classification/submission_numpy_mlp.csv'

    # MLPモデルのインスタンス化 (学習率や反復回数は調整が必要かも)
    mlp_model = NumpyMLPModel(learning_rate=0.1, n_iterations=1000, verbose=True)

    mlp_model.load_data(X_TRAIN_PATH, Y_TRAIN_PATH, X_TEST_PATH)
    mlp_model.split_data(validation_size=0.2)
    mlp_model.train() # ★ MLP の訓練実行
    mlp_model.evaluate()
    mlp_model.predict_for_submission(OUTPUT_CSV_PATH)