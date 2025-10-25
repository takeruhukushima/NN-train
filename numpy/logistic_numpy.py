# numpy onlyでロジスティック回帰（ソフトマックス回帰）を実装するコード
# solver（Optimizer）はBGDを利用している。
# L2正則化は実装していない。
#二次元のピクセルデータ（N ,28,28）を平坦化して（N,784）を作り、線型写像して、(N,10)にする。それでソフトマックスを使って確率化して、ワンホットの答えとクロスエントロピー誤差を用いて比べる
#Accuracyは大体0.85くらいになる。
import numpy as np
import pandas as pd
import sys
# 以下のライブラリは「モデルの実装」ではなく「データ準備」と「評価」のために使用します
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

class NumpyLogisticRegressionModel:
    """
    ロジスティック回帰（ソフトマックス回帰）を
    NumPyのみで実装するクラス。
    """
    
    def __init__(self, learning_rate=0.1, n_iterations=1000, random_state=42, verbose=True):
        """
        ハイパーパラメータを初期化
        learning_rate: 学習率
        n_iterations: 勾配降下法の反復回数 (sklearnのmax_iterに相当)
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.random_state = random_state
        self.verbose = verbose
        self.weights = None # (特徴量 x クラス数)
        self.bias = None    # (1 x クラス数)
        self.n_classes = 10 # Fashion MNISTは10クラス
        np.random.seed(self.random_state) # 重みの初期化のため

    def load_data(self, x_train_path, y_train_path, x_test_path):
        """
        Numpy (.npy) ファイルからデータを読み込む (sklearn版と同じ)
        """
        try:
            X_train_orig = np.load(x_train_path)
            self.y_train_orig = np.load(y_train_path) # 分割前のy
            X_test_orig = np.load(x_test_path)
            
            print("データの読み込み完了")
            print(f"  X_train (元): {X_train_orig.shape}")
            print(f"  y_train (元): {self.y_train_orig.shape}")
            print(f"  X_test (最終予測用): {X_test_orig.shape}")

            # 前処理 (Flatten と Normalization)
            self.X_train_processed = self._preprocess(X_train_orig)
            self.X_test_processed = self._preprocess(X_test_orig) 
            print("データの前処理 (Flatten, Normalization) 完了")

        except FileNotFoundError as e:
            print(f"エラー: ファイルが見つかりません - {e}", file=sys.stderr)
            sys.exit(1)

    def _preprocess(self, data):
        """
        データの前処理 (sklearn版と同じ)
        """
        if data.ndim == 3:
            data = data.reshape(data.shape[0], -1) # (N, 28, 28) -> (N, 784)
        data = data.astype('float32') / 255.0
        return data

    def _one_hot(self, y):
        """
        ラベル (例: [1, 0, 9]) を One-Hot 形式 (例: [[0,1,0..], [1,0,0..], [0,0,..1]]) に変換
        """
        one_hot_y = np.zeros((y.shape[0], self.n_classes))
        one_hot_y[np.arange(y.shape[0]), y] = 1
        return one_hot_y

    def split_data(self, validation_size=0.2):
        """
        訓練データを訓練用と検証用に分割 (sklearn版とほぼ同じ)
        """
        if self.X_train_processed is None or self.y_train_orig is None:
            print("エラー: 訓練データが読み込まれていません", file=sys.stderr)
            return

        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            self.X_train_processed,
            self.y_train_orig,
            test_size=validation_size,
            random_state=self.random_state
        )
        
        # ★ NumPy実装のキモ ★
        # 訓練(計算)のために、y_trainをOne-Hot形式に変換しておく
        self.y_train_one_hot = self._one_hot(self.y_train)
        # y_valは評価時に使うだけなので [1, 9, 0] の形式のまま保持
        
        print(f"訓練データを分割しました (検証用: {validation_size*100}%)")
        print(f"  新 X_train shape: {self.X_train.shape}")
        print(f"  新 y_train (One-Hot) shape: {self.y_train_one_hot.shape}")
        print(f"  X_val shape: {self.X_val.shape}")
        print(f"  y_val shape: {self.y_val.shape}")

    def _softmax(self, z):
        """
        ソフトマックス関数
        (入力zの各行について、合計が1になる確率に変換する)
        """
        # オーバーフロー防止のため、各行の最大値を引く
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def train(self):
        """
        NumPyによる勾配降下法でモデルを訓練する
        """
        if self.X_train is None or self.y_train_one_hot is None:
            print("エラー: 訓練データがありません (split_data() を先に実行してください)", file=sys.stderr)
            return

        print(f"NumPyモデルの訓練を開始します (lr={self.learning_rate}, iter={self.n_iterations})...")
        n_samples, n_features = self.X_train.shape
        
        # 1. 重み(W)とバイアス(b)を初期化
        # W: (784 features, 10 classes)
        # b: (1, 10 classes)
        self.weights = np.random.randn(n_features, self.n_classes) * 0.01
        self.bias = np.zeros((1, self.n_classes))
        
        # 2. 勾配降下法のループ
        for i in range(self.n_iterations):
            # 2a. 予測 (フォワードプロパゲーション)
            # z = X * W + b
            z = self.X_train.dot(self.weights) + self.bias
            # a = softmax(z) ... (N, 10) の確率行列
            a = self._softmax(z) 
            
            # 2b. 損失(Loss)の計算 (verbose用)
            if self.verbose and (i % 100 == 0 or i == self.n_iterations - 1):
                # カテゴリカル・クロスエントロピー損失
                # (1e-9は log(0) を防ぐための微小値)
                loss = -np.mean(self.y_train_one_hot * np.log(a + 1e-9))
                print(f"  Epoch {i}, Loss: {loss:.4f}")

            # 2c. 勾配(Gradients)の計算 (バックプロパゲーション)
            # 誤差 (予測確率 - 正解OneHot)
            dz = a - self.y_train_one_hot  # (N, 10)
            
            # 重みの勾配
            dW = (1 / n_samples) * self.X_train.T.dot(dz) # (784, 10)
            # バイアスの勾配
            db = (1 / n_samples) * np.sum(dz, axis=0, keepdims=True) # (1, 10)
            
            # 2d. パラメータの更新
            self.weights -= self.learning_rate * dW
            self.bias -= self.learning_rate * db
            
        print("モデルの訓練が完了しました。")

    def _predict_proba(self, X):
        """
        入力Xに対する各クラスの「確率」を計算する
        """
        if self.weights is None or self.bias is None:
            raise RuntimeError("モデルが訓練されていません。train()を先に実行してください。")
        z = X.dot(self.weights) + self.bias
        return self._softmax(z)

    def predict(self, X):
        """
        入力Xに対する「予測ラベル」を計算する
        """
        # (N, 10) の確率行列を取得
        probabilities = self._predict_proba(X)
        # 最も確率が高い列の「インデックス」(=クラスラベル)を返す
        return np.argmax(probabilities, axis=1)

    def evaluate(self):
        """
        (分割して作成した) 検証用データでモデルの精度を評価する
        """
        if self.X_val is None or self.y_val is None:
            print("エラー: 検証データがありません", file=sys.stderr)
            return

        print("\n--- モデル評価 (検証データ) ---")
        y_pred_val = self.predict(self.X_val) # (N,) の予測ラベル
        
        # 評価は (N,) の予測と (N,) の正解ラベル (y_val) で行う
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
        x_test.npy に対する最終的な予測をCSVに保存する (sklearn版と同じ)
        """
        if self.X_test_processed is None:
            print("エラー: テストデータがありません", file=sys.stderr)
            return
            
        print(f"\n最終的なテストデータ (x_test.npy) に対する予測を実行します...")
        y_pred_test = self.predict(self.X_test_processed)
        
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
    OUTPUT_CSV_PATH = 'data/classification/submission_numpy_logistic.csv'

    # 1. モデルのインスタンス化
    # (n_iterations=1000, learning_rate=0.1 は良い出発点です)
    numpy_model = NumpyLogisticRegressionModel(learning_rate=0.1, n_iterations=1000, verbose=True)
    
    # 2. データの読み込みと前処理
    numpy_model.load_data(X_TRAIN_PATH, Y_TRAIN_PATH, X_TEST_PATH)
    
    # 3. 訓練データを「訓練用」と「検証用」に分割
    numpy_model.split_data(validation_size=0.2)
    
    # 4. モデルの訓練 (★ ここがNumPy実装の核心 ★)
    numpy_model.train()
    
    # 5. モデルの評価 (NumPy実装のpredict()が使われる)
    numpy_model.evaluate()
    
    # 6. 最終的な予測の実行
    numpy_model.predict_for_submission(OUTPUT_CSV_PATH)