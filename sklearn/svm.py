#Accuracyは大体0.85くらいになる。
import numpy as np
import pandas as pd
# ★ ここを変更: LogisticRegression の代わりに LinearSVC をインポート
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import sys

# ★ クラス名を変更
class SVMModel:
    """
    サポートベクターマシン (LinearSVC) を訓練・評価するクラス。
    """
    
    # ★ __init__ を LinearSVC 用に変更
    def __init__(self, max_iter=2000, random_state=42, C=1.0):
        """
        LinearSVC のハイパーパラメータを初期化
        C: 正則化パラメータ
        max_iter: 最大反復回数 (LogisticRegressionより収束が遅いことがあるため少し多めに)
        """
        self.model = LinearSVC(
            C=C,
            max_iter=max_iter, 
            random_state=random_state,
            verbose=1 # ★ 学習の進捗を表示
        )
        self.X_train = None
        self.y_train = None
        self.X_val = None # 検証用データ (X)
        self.y_val = None # 検証用データ (y)
        self.X_test = None # 最終的な予測用データ
        self.X_train_processed = None # 前処理済み訓練データ
        self.X_test_processed = None # 前処理済みテストデータ
        self.y_train_orig = None # 分割前のy

    # --- 以下のメソッドは logistic.py と全く同じ ---

    def load_data(self, x_train_path, y_train_path, x_test_path):
        """
        Numpy (.npy) ファイルからデータを読み込む
        """
        try:
            X_train_orig = np.load(x_train_path)
            self.y_train_orig = np.load(y_train_path)
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
        """
        データの前処理 (Flatten と Normalization)
        LinearSVCもロジスティック回帰と同様にスケーリングが必要
        """
        if data.ndim == 3:
            data = data.reshape(data.shape[0], -1) # (N, 28, 28) -> (N, 784)
        data = data.astype('float32') / 255.0
        return data

    def split_data(self, validation_size=0.2):
        """
        訓練データを、さらに訓練用と検証用に分割する
        """
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
        print(f"  新 X_train shape: {self.X_train.shape}")
        print(f"  新 y_train shape: {self.y_train.shape}")
        print(f"  X_val shape: {self.X_val.shape}")
        print(f"  y_val shape: {self.y_val.shape}")


    def train(self):
        """
        (分割後の) 訓練データでモデルを訓練する
        """
        if self.X_train is None or self.y_train is None:
            print("エラー: 訓練データがありません (split_data() を先に実行してください)", file=sys.stderr)
            return

        print("モデルの訓練を開始します (LinearSVC)...")
        # .fit() を呼び出すだけ (ロジスティック回帰と同じ)
        self.model.fit(self.X_train, self.y_train)
        print("モデルの訓練が完了しました。")

    def evaluate(self):
        """
        (分割して作成した) 検証用データでモデルの精度を評価する
        """
        if self.model is None or self.X_val is None or self.y_val is None:
            print("エラー: モデルまたは検証データがありません", file=sys.stderr)
            return

        print("\n--- モデル評価 (検証データ) ---")
        y_pred_val = self.model.predict(self.X_val)
        
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
        except:
            report = classification_report(self.y_val, y_pred_val, zero_division=0)
            print("\n分類レポート (検証データ):")
            print(report)

        return accuracy

    def predict_for_submission(self, output_csv_path):
        """
        x_test.npy に対する最終的な予測（ラベル）をCSVに保存する
        """
        if self.model is None or self.X_test_processed is None:
            print("エラー: モデルまたはテストデータがありません", file=sys.stderr)
            return
            
        print(f"\n最終的なテストデータ (x_test.npy) に対する予測を実行します...")
        y_pred_test = self.model.predict(self.X_test_processed)
        
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
    
    # ★ 提出用CSVの保存先パスを変更
    OUTPUT_CSV_PATH = 'data/classification/submission_svm.csv'

    # 1. ★ モデルのインスタンス化 (クラス名を変更)
    svm_model = SVMModel()
    
    # 2. データの読み込みと前処理
    svm_model.load_data(X_TRAIN_PATH, Y_TRAIN_PATH, X_TEST_PATH)
    
    # 3. 訓練データを「訓練用」と「検証用」に分割
    svm_model.split_data(validation_size=0.2)
    
    # 4. モデルの訓練
    svm_model.train()
    
    # 5. モデルの評価
    svm_model.evaluate()
    
    # 6. 最終的な予測の実行
    svm_model.predict_for_submission(OUTPUT_CSV_PATH)