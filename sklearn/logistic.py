# sklearnを用いてロジスティック（ソフトマックス）回帰モデルを構築・評価するコード
#Accuracyは大体0.85くらいになる。
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
# データを訓練用と検証用に分割するライブラリをインポート
from sklearn.model_selection import train_test_split
import sys

class LogisticRegressionModel:
    """
    ロジスティック回帰モデルを訓練・評価するクラス。
    y_test がないことを想定し、訓練データを分割して検証する。
    """
    
    def __init__(self, solver='saga', max_iter=1000, random_state=42):
        self.model = LogisticRegression(
            solver=solver, 
            max_iter=max_iter, 
            random_state=random_state,
            multi_class='auto',
            verbose=1
        )
        self.X_train = None
        self.y_train = None
        self.X_val = None # 検証用データ (X)
        self.y_val = None # 検証用データ (y)
        self.X_test = None # 最終的な予測用データ

    def load_data(self, x_train_path, y_train_path, x_test_path):
        """
        Numpy (.npy) ファイルからデータを読み込む (y_testなし)
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
            self.X_test_processed = self._preprocess(X_test_orig) # 最終予測用も前処理
            print("データの前処理 (Flatten, Normalization) 完了")

        except FileNotFoundError as e:
            print(f"エラー: ファイルが見つかりません - {e}", file=sys.stderr)
            sys.exit(1)

    def _preprocess(self, data):
        """
        データの前処理 (Flatten と Normalization) を行う内部メソッド
        """
        # 1. データの1次元化 (Flatten)
        if data.ndim == 3:
            data = data.reshape(data.shape[0], -1) # (N, 28, 28) -> (N, 784)
            
        # 2. データの正規化 (Normalization)
        data = data.astype('float32') / 255.0
        return data

    def split_data(self, validation_size=0.2):
        """
        読み込んだ訓練データを、さらに訓練用と検証用に分割する
        """
        if self.X_train_processed is None or self.y_train_orig is None:
            print("エラー: 訓練データが読み込まれていません", file=sys.stderr)
            return

        # 訓練データを「新訓練データ」と「検証データ」に分割
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            self.X_train_processed,
            self.y_train_orig,
            test_size=validation_size, # (例: 20% を検証用にする)
            random_state=42 # 再現性のための乱数シード
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

        print("モデルの訓練を開始します...")
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
        
        # 1. 正解率 (Accuracy)
        accuracy = accuracy_score(self.y_val, y_pred_val)
        print(f"検証データの正解率 (Accuracy): {accuracy * 100:.2f} %")
        
        # 2. 分類レポート (詳細な評価)
        class_names = [
            "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
            "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
        ]
        try:
            report = classification_report(self.y_val, y_pred_val, target_names=class_names)
            print("\n分類レポート (検証データ):")
            print(report)
        except:
            report = classification_report(self.y_val, y_pred_val)
            print("\n分類レポート (検証データ):")
            print(report)

        return accuracy, report

    def predict_for_submission(self, output_csv_path):
        """
        x_test.npy に対する最終的な予測（ラベル）を
        指定されたパスにCSVファイルとして保存する。
        """
        if self.model is None or self.X_test_processed is None:
            print("エラー: モデルまたはテストデータがありません", file=sys.stderr)
            return
            
        print(f"\n最終的なテストデータ (x_test.npy) に対する予測を実行します...")
        y_pred_test = self.model.predict(self.X_test_processed)
        
        # --- ここからが変更点 ---
        
        # 1. 提出用のDataFrameを作成
        # KaggleなどではIDが1から始まることが多いため、np.arange(1, ...) としています。
        # もしIDが0から始まるなら np.arange(len(y_pred_test)) でOKです。
        submission_df = pd.DataFrame({
            "ID": np.arange(1, len(y_pred_test) + 1), # 1から始まるID列
            "Label": y_pred_test                     # 予測したラベル列
        })

        # 2. CSVファイルとして保存
        try:
            submission_df.to_csv(output_csv_path, index=False) # index=False で行番号を保存しない
            print(f"予測結果を {output_csv_path} に保存しました。")
            print("\n--- 保存されたCSV (先頭5件) ---")
            print(submission_df.head())
            
        except Exception as e:
            print(f"エラー: CSVファイルの保存に失敗しました - {e}", file=sys.stderr)

        # 予測結果の配列自体も念のため返す
        return y_pred_test

if __name__ == "__main__":
    # データパス
    X_TRAIN_PATH = 'data/classification/x_train.npy'
    Y_TRAIN_PATH = 'data/classification/y_train.npy'
    X_TEST_PATH = 'data/classification/x_test.npy'
    
    # ★ 提出用CSVの保存先パスを定義
    OUTPUT_CSV_PATH = 'data/classification/submission_logistic_regression.csv'

    # 1. モデルのインスタンス化
    logistic_model = LogisticRegressionModel()
    
    # 2. データの読み込みと前処理
    logistic_model.load_data(X_TRAIN_PATH, Y_TRAIN_PATH, X_TEST_PATH)
    
    # 3. 訓練データを「訓練用」と「検証用」に分割
    logistic_model.split_data(validation_size=0.2)
    
    # 4. モデルの訓練
    logistic_model.train()
    
    # 5. モデルの評価
    logistic_model.evaluate()
    
    # 6. 最終的な予測の実行 (★ 保存先パスを渡す)
    logistic_model.predict_for_submission(OUTPUT_CSV_PATH)