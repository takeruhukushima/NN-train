# PyTorch で MLP を構築・評価するクラス
#めちゃくちゃ速かった。
# --- モデル評価 (検証データ) ---
#   検証データの正解率 (Accuracy): 88.48 %

# --- 分類レポート (検証データ) ---
#               precision    recall  f1-score   support

#  T-shirt/top       0.84      0.85      0.84      1182
#      Trouser       0.97      0.99      0.98      1222
#     Pullover       0.88      0.72      0.79      1210
#        Dress       0.91      0.88      0.89      1153
#         Coat       0.71      0.89      0.79      1180
#       Sandal       0.98      0.95      0.96      1215
#        Shirt       0.75      0.69      0.72      1239
#      Sneaker       0.94      0.95      0.94      1228
#          Bag       0.96      0.97      0.97      1185
#   Ankle boot       0.95      0.96      0.96      1186

#     accuracy                           0.88     12000
#    macro avg       0.89      0.89      0.88     12000
# weighted avg       0.89      0.88      0.88     12000
#似た画像における精度が高い。
import numpy as np
import pandas as pd
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# ★ PyTorch のライブラリをインポート
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm # (オプション: 進捗表示用)

# --- 1. PyTorch のモデル (nn.Module) を定義 ---
class PyTorchMLP(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=128, output_dim=10):
        super(PyTorchMLP, self).__init__()
        # Keras と同じ構造を定義
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        # ★ Softmax は不要 (CrossEntropyLoss が内部で計算するため)

    def forward(self, x):
        # データの流れを定義
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

# --- 2. メインのクラス (sklearn版の構造を踏襲) ---
class PyTorchMLPModel:
    """
    PyTorch で MLP を構築・評価するクラス。
    """
    
    def __init__(self, hidden_dim=128, epochs=10, batch_size=32, learning_rate=0.001, random_state=42):
        np.random.seed(random_state)
        torch.manual_seed(random_state)
        
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        
        # デバイス (GPUが使えるならGPU, なければCPU) を設定
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # ★ モデルをインスタンス化し、デバイスに送る
        self.model = PyTorchMLP(hidden_dim=hidden_dim).to(self.device)
        
        # ★ 損失関数とオプティマイザを定義
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        print("PyTorch モデルの定義完了:")
        print(self.model)

        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_train_processed = None
        self.X_test_processed = None
        self.y_train_orig = None

    # --- load_data, _preprocess, split_data は Keras版/sklearn版と全く同じ ---

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

    # --- train メソッドを PyTorch (手書きループ) 用に書き換え ---

    def train(self):
        """
        PyTorch の学習ループを手書きで実行する
        """
        if self.X_train is None or self.y_train is None:
            print("エラー: 訓練データがありません", file=sys.stderr)
            return

        print(f"PyTorch モデルの訓練を開始します (Epochs: {self.epochs}, Batch Size: {self.batch_size})...")

        # 1. NumPy配列をPyTorchテンソルに変換
        X_train_tensor = torch.tensor(self.X_train).float()
        y_train_tensor = torch.tensor(self.y_train).long() # 損失関数がlong型(整数)を要求
        
        # 2. DataLoader (バッチ処理機) を作成
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        # 3. ★ 訓練ループ (Epoch)
        for epoch in range(self.epochs):
            self.model.train() # モデルを訓練モードに
            running_loss = 0.0
            
            # 4. ★ バッチループ (tqdmで進捗を表示)
            for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.epochs}", leave=False):
                # データをGPU/CPUに送る
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # 4a. 勾配をリセット
                self.optimizer.zero_grad()
                
                # 4b. 予測 (フォワード)
                outputs = self.model(inputs)
                
                # 4c. 損失の計算
                loss = self.criterion(outputs, labels)
                
                # 4d. 勾配の計算 (バックプロパゲーション)
                loss.backward()
                
                # 4e. パラメータの更新
                self.optimizer.step()
                
                running_loss += loss.item()
            
            # エポック終了後に平均損失を表示
            avg_loss = running_loss / len(train_loader)
            print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {avg_loss:.4f}")
            
            # (オプション: エポック毎に検証精度をチェック)
            self.evaluate(epoch=epoch+1)

        print("モデルの訓練が完了しました。")

    # --- evaluate メソッドを PyTorch 用に書き換え ---

    def evaluate(self, epoch=None): # epoch引数は訓練中の呼び出し用
        """
        PyTorch モデルを評価する (sklearnのmetricsを使用)
        """
        if self.model is None or self.X_val is None or self.y_val is None:
            print("エラー: モデルまたは検証データがありません", file=sys.stderr)
            return

        if epoch is None:
             print("\n--- モデル評価 (検証データ) ---")
        else:
             print(f"  --- Epoch {epoch} 検証 ---")

        self.model.eval() # モデルを評価モードに (Dropoutなどを無効化)
        
        X_val_tensor = torch.tensor(self.X_val).float().to(self.device)
        y_val_tensor = torch.tensor(self.y_val).long() # y_val は NumPy のまま
        
        all_preds = []
        
        # ★ 勾配計算をオフにしてメモリ節約
        with torch.no_grad():
            # DataLoaderを使っても良いが、検証データが小さいなら一括処理でもOK
            outputs = self.model(X_val_tensor)
            # Softmaxを適用し、最も確率の高いクラス (インデックス) を取得
            _, predicted_labels = torch.max(outputs.data, 1)
            # GPU/CPUからNumPy配列に変換
            all_preds = predicted_labels.cpu().numpy()

        # NumPy配列になったので、sklearnの関数が使える
        accuracy = accuracy_score(self.y_val, all_preds)
        print(f"  検証データの正解率 (Accuracy): {accuracy * 100:.2f} %")
        
        # (訓練の最終回のみ詳細レポートを表示)
        if epoch is None:
            print("\n--- 分類レポート (検証データ) ---")
            class_names = [
                "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
            ]
            try:
                report = classification_report(self.y_val, all_preds, target_names=class_names, zero_division=0)
                print(report)
            except Exception as e:
                print(f"分類レポートの生成に失敗: {e}")
        return accuracy

    # --- predict_for_submission メソッドを PyTorch 用に書き換え ---

    def predict_for_submission(self, output_csv_path):
        """
        x_test.npy に対する最終的な予測（ラベル）をCSVに保存する
        """
        if self.model is None or self.X_test_processed is None:
            print("エラー: モデルまたはテストデータがありません", file=sys.stderr)
            return

        print(f"\n最終的なテストデータ (x_test.npy) に対する予測を実行します...")
        
        self.model.eval()
        X_test_tensor = torch.tensor(self.X_test_processed).float().to(self.device)
        
        all_preds = []
        
        # DataLoader (バッチ処理) を使って予測 (メモリ節約のため)
        test_dataset = TensorDataset(X_test_tensor)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size * 2) # 予測時はバッチサイズを大きくできる

        with torch.no_grad():
            for [inputs] in test_loader: # ラベルyは無い
                outputs = self.model(inputs)
                _, predicted_labels = torch.max(outputs.data, 1)
                all_preds.extend(predicted_labels.cpu().numpy())

        y_pred_test = np.array(all_preds)
        
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
    OUTPUT_CSV_PATH = 'data/classification/submission_pytorch_mlp.csv'

    # 1. モデルのインスタンス化 (10エポック実行)
    mlp_model = PyTorchMLPModel(epochs=10, batch_size=32)
    
    mlp_model.load_data(X_TRAIN_PATH, Y_TRAIN_PATH, X_TEST_PATH)
    mlp_model.split_data(validation_size=0.2)
    mlp_model.train()
    # (train()の内部で evaluate() がエポック毎に呼ばれる)
    # (最終評価は train() が終わった後に自動では呼ばれないので、ここで呼ぶ)
    print("\n--- 訓練完了後の最終評価 ---")
    mlp_model.evaluate() 
    
    mlp_model.predict_for_submission(OUTPUT_CSV_PATH)