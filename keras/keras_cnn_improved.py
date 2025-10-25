# keras/keras_cnn.pyを改善したコード
# ベースモデルとして Batch Normalization (BN) や Dropout を組み込んだ高性能な CNN アーキテクチャ
#手動で設定していた以下の職人技的なパラメータを、Optunaが効率的かつ自動的に見つけ出す
#長時間かつ無駄な訓練を防ぐためのテクニックEarly Stopping & Pruningの実装
#最終出力のaccuracy:
# 938/938 ━━━━━━━━━━━━━━━━━━━━ 18s 19ms/step - accuracy: 0.9765 - loss: 0.0683

# --- 5. 最終予測 (x_test.npy) をCSVに出力 ---

# --- 最終モデルの検証データでの分類レポート ---
#               precision    recall  f1-score   support

#  T-shirt/top       0.88      0.89      0.88      1182
#      Trouser       0.99      0.99      0.99      1222
#     Pullover       0.88      0.88      0.88      1210
#        Dress       0.90      0.93      0.92      1153
#         Coat       0.89      0.86      0.87      1180
#       Sandal       0.98      0.98      0.98      1215
#        Shirt       0.79      0.79      0.79      1239
#      Sneaker       0.97      0.95      0.96      1228
#          Bag       0.98      0.98      0.98      1185
#   Ankle boot       0.95      0.98      0.96      1186

#     accuracy                           0.92     12000
#    macro avg       0.92      0.92      0.92     12000
# weighted avg       0.92      0.92      0.92     12000

# 完了: 予測結果を data/classification/submission_optuna_best_cnn.csv に保存しました。
# --- 最終モデルの検証データでの精度 ---
# 最終モデルの検証精度: 92.17 %
import numpy as np
import pandas as pd
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Keras (TensorFlow) のライブラリ
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ★ Optuna と Optuna-Integration をインポート
import optuna
# ★ 修正点: TFKerasPruningCallback は optuna_integration からインポートする
from optuna_integration.tfkeras import TFKerasPruningCallback 

# tf のログレベル調整 (Optuna実行中は特に重要)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # WARNING と ERROR のみ表示
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(0)

# --- データの前処理関数 (クラス外に定義) ---
def preprocess_data(data):
    """データを CNN が扱える形式 (Normalization + チャネル次元) に前処理"""
    data = data.astype('float32') / 255.0
    data = np.expand_dims(data, axis=-1) # (N, H, W) -> (N, H, W, 1)
    return data

# --- ★ Optuna の目的関数 (objective) ---
def objective(trial, X_train, y_train, X_val, y_val):
    """Optuna が呼び出す関数。ハイパーパラメータを試し、精度を返す"""
    
    # --- 1. 試行するハイパーパラメータを定義 ---
    # Conv2D のフィルター数 (カテゴリカル/離散値で定義)
    filters1 = trial.suggest_categorical("filters1", [16, 32, 64])
    filters2 = trial.suggest_categorical("filters2", [32, 64, 128])
    # Dense 層のユニット数 (整数値で定義)
    dense_units = trial.suggest_int("dense_units", 64, 256, step=32)
    # Dropout率 (連続値で定義)
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
    # 学習率 (対数スケールで定義)
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    
    # --- 2. モデル構築 ---
    model = keras.Sequential([
        layers.Input(shape=(28, 28, 1)),
        # CNN ブロック 1
        layers.Conv2D(filters1, (3, 3), padding='same'), # ★ Optuna
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        # CNN ブロック 2
        layers.Conv2D(filters2, (3, 3), padding='same'), # ★ Optuna
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        # 分類ヘッド
        layers.Flatten(),
        layers.Dense(dense_units),                       # ★ Optuna
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(dropout_rate),                    # ★ Optuna
        layers.Dense(10, activation='softmax')
    ])

    # --- 3. コンパイル ---
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate) # ★ Optuna
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # --- 4. 訓練 (早期終了と枝刈りを使用) ---
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    
    # ★ 修正: インポートした PruningCallback を直接使用
    pruning_callback = TFKerasPruningCallback(trial, 'val_accuracy')

    # Optunaの試行時間が長すぎるので、epochsを30に固定
    # batch_size=64 に固定 (GPUメモリとの兼ね合いのため)
    history = model.fit(
        X_train, y_train,
        epochs=30, 
        batch_size=64, 
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, pruning_callback], 
        verbose=0 # ログを抑制
    )

    # --- 5. 評価 ---
    # EarlyStopping で最良の重みが復元されているはず
    loss, accuracy = model.evaluate(X_val, y_val, verbose=0)

    # --- 6. Optuna に返す値 (Validation Accuracy を最大化) ---
    return accuracy

# --- メインの実行部分 ---
if __name__ == "__main__":
    X_TRAIN_PATH = 'data/classification/x_train.npy'
    Y_TRAIN_PATH = 'data/classification/y_train.npy'

    # --- 1. データ準備 (Optunaの前に一度だけ行う) ---
    try:
        X_train_orig = np.load(X_TRAIN_PATH)
        y_train_orig = np.load(Y_TRAIN_PATH)
        print("データの読み込み完了")
    except FileNotFoundError as e:
        print(f"エラー: ファイルが見つかりません - {e}", file=sys.stderr)
        sys.exit(1)

    # 訓練データから訓練セットと検証セットに分割
    X_train_split, X_val_split, y_train, y_val = train_test_split(
        X_train_orig, y_train_orig, test_size=0.2, random_state=42
    )
    # 前処理
    X_train = preprocess_data(X_train_split)
    X_val = preprocess_data(X_val_split)
    print("データ準備完了")

    # --- 2. Optuna Study の作成と最適化の実行 ---
    study = optuna.create_study(
        direction='maximize', # Accuracy を最大化
        pruner=optuna.pruners.MedianPruner() # 早期枝刈りアルゴリズム
    )

    n_trials = 30 
    print(f"Optunaによるハイパーパラメータ探索を開始します (n_trials={n_trials})...")
    
    # objective 関数にデータセットを渡す
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_val, y_val),
                   n_trials=n_trials,
                   timeout=600) # 10分 (600秒) で強制終了

    # --- 3. 結果の表示 ---
    print("\n--- Optuna 探索結果 ---")
    print(f"試行回数: {len(study.trials)}")

    best_trial = study.best_trial
    print(f"最高精度 (Validation Accuracy): {best_trial.value * 100:.2f} %")

    print("最適なハイパーパラメータ:")
    for key, value in best_trial.params.items():
        print(f"  {key}: {value}")
# ... (Optuna 探索結果の表示まで完了) ...

    
    # --- 4. 最適パラメータで最終モデルを訓練 (X_train のみを使用) ---

    X_TEST_PATH = 'data/classification/x_test.npy'
    OUTPUT_CSV_PATH = 'data/classification/submission_optuna_best_cnn.csv'

    try:
        X_test_orig = np.load(X_TEST_PATH)
    except FileNotFoundError as e:
        print(f"エラー: テストファイルが見つかりません - {e}", file=sys.stderr)
        sys.exit(1)
        
    X_test_processed = preprocess_data(X_test_orig)
    
    print("\n--- 4. 最適モデルの再構築と最終訓練 ---")
    best_params = study.best_params
    
    # モデル構築（Optunaのコードを流用）
    final_model = keras.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(best_params["filters1"], (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(best_params["filters2"], (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(best_params["dense_units"]),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(best_params["dropout_rate"]),
        layers.Dense(10, activation='softmax')
    ])

    # コンパイル
    final_optimizer = tf.keras.optimizers.Adam(learning_rate=best_params["learning_rate"])
    final_model.compile(optimizer=final_optimizer,
                        loss='sparse_categorical_crossentropy',
                        metrics=['accuracy'])
    
    # ★ 最終訓練 (X_train のみで訓練し、X_val で Early Stopping を監視)
    #    これにより、X_val での評価が真の性能を反映します。
    print(f"訓練データ ({X_train.shape[0]}件) で最終訓練を開始します...")
    final_model.fit(
        X_train, y_train,
        epochs=30, # Optunaと同じ最大エポック数
        batch_size=64, 
        # ★ ここを修正: X_val を EarlyStopping の監視用に渡し、monitor='val_loss' に戻す ★
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)], 
        validation_data=(X_val, y_val), # ★ EarlyStopping のためにX_valを渡す
        verbose=1
    )
    
    # 5. 予測とCSV出力
    print(f"\n--- 5. 最終予測 (x_test.npy) をCSVに出力 ---")
    # CSV用の予測 (X_test_processed)
    y_pred_proba = final_model.predict(X_test_processed, verbose=0)
    y_pred_test = np.argmax(y_pred_proba, axis=1)

    # 検証データ (X_val) に対する予測を実行 (最終評価用)
    y_val_pred_proba = final_model.predict(X_val, verbose=0)
    y_val_pred = np.argmax(y_val_pred_proba, axis=1)

    # -----------------------------------------------------
    # 評価結果の出力 (X_val は訓練に使われていないため、真の性能に近い)
    # -----------------------------------------------------
    
    # 分類レポートを出力
    class_names = [
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
    ]
    
    print("\n--- 最終モデルの検証データでの分類レポート ---")
    print(classification_report(y_val, y_val_pred, target_names=class_names, zero_division=0))
    
    submission_df = pd.DataFrame({
        "ID": np.arange(1, len(y_pred_test) + 1),
        "Label": y_pred_test
    })

    try:
        submission_df.to_csv(OUTPUT_CSV_PATH, index=False)
        print(f"完了: 予測結果を {OUTPUT_CSV_PATH} に保存しました。")
        # 最終的な精度を evaluate で再度確認
        final_loss, final_accuracy = final_model.evaluate(X_val, y_val, verbose=0)
        print("--- 最終モデルの検証データでの精度 ---")
        print(f"最終モデルの検証精度: {final_accuracy * 100:.2f} %")
    except Exception as e:
        print(f"エラー: CSVファイルの保存に失敗しました - {e}", file=sys.stderr)