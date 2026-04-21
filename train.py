import numpy as np
import requests
import os
import tensorflow as tf
from tensorflow import keras

# カテゴリ定義（英語名: 日本語名）
CATEGORIES = {
    'giraffe': 'キリン',
    'rabbit': 'うさぎ',
    'elephant': 'ぞう',
    'lion': 'ライオン',
    'crocodile': 'ワニ',
}

DATA_DIR = 'data'
os.makedirs(DATA_DIR, exist_ok=True)

# --- Step 1: データダウンロード ---
print('=== データをダウンロード中 ===')
for en_name in CATEGORIES.keys():
    path = f'{DATA_DIR}/{en_name}.npy'
    if os.path.exists(path):
        print(f'  {en_name}: スキップ（既存）')
        continue
    url = f'https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/{en_name}.npy'
    print(f'  {en_name} をダウンロード中...')
    r = requests.get(url, stream=True)
    with open(path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f'  {en_name}: 完了')

# --- Step 2: データ読み込み・前処理 ---
print('\n=== データを読み込み中 ===')
SAMPLES_PER_CLASS = 5000  # 1カテゴリあたりのサンプル数

X, y = [], []
for i, en_name in enumerate(CATEGORIES.keys()):
    data = np.load(f'{DATA_DIR}/{en_name}.npy')
    data = data[:SAMPLES_PER_CLASS]
    X.append(data)
    y.append(np.full(len(data), i))
    print(f'  {en_name}: {len(data)}件読み込み')

X = np.concatenate(X).reshape(-1, 28, 28, 1) / 255.0
y = np.concatenate(y)

# シャッフル
idx = np.random.permutation(len(X))
X, y = X[idx], y[idx]

# 訓練・テスト分割
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]
print(f'\n訓練データ: {len(X_train)}件 / テストデータ: {len(X_test)}件')

# --- Step 3: モデル作成 ---
print('\n=== モデルを作成中 ===')
model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(len(CATEGORIES), activation='softmax'),
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

# --- Step 4: 学習 ---
print('\n=== 学習開始 ===')
model.fit(X_train, y_train,
          epochs=10,
          batch_size=64,
          validation_split=0.1)

# --- Step 5: 評価・保存 ---
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'\nテスト精度: {test_acc:.4f}')

model.save('model.keras')
print('モデルを保存しました: model.keras')
