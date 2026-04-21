/**
 * Quick Draw 動物分類モデル学習スクリプト（Node.js + TF.js）
 * 学習後に webapp/ で読み込める形式で model/ に保存します
 */

const tf = require('./node_modules/@tensorflow/tfjs-node');
const fs = require('fs');
const path = require('path');
const https = require('https');

const CATEGORIES = ['giraffe', 'rabbit', 'elephant', 'lion', 'crocodile'];
const SAMPLES_PER_CLASS = 5000;
const DATA_DIR = path.join(__dirname, 'data');
const MODEL_DIR = path.join(__dirname, 'model');

// ── .npy ファイルパーサー ────────────────────────────────
function parseNpy(buffer) {
  // Magic: \x93NUMPY
  const magic = buffer.slice(0, 6).toString('latin1');
  if (magic !== '\x93NUMPY') throw new Error('Not a numpy file');

  const major = buffer[6];
  let headerLen, headerStart;
  if (major === 1) {
    headerLen = buffer.readUInt16LE(8);
    headerStart = 10;
  } else {
    headerLen = buffer.readUInt32LE(8);
    headerStart = 12;
  }

  const header = buffer.slice(headerStart, headerStart + headerLen).toString('ascii');

  // shape を抽出
  const shapeMatch = header.match(/'shape'\s*:\s*\(([^)]+)\)/);
  if (!shapeMatch) throw new Error('Cannot parse shape from header: ' + header);
  const shape = shapeMatch[1].split(',').map(s => parseInt(s.trim())).filter(n => !isNaN(n));

  const dataStart = headerStart + headerLen;
  const data = buffer.slice(dataStart);

  return { shape, data };
}

// ── データダウンロード ────────────────────────────────────
function download(url, dest) {
  return new Promise((resolve, reject) => {
    const file = fs.createWriteStream(dest);
    const request = (urlStr, redirectCount = 0) => {
      if (redirectCount > 5) return reject(new Error('Too many redirects'));
      const mod = urlStr.startsWith('https') ? https : require('http');
      mod.get(urlStr, res => {
        if (res.statusCode === 301 || res.statusCode === 302) {
          return request(res.headers.location, redirectCount + 1);
        }
        if (res.statusCode !== 200) {
          return reject(new Error(`HTTP ${res.statusCode} for ${urlStr}`));
        }
        let received = 0;
        res.on('data', chunk => { received += chunk.length; process.stdout.write(`\r  ${(received / 1024 / 1024).toFixed(1)} MB`); });
        res.pipe(file);
        file.on('finish', () => { file.close(); console.log(''); resolve(); });
      }).on('error', reject);
    };
    request(url);
  });
}

// ── メイン ───────────────────────────────────────────────
async function main() {
  if (!fs.existsSync(DATA_DIR)) fs.mkdirSync(DATA_DIR, { recursive: true });
  if (!fs.existsSync(MODEL_DIR)) fs.mkdirSync(MODEL_DIR, { recursive: true });

  // Step 1: データダウンロード
  console.log('=== Step 1: データをダウンロード中 ===');
  for (const name of CATEGORIES) {
    const dest = path.join(DATA_DIR, `${name}.npy`);
    if (fs.existsSync(dest)) {
      console.log(`  ${name}: スキップ（既存）`);
      continue;
    }
    const url = `https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/${name}.npy`;
    console.log(`  ${name} をダウンロード中...`);
    await download(url, dest);
    console.log(`  ${name}: 完了`);
  }

  // Step 2: データ読み込み・前処理
  console.log('\n=== Step 2: データ読み込み ===');
  const xArrays = [];
  const yArrays = [];

  for (let i = 0; i < CATEGORIES.length; i++) {
    const name = CATEGORIES[i];
    const buf = fs.readFileSync(path.join(DATA_DIR, `${name}.npy`));
    const { shape, data } = parseNpy(buf);
    const total = shape[0];
    const count = Math.min(SAMPLES_PER_CLASS, total);
    const pixelsPerSample = shape[1]; // 784

    // uint8 → Float32（0〜1正規化）
    const xs = new Float32Array(count * pixelsPerSample);
    for (let j = 0; j < count * pixelsPerSample; j++) {
      xs[j] = data[j] / 255.0;
    }
    xArrays.push(xs);

    const ys = new Float32Array(count);
    ys.fill(i);
    yArrays.push(ys);

    console.log(`  ${name}: ${count} 件`);
  }

  // 結合
  const totalSamples = SAMPLES_PER_CLASS * CATEGORIES.length;
  const xFlat = new Float32Array(totalSamples * 784);
  const yFlat = new Float32Array(totalSamples);
  let offset = 0;
  for (let i = 0; i < CATEGORIES.length; i++) {
    xFlat.set(xArrays[i], offset * 784);
    yFlat.set(yArrays[i], offset);
    offset += SAMPLES_PER_CLASS;
  }

  // シャッフル
  console.log('\nシャッフル中...');
  const idx = Array.from({ length: totalSamples }, (_, i) => i);
  for (let i = totalSamples - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [idx[i], idx[j]] = [idx[j], idx[i]];
  }
  const xShuffled = new Float32Array(totalSamples * 784);
  const yShuffled = new Float32Array(totalSamples);
  for (let i = 0; i < totalSamples; i++) {
    xShuffled.set(xFlat.slice(idx[i] * 784, (idx[i] + 1) * 784), i * 784);
    yShuffled[i] = yFlat[idx[i]];
  }

  // テスト分割（80:20）
  const splitAt = Math.floor(totalSamples * 0.8);
  const xTrain = tf.tensor4d(xShuffled.slice(0, splitAt * 784), [splitAt, 28, 28, 1]);
  const yTrain = tf.tensor1d(yShuffled.slice(0, splitAt));
  const xTest  = tf.tensor4d(xShuffled.slice(splitAt * 784), [totalSamples - splitAt, 28, 28, 1]);
  const yTest  = tf.tensor1d(yShuffled.slice(splitAt));
  console.log(`訓練: ${splitAt} 件 / テスト: ${totalSamples - splitAt} 件`);

  // Step 3: モデル定義（CNN）
  console.log('\n=== Step 3: モデル構築 ===');
  const model = tf.sequential();
  model.add(tf.layers.conv2d({ inputShape: [28, 28, 1], filters: 32, kernelSize: 3, activation: 'relu' }));
  model.add(tf.layers.maxPooling2d({ poolSize: 2 }));
  model.add(tf.layers.conv2d({ filters: 64, kernelSize: 3, activation: 'relu' }));
  model.add(tf.layers.maxPooling2d({ poolSize: 2 }));
  model.add(tf.layers.conv2d({ filters: 64, kernelSize: 3, activation: 'relu' }));
  model.add(tf.layers.flatten());
  model.add(tf.layers.dense({ units: 128, activation: 'relu' }));
  model.add(tf.layers.dropout({ rate: 0.3 }));
  model.add(tf.layers.dense({ units: CATEGORIES.length, activation: 'softmax' }));

  model.compile({
    optimizer: 'adam',
    loss: 'sparseCategoricalCrossentropy',
    metrics: ['accuracy'],
  });
  model.summary();

  // Step 4: 学習
  console.log('\n=== Step 4: 学習開始 ===');
  await model.fit(xTrain, yTrain, {
    epochs: 10,
    batchSize: 64,
    validationSplit: 0.1,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        console.log(`  Epoch ${epoch + 1}/10 — loss: ${logs.loss.toFixed(4)}, acc: ${logs.acc.toFixed(4)}, val_acc: ${logs.val_acc.toFixed(4)}`);
      },
    },
  });

  // Step 5: 評価・保存
  console.log('\n=== Step 5: 評価・保存 ===');
  const evalResult = model.evaluate(xTest, yTest, { batchSize: 64 });
  const testAcc = (await evalResult[1].data())[0];
  console.log(`テスト精度: ${(testAcc * 100).toFixed(2)}%`);

  await model.save(`file://${MODEL_DIR}`);
  console.log(`\nモデルを保存しました: ${MODEL_DIR}`);
  console.log('次のステップ: webapp/index.html をブラウザで開く');
}

main().catch(err => { console.error(err); process.exit(1); });
