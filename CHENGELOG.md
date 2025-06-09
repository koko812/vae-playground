# 🔧 CHENGELOG-v1.1 VAE-Playground 2025-06-09

---

## 1. **目的・背景**

MNIST で **latent_dim = 2** の Conv-VAE を検証したところ、  
- 再構成が極端にぼやける  
- KL 項が強すぎて posterior collapse が起きやすい  
という課題が判明。  
本コミットでは **「モデル容量の底上げ」** と **「学習ループの改善」** によって  
*再構成精度* と *学習安定性* を同時に引き上げることを目的とした。

---

## 2. **主な変更点**

| 区分 | ファイル / モジュール | 内容 |
|------|----------------------|------|
| ➕ **追加** | `models/conv_vae.py` | *dec+1* オプション（ConvTranspose 追加）・LeakyReLU 化・βパラメータ受け入れ |
| ✍ **更新** | `train/run_convVAE_grid.py` | ① EPOCHS ループ実装<br>② loss を **サンプル平均** で記録<br>③ 学習率・β・バッチサイズを config で切替 |
| ✍ **更新** | `loss_fn_vae()` | `reduction="sum" → /batch_size` で正規化 |
| ➕ **追加** | `samples/conv_VAE/*` | 改良モデルの再構成画像（dec+1/β sweep/epoch sweep） |
| ➕ **追加** | `logs/<各種>.json` | 平均 loss / recon / kl、学習時間、GPU 使用量を追記 |
| 🛠 **修正** | 画像保存ヘルパ | tuple unpack (`recon, *_ = model(x)`) と `squeeze()` のバグ修正 |

---

## 3. **設計変更の内容と理由**

|   |   |
|---|---|
| :🔹: **変更対象** | `ConvVAE` / `run_convVAE_grid.py` |
| **Before** | - Decoder が `ConvT → ReLU` × *n* のみ<br>- loss = `BCE(sum) + β·KL(sum)`<br>- 1 Epoch 固定・学習率=1e-3 固定 |
| **After** | - Decoder 先頭に **ConvTranspose(ks=3,stride=1)** を 1段追加 (dec+1)<br>- 活性化を **LeakyReLU(0.2)** で勾配死防止<br>- `loss = BCE(sum)/B + β·KL(sum)/B` で **バッチサイズ非依存**<br>- `EPOCHS`, `lr`, `β`, `batch_size` を config で柔軟に |
| **理由** | - latent=2 でも線画情報を復元できるよう Decoder 表現力を増強<br>- 平均化しないと batch を変えるたびに loss が跳ね上がり比較不能<br>- 低 lr & 複数 epoch で KL と recon のバランスを段階的に最適化 |

---

## 4. **検討した別案や悩んだポイント**

| 別案 | 却下理由 |
|------|---------|
| β を epoch ごとに線形ウォームアップ | 実装負荷が大きい割に β=0.1 で十分改善を確認 |
| FullSkip‐KL（序盤数 epoch KL 無効化） | Posterior collapse は確認できず、導入効果が不透明 |
| PyTorch-Lightning への全面移行 | ローカル検証段階では純 PyTorch の方がデバッグしやすい |

---

## 5. **既存コードとの関係・依存箇所**

* `ConvAutoEncoder` 系スクリプトには一切影響なし  
* 旧 `ConvVAE` call でも **config の既定値で後方互換**  
* ログ／サンプル保存ディレクトリ構造は変更なし（ファイル名 prefix 追加で衝突回避）

---

## 6. **具体的な使い方・CLI 実行例**

```bash
# latent=2 / β=0.1 / extra decoder 1段 / 5 epoch / lr=1e-4
uv run train/run_convVAE_grid.py \
    --cfg dec+1_beta_0.1_latent2_large \
    --epochs 5 \
    --batch-size 256 \
    --lr 1e-4
````

生成物

* `samples/conv_VAE/dec+1_beta_0.1_latent2_large.png`
* `logs/dec_plus_one_beta_conv_ae_logs.json`

---

## 7. **コメントや議論の抜粋**

> **「kl=0.05 まで下がったら潜在が死んでるサイン」**
> ↳ β=0.1 と低 lr でちょうど良いバランスに落ち着いた。（Slack #vae-debug）

> **「batch でかくしたら loss が跳ねた！」**
> ↳ sum→mean 正規化を入れて解決。（PR review）

---

## 8. **既知のバグ・今の限界**

* latent=2 では *still blurry*（情報量不足は根本課題）
* KL warm-up 未実装（後続タスク）
* サンプル生成（prior サンプリング）UI は未整備

---

## 9. **今後の TODO リスト**

* [ ] β‐schedule (linear warm-up, cyclical) 対応
* [ ] latent\_dim sweep 自動化 & 結果プロット
* [ ] Lightning 版モジュール作成（実験管理簡略化）
* [ ] Posterior 可視化ツール（t-SNE / UMAP）

---

## 10. **感想・思ったこと**

> **「latent=2 でも意外と線画っぽさは残せる。
> Decoder の一段で世界が変わった。」**
> 小さな改良でも体感できる効果があるのは研究っぽくて楽しい。


</br>
</br>
</br>
</br>
</br>

# CHENGELOG-v1.0 VAE-Playground 2025-06-09

---

## 1. **目的・背景**

- これまで **Linear AE/Conv AE** だけで検証していたため、  
  **「AE と VAE を同一条件で比較するベンチマーク」** が存在しなかった。  
- VAE の導入効果（再構成 vs 生成能力・潜在空間の滑らかさ）を **定量／定性** の両面で評価する土台を作り、  
  後続の **Conv VAE・β-VAE・VQ-VAE** へスムーズに拡張できる構成を整えるのが目的。

---

## 2. **主な変更点**

| 区分 | ファイル / ディレクトリ | 内容 |
|------|------------------------|------|
| ➕ **追加** | `models/ae.py` | 全結合 AE クラスを分離実装 |
| ➕ **追加** | `models/vae.py` | 全結合 VAE クラス（μ・log σ² + 再パラメトリゼーション） |
| ➕ **追加** | `train/run_ae_vae_grid.py` | AE/VAE のグリッドサーチ兼比較スクリプト |
| ➕ **追加** | `samples/AE_*.png`, `samples/VAE_*.png` | 各設定での再構成結果を自動保存 |
| ✍ **更新** | `logs/ae_vae_logs.json` | loss・GPU 使用量・時間を JSON で記録 |

---

## 3. **設計変更の内容と理由**

|  |  |
|--|--|
| 🔹 **変更対象** | `AutoEncoder` / `VariationalAutoEncoder` クラス構成 |
| 🔹 **Before** | - AE / VAE の実装が混在しておらず、<br> -  comparison 用スクリプトが存在しなかった |
| 🔹 **After** | - `models/ae.py` と `models/vae.py` にクラスを分離。<br> -  `run_ae_vae_grid.py` で **mode フラグ**だけ切替えれば比較可能に。 |
| 🔹 **理由** | - モデルを差し替えても学習ループは共通化したい。<br> - 将来の **Conv VAE** 追加時に `build_*` 関数を差し替えるだけで済む拡張性を確保。 |

---

## 4. **検討した別案や悩んだポイント**

| 別案 | 採用しなかった理由 |
|------|-------------------|
| `build_model(mode, cfg)` ヘルパー関数で AE/VAE/Conv を動的生成 | VAE 特有の μ/logσ² → reparameterize の扱いが冗長になり、学習ループ側で分岐が増えるため却下 |
| `pytorch-lightning` を導入し Trainer に統一 | 学習ステップがブラックボックス化し、学習ロジックを触って理解したい本検証フェーズには不向き |

---

## 5. **既存コードとの関係・依存箇所**

| 影響範囲 | 互換性 |
|----------|--------|
| 旧 `ConvAE` 系コード | 影響なし（別スクリプト） |
| `samples/`, `logs/` 構造 | 共通ディレクトリを共有するのみ。ファイル名 prefix で衝突回避済み |

---

## 6. **具体的な使い方 / CLI 実行例**

```bash
# AE / VAE を一括比較（latent=2,8 / depth=1,2）
uv run train/run_ae_vae_grid.py
````

生成物：

* `samples/AE_latent2_depth1_ReLU.png`
* `samples/VAE_latent2_depth1_ReLU.png`
* `logs/ae_vae_logs.json`

---

## 7. **コメントや議論の抜粋**

> **「VAE は KL が強くて再構成ボケるね…」**
> → Conv 化と β-VAE で調整する方針に合意（@koko @assistant）

> **「build\_model() で動的生成案どう？」**
> → 「可読性より DSL 化しすぎるデメリットが今は大」 ――却下。（Slack #dev-ml）

---

## 8. **既知のバグ・今の限界**

* LinearVAE の再構成画像が著しくぼやける（latent\_dim=2 では特に顕著）
* β 値（KLスケール）は固定＝1。チューニング UI 未整備
* GPU 複数枚環境では peak\_memory 取得が 0 番 GPU 固定

---

## 9. **今後の TODO リスト**

* [ ] ConvVAE モジュール `models/conv_vae.py` を追加
* [ ] β-VAE、β の CLI 引数対応
* [ ] 潜在空間の 2D 可視化ツール（t-SNE / UMAP）
* [ ] `samples/` を HTML ギャラリー表示するビューワ
* [ ] マルチ GPU 対応のメモリロガー

---

## 10. **感想・思ったこと**

> 「AE と VAE を同じ土俵で比較すると、\*\*“再構成を犠牲にして滑らかな潜在空間を得る”\*\*という VAE の性質がよく見える。
> Conv 化・β 調整でどこまで両立できるかが次の山場！」

