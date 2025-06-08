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

