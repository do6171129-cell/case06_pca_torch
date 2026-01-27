# 事例6：最適化として解く主成分分析（PCA）

本リポジトリは、主成分分析（PCA）を固有値分解ではなく **最適化問題として定式化し、PyTorch により実装・検証する事例**である。  
事例5（NumPy・固有値分解PCA）と同一の定義量に到達することを確認することで、両手法の同値性を示す。

## 目的

- PCA を最適化問題として直接解く
- autograd による勾配最適化と線形代数的定義の一致を確認する
- 固有値分解を用いずに、主成分・主成分得点・寄与率を得る

## 内容

- 中心化 → 射影 → 最適化（制約付き）
- 単位ノルム制約を no_grad による正規化で処理
- 出力仕様を事例5と互換に設計
- 数値一致の検証のみを実施（可視化・解釈は行わない）

## ディレクトリ構成

```text
case06_pca_torch/
├─ case06_pca_torch.ipynb   # 本文（数式・説明・検証）
├─ src/
│  ├─ pca.py                # 最適化PCA実装
│  └─ make_synthetic.py     # 合成データ生成
├─ data/
│  ├─ meta.json
│  └─ X.csv                 # 合成データ（固定）
├─ requirements.txt
├─ .gitignore
└─ README.md
```

## 出力仕様
```text
mean            : (1, d)
components      : (d, 1)
scores          : (n, 1)
eigvals         : (1,)
explained_ratio : (1,)
```
- eigvals は主成分方向の分散として定義
- explained_ratio は全分散に対する比
