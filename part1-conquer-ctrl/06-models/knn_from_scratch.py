"""从零手写 KNN，并在真实数据上验证与 sklearn 一致（含"距离并列"的处理）。

为什么要手写：KNN 是所有模型里最容易从零写一遍的——它没有训练过程，
核心只有三步：算距离、找最近 k 个、投票取平均。

这个脚本还揭示了一个真实存在的细节：
  **我们的数据里有 665 行特征是重复的（其中 522 行是全零向量，即两队都没有历史记录）。**
  这些点在距离上完全并列，第 k 个与第 k+1 个邻居距离相等时，
  "选哪 k 个"取决于实现的取舍规则，于是手写版和 sklearn 会给出不同的概率。
  这不是 bug，是 KNN 在这种数据上的真实脆弱性。

用法（项目根目录）：
    python -B -m scripts.knn_from_scratch
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from scripts.train_m1_m2 import design, partition

K = 135


# ======================================================================
# 第一步：算距离
# ======================================================================
def knn_distance(X_train: np.ndarray, X_query: np.ndarray) -> np.ndarray:
    """欧氏距离矩阵，形状 (n_query, n_train)。

    数学：||a - b||² = ||a||² + ||b||² - 2·a·b
    把"逐对相减再平方"改写成一次矩阵乘法，快几十倍。
    """
    q2 = (X_query ** 2).sum(axis=1)[:, None]      # (n_query, 1)
    t2 = (X_train ** 2).sum(axis=1)[None, :]      # (1, n_train)
    cross = X_query @ X_train.T                   # (n_query, n_train)
    d2 = q2 + t2 - 2.0 * cross
    return np.sqrt(np.maximum(d2, 0.0))           # 夹住浮点误差产生的负数


# ======================================================================
# 第二、三步：找最近 k 个 + 投票
# ======================================================================
def knn_predict_proba(X_train, y_train, X_query, k=K, chunk=400):
    """返回 (概率数组, 并列标记数组)。

    并列标记为 True 表示：第 k 个邻居和第 k+1 个邻居距离相等，
    此时"哪 k 个入选"取决于实现的取舍规则。
    """
    probs = np.empty(len(X_query), dtype=float)
    tie = np.zeros(len(X_query), dtype=bool)
    for start in range(0, len(X_query), chunk):
        sl = slice(start, start + chunk)
        dist = knn_distance(X_train, X_query[sl])
        # 第 k 个与第 k+1 个邻居的距离（k-1 和 k 是 0-based 下标）
        two = np.partition(dist, [k - 1, k], axis=1)
        # 注意：np.partition 返回整个数组，所以要取第 k-1、k 列，不是第 0、1 列
        tie[sl] = np.isclose(two[:, k - 1], two[:, k], rtol=0, atol=1e-12)
        # argpartition 只保证前 k 个最小，不排序 —— 比 argsort 快
        idx = np.argpartition(dist, k - 1, axis=1)[:, :k]
        # 第三步：k 个邻居的标签取平均 = 概率
        probs[sl] = y_train[idx].mean(axis=1)
    return probs, tie


def main():
    train, val = partition("train"), partition("val")
    Xtr, Xva = design(train).to_numpy(float), design(val).to_numpy(float)
    ytr, yva = train.team1_won.to_numpy(), val.team1_won.to_numpy()

    both = np.vstack([Xtr, -Xtr])          # 镜像扩增
    yb = np.r_[ytr, 1 - ytr]
    sc = StandardScaler().fit(both)        # KNN 必须标准化
    Ztr, Zva = sc.transform(both), sc.transform(Xva)

    n_uniq = len(np.unique(both, axis=0))
    n_zero = int((np.abs(both).sum(axis=1) == 0).sum())
    print(f"训练样本（镜像后）{len(Ztr):,} × {Ztr.shape[1]} 维")
    print(f"其中唯一特征向量 {n_uniq:,} 个，重复 {len(Ztr) - n_uniq:,} 行")
    print(f"全零向量（两队都无历史）{n_zero} 行 —— 这些点在距离上完全并列\n")

    p_hand, tie = knn_predict_proba(Ztr, yb, Zva, k=K)
    m = KNeighborsClassifier(n_neighbors=K, weights="uniform", algorithm="brute", n_jobs=1)
    m.fit(Ztr, yb)
    p_sk = m.predict_proba(Zva)[:, 1]

    ok = ~tie
    diff_ok = float(np.max(np.abs(p_hand[ok] - p_sk[ok]))) if ok.any() else 0.0
    diff_all = float(np.max(np.abs(p_hand - p_sk)))

    print("=" * 66)
    print("手写版 vs sklearn（brute 算法）")
    print("=" * 66)
    print(f"  无并列的样本   {ok.sum():,} / {len(Zva):,}（{ok.mean():.1%}）")
    print(f"    最大概率差   {diff_ok:.3e}   ← 这些样本上两边应完全一致")
    print(f"  存在并列的样本 {(~ok).sum():,} 场")
    print(f"    最大概率差   {diff_all:.3e}（发生在并列样本上，见下）")
    print(f"\n  整体 log loss：手写 {log_loss(yva, p_hand, labels=[0,1]):.6f}"
          f" / sklearn {log_loss(yva, p_sk, labels=[0,1]):.6f}")
    print(f"  整体准确率    ：手写 {((p_hand>.5)==yva).mean():.2%}"
          f" / sklearn {((p_sk>.5)==yva).mean():.2%}")

    if ok.any() and diff_ok > 1e-9:
        raise SystemExit("无并列样本上两边不一致，说明手写版有 bug")
    print("\n  ✅ 在距离无并列的样本上，手写版与 sklearn 完全一致\n")

    # ---- 为什么并列会改变结果：看全零样本 ----
    zero_q = np.abs(Xva).sum(axis=1) == 0
    if zero_q.any():
        print("=" * 66)
        print("并列的极端例子：特征全零的验证样本（两队都无历史）")
        print("=" * 66)
        print(f"  验证集里有 {zero_q.sum()} 场这样的比赛。")
        print(f"  它们到全部 {n_zero} 个全零训练点的距离都是 0，")
        print(f"  所以「选哪 {K} 个」完全由实现的取舍规则决定：")
        print(f"    手写版平均概率   {p_hand[zero_q].mean():.4f}")
        print(f"    sklearn 平均概率  {p_sk[zero_q].mean():.4f}")
        print(f"    这些样本的真实胜率 {yva[zero_q].mean():.4f}")
        print("  → KNN 在冷启动样本上的预测，本质是在猜这批数据的基准胜率。\n")

    # ---- 具体例子：一场比赛的邻居 ----
    s = int(np.argmax(~tie))   # 挑一个无并列的样本来看
    d = knn_distance(Ztr, Zva[s:s + 1])[0]
    near = np.argsort(d)[:6]
    print("=" * 66)
    print(f"例子：验证集第 {s} 场（match_id={val.match_id.iloc[s]}）")
    print("=" * 66)
    print(f"  真实结果：{'队1赢' if yva[s] == 1 else '队2赢'}")
    print(f"  概率 {p_hand[s]:.4f} → 最近 {K} 个邻居里约 {p_hand[s]*K:.0f} 个是队1赢")
    print(f"\n  {'排名':<4}{'距离':>9}  {'标签'}")
    for r, i in enumerate(near, 1):
        print(f"  {r:<4}{d[i]:>9.3f}  {'队1赢' if yb[i] == 1 else '队2赢'}")

    # ---- 为什么必须标准化 ----
    print("\n" + "=" * 66)
    print("为什么必须先标准化")
    print("=" * 66)
    p_raw, _ = knn_predict_proba(both, yb, Xva, k=K)
    print(f"  不标准化  log loss {log_loss(yva, p_raw, labels=[0,1]):.4f}  "
          f"准确率 {((p_raw>.5)==yva).mean():.2%}")
    print(f"  标准化后  log loss {log_loss(yva, p_hand, labels=[0,1]):.4f}  "
          f"准确率 {((p_hand>.5)==yva).mean():.2%}")
    print("  原因：不标准化时距离几乎完全由量纲最大的特征（Elo 差，±400）决定，")
    print("        其余 40 个特征等于没参与。")


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
