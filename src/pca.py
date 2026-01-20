#"mean"         平均ベクトル μ（中心化に使用）
#"eigvals"      固有値
#"components"   主成分
#"scores"       主成分得点
#"explained_ratio"   寄与率

import torch

def pca(X: torch.Tensor) -> dict:
    # 中心化
    mean = X.mean(dim=0, keepdim=True)
    Xc = X - mean

    #vの初期化
    d = Xc.shape[1]
    v = torch.randn(d, 1, dtype=X.dtype, device=X.device, requires_grad=True)

    #optimizer
    optimizer = torch.optim.SGD([v], lr=0.1)

    steps = 1000
    for _ in range(steps):
        y = Xc @ v
        loss = -(y.norm() ** 2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            v /= v.norm()

    #components
    components = v.detach()

    #主成分得点
    scores = Xc @ components

    #固有値
    n = X.shape[0]
    eigvals = (scores.pow(2).sum() / n).unsqueeze(0)

    #寄与率
    total_var = (Xc.pow(2).sum() / n)
    explained_ratio = eigvals / total_var

    return {
        "mean": mean,
        "eigvals": eigvals,
        "components": components,
        "scores": scores,
        "explained_ratio": explained_ratio,
    }

