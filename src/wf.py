"""Walk-forward: честная оценка прогноза вне выборки."""
import numpy as np, pandas as pd

def walk_forward(X, y, refit=252, start=2000, embargo=21, min_train=750, ridge=1e-6):
    """Расширяющееся окно. На момент T обучаемся только на строках, чьи таргеты
    УЖЕ полностью реализованы (t + embargo <= T-1), и предсказываем [T, T+refit).

    Возвращает pred (nan там, где предсказание не делалось) и bench —
    расширяющееся среднее таргета, известное на момент T (бенчмарк для OOS R^2).
    """
    n = len(y)
    pred = np.full(n, np.nan); bench = np.full(n, np.nan)
    for T in range(start, n, refit):
        tr_end = T - embargo
        rows = np.arange(0, tr_end)
        ok = ~np.isnan(y[rows]) & ~np.isnan(X[rows]).any(axis=1)
        rows = rows[ok]
        if len(rows) < min_train: continue
        Xt = X[rows]; yt = y[rows]
        mu, sd = Xt.mean(0), Xt.std(0); sd[sd == 0] = 1
        Z = np.c_[np.ones(len(rows)), (Xt - mu) / sd]
        A = Z.T @ Z + ridge * np.eye(Z.shape[1]); A[0, 0] -= ridge
        beta = np.linalg.solve(A, Z.T @ yt)
        te = np.arange(T, min(T + refit, n))
        Zte = np.c_[np.ones(len(te)), (X[te] - mu) / sd]
        p = Zte @ beta
        p[np.isnan(X[te]).any(axis=1)] = np.nan
        pred[te] = p; bench[te] = yt.mean()
    return pred, bench

def oos_r2(y, pred, bench):
    m = ~np.isnan(y) & ~np.isnan(pred) & ~np.isnan(bench)
    if m.sum() < 50: return np.nan, 0
    sse = ((y[m] - pred[m])**2).sum(); sst = ((y[m] - bench[m])**2).sum()
    return 1 - sse / sst, int(m.sum())

def oos_corr(y, pred):
    m = ~np.isnan(y) & ~np.isnan(pred)
    return np.corrcoef(y[m], pred[m])[0, 1] if m.sum() > 50 else np.nan

def walk_forward_select(X, y, k=5, refit=252, start=2000, embargo=21,
                        min_train=750, ridge=10.0, keep=()):
    """То же, но отбор k лучших признаков по |corr| ВНУТРИ обучающей части.
    keep — индексы столбцов, которые включаются всегда (например, HAR)."""
    n = len(y)
    pred = np.full(n, np.nan); bench = np.full(n, np.nan)
    keep = list(keep)
    chosen_log = []
    for T in range(start, n, refit):
        tr_end = T - embargo
        rows = np.arange(0, tr_end)
        ok = ~np.isnan(y[rows])
        rows = rows[ok]
        if len(rows) < min_train: continue
        cand = [j for j in range(X.shape[1]) if j not in keep]
        # признак доступен, если он не nan хотя бы на 60% обучающих строк
        cors = []
        for j in cand:
            m = ~np.isnan(X[rows, j])
            if m.sum() < 0.6 * len(rows): continue
            yy = y[rows][m]; xx = X[rows, j][m]
            if xx.std() == 0: continue
            cors.append((abs(np.corrcoef(xx, yy)[0, 1]), j))
        cors.sort(reverse=True)
        sel = keep + [j for _, j in cors[:k]]
        chosen_log.append((T, [j for _, j in cors[:k]]))
        rr = rows[~np.isnan(X[np.ix_(rows, sel)]).any(axis=1)]
        if len(rr) < min_train: continue
        Xt = X[np.ix_(rr, sel)]; yt = y[rr]
        mu, sd = Xt.mean(0), Xt.std(0); sd[sd == 0] = 1
        Z = np.c_[np.ones(len(rr)), (Xt - mu) / sd]
        A = Z.T @ Z + ridge * np.eye(Z.shape[1]); A[0, 0] -= ridge
        beta = np.linalg.solve(A, Z.T @ yt)
        te = np.arange(T, min(T + refit, n))
        Xte = X[np.ix_(te, sel)]
        Zte = np.c_[np.ones(len(te)), (Xte - mu) / sd]
        p = Zte @ beta
        p[np.isnan(Xte).any(axis=1)] = np.nan
        pred[te] = p; bench[te] = yt.mean()
    return pred, bench, chosen_log
