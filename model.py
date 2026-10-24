"""Meta-modelo: estima P(trade ganador) con un modelo ÚNICO entrenado sobre todos los
trades cerrados (pooled) y probabilidades calibradas. Esto da cientos de muestras en vez
de ~20 por activo, que era el problema del Random Forest por ticker."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from .config import FEATURE_COLS, ModelConfig


def purged_folds(starts, ends, n_splits: int, embargo_days: int):
    """Genera (train_idx, test_idx): bloques contiguos de tiempo; el entrenamiento excluye todo trade cuyo intervalo
    toque [inicio_bloque - embargo, fin_bloque + embargo]."""
    starts, ends = np.asarray(starts, dtype="datetime64[ns]"), np.asarray(ends, dtype="datetime64[ns]")
    order = np.argsort(starts, kind="stable")
    emb = np.timedelta64(int(embargo_days), "D")
    for test in np.array_split(order, n_splits):
        if len(test) == 0:
            continue
        t0, t1 = starts[test].min(), ends[test].max()
        ok = (ends < t0 - emb) | (starts > t1 + emb)
        ok[test] = False
        yield np.where(ok)[0], test


class MetaModel:
    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        self.clf = None
        self.platt: LogisticRegression | None = None
        self.prior = 0.4
        self.payoff = 1.5            # ganancia media / pérdida media, en múltiplos de R
        self.payoff_sys: dict[str, float] = {}
        self.medians: pd.Series | None = None
        self.n_train = 0
        self.mode = "prior"          # "prior" (sin modelo) o "rf+platt"

    def fit(self, trades: pd.DataFrame) -> "MetaModel":
        t = trades.dropna(subset=["R_net"])
        self.n_train = len(t)
        y = (t["R_net"] > 0).astype(int).to_numpy()
        if len(t) == 0:
            return self
        self.prior = float(y.mean())
        self.payoff = self._payoff(t)
        if "system" in t.columns:
            for sname, g in t.groupby("system"):
                if len(g) >= 15:
                    self.payoff_sys[sname] = self._payoff(g)
        self.medians = t[FEATURE_COLS].median()
        if len(t) >= 60 and y.sum() >= 12 and (len(y) - y.sum()) >= 12:
            X = t[FEATURE_COLS].fillna(self.medians).fillna(0.0).to_numpy()
            rf = RandomForestClassifier(
                n_estimators=self.cfg.n_estimators, max_depth=self.cfg.max_depth, oob_score=True,
                min_samples_leaf=self.cfg.min_samples_leaf, random_state=self.cfg.random_state, n_jobs=-1).fit(X, y)
            oof = None
            if {"signal_date", "exit_date"} <= set(t.columns):
                oof = self._purged_oof(X, y, t["signal_date"].to_numpy(), t["exit_date"].to_numpy())
            if oof is None:     # sin fechas: respaldo con OOB (menos honesto: trades superpuestos comparten información)
                oof = np.nan_to_num(rf.oob_decision_function_[:, 1], nan=self.prior)
            self.platt = LogisticRegression(C=1.0).fit(self._logit(oof), y)
            self.clf = rf
            self.mode = "rf+platt"
        return self

    def _purged_oof(self, X, y, starts, ends):
        """Predicciones fuera de muestra con CV por bloques de TIEMPO, purgando de entrenamiento los trades cuyo
        intervalo [señal, salida] se solapa con el bloque de prueba (+ embargo). Evita que la calibración vea
        información de trades superpuestos (López de Prado)."""
        oof = np.full(len(y), np.nan)
        for train, test in purged_folds(starts, ends, self.cfg.cv_splits, self.cfg.embargo_days):
            ytr = y[train]
            if len(train) < 30 or ytr.sum() < 5 or (len(ytr) - ytr.sum()) < 5:
                oof[test] = self.prior
                continue
            rf = RandomForestClassifier(n_estimators=self.cfg.cv_trees, max_depth=self.cfg.max_depth,
                                        min_samples_leaf=self.cfg.min_samples_leaf, random_state=self.cfg.random_state,
                                        n_jobs=-1).fit(X[train], ytr)
            oof[test] = rf.predict_proba(X[test])[:, 1]
        return np.where(np.isnan(oof), self.prior, oof)

    @staticmethod
    def _logit(p: np.ndarray) -> np.ndarray:
        p = np.clip(np.asarray(p, dtype=float), 0.01, 0.99)
        return np.log(p / (1 - p)).reshape(-1, 1)

    @staticmethod
    def _payoff(t: pd.DataFrame) -> float:
        wins, losses = t.loc[t["R_net"] > 0, "R_net"], t.loc[t["R_net"] <= 0, "R_net"]
        if len(wins) and len(losses) and losses.mean() < 0:
            return float(wins.mean() / abs(losses.mean()))
        return 1.5

    def payoff_for(self, system: str) -> float:
        return self.payoff_sys.get(system, self.payoff)

    def predict(self, feats: pd.DataFrame) -> np.ndarray:
        if self.clf is None:
            return np.full(len(feats), self.prior)
        X = feats[FEATURE_COLS].fillna(self.medians).fillna(0.0).to_numpy()
        raw = self.clf.predict_proba(X)[:, 1]
        return np.clip(self.platt.predict_proba(self._logit(raw))[:, 1], 0.02, 0.98)
