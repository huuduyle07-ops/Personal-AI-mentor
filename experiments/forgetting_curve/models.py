"""Half-life regression (HLR) and baselines from Settles & Meeder (2016), in plain numpy.

Every model predicts p_recall, the fraction of times a word was recalled correctly in a session,
from the time since the word was last seen (delta, in days) and the learner's history with it.

HLR:  p_hat = 2^(-delta / h_hat),   h_hat = 2^(theta . x),   x = [sqrt(1+right), sqrt(1+wrong), 1]
Loss: mean[(p - p_hat)^2 + alpha * (h - h_hat)^2] + lam * ||theta||^2,   where h = -delta / log2(p)
"""
import numpy as np

LN2 = np.log(2)
MIN_HALF_LIFE, MAX_HALF_LIFE = 15 / (24 * 60), 274.0  # 15 minutes to 9 months, in days (as in the paper)
MIN_P, MAX_P = 0.0001, 0.9999
FEATURES = ["sqrt_right", "sqrt_wrong", "bias"]


def features(right, wrong):
    return np.column_stack([np.sqrt(1 + right), np.sqrt(1 + wrong), np.ones(len(right))])


def target_half_life(p, delta):
    return np.clip(-delta / np.log2(np.clip(p, MIN_P, MAX_P)), MIN_HALF_LIFE, MAX_HALF_LIFE)


def recall(delta, half_life):
    return np.clip(2.0 ** (-delta / half_life), MIN_P, MAX_P)


class Constant:
    """Predict the training-set mean recall for everything."""
    name = "constant_mean"

    def fit(self, data):
        self.p = float(np.mean(data["p"]))
        return self

    def predict(self, data):
        n = len(data["p"])
        return np.full(n, self.p), np.full(n, np.nan)


class Leitner:
    """Fixed rule: half-life doubles per net correct answer, h = 2^(right - wrong)."""
    name = "leitner"

    def fit(self, data):
        return self

    def predict(self, data):
        h = np.clip(2.0 ** (data["right"] - data["wrong"]), MIN_HALF_LIFE, MAX_HALF_LIFE)
        return recall(data["delta_days"], h), h


class Pimsleur:
    """Fixed rule: h = 2^(2.35 * seen - 16.46), the paper's Pimsleur baseline."""
    name = "pimsleur"

    def fit(self, data):
        return self

    def predict(self, data):
        seen = data["right"] + data["wrong"]
        h = np.clip(2.0 ** (2.35 * seen - 16.46), MIN_HALF_LIFE, MAX_HALF_LIFE)
        return recall(data["delta_days"], h), h


def adam(gradient, theta, steps, lr=0.05):
    m, v = np.zeros_like(theta), np.zeros_like(theta)
    for step in range(1, steps + 1):
        g = gradient(theta)
        m = 0.9 * m + 0.1 * g
        v = 0.999 * v + 0.001 * g * g
        theta = theta - lr * (m / (1 - 0.9 ** step)) / (np.sqrt(v / (1 - 0.999 ** step)) + 1e-8)
    return theta


class LogisticRegression:
    """Learned baseline without a forgetting curve: sigmoid(w . [x, log(1+delta)]), fit to p by cross-entropy."""
    name = "logistic_regression"

    def __init__(self, lam=0.0, steps=400):
        self.lam, self.steps = lam, steps

    @staticmethod
    def design(data):
        return np.column_stack([features(data["right"], data["wrong"]), np.log1p(data["delta_days"])])

    def fit(self, data):
        X, p = self.design(data), data["p"]

        def gradient(w):
            q = 1 / (1 + np.exp(-(X @ w)))
            return X.T @ (q - p) / len(p) + 2 * self.lam * w

        self.w = adam(gradient, np.zeros(X.shape[1]), self.steps)
        return self

    def predict(self, data):
        q = 1 / (1 + np.exp(-(self.design(data) @ self.w)))
        return np.clip(q, MIN_P, MAX_P), np.full(len(q), np.nan)


class HalfLifeRegression:
    """HLR without per-word features, so it transfers to words (and languages) never seen in training."""
    name = "hlr"

    def __init__(self, alpha=0.01, lam=0.0, steps=400):
        self.alpha, self.lam, self.steps = alpha, lam, steps

    def loss_and_gradient(self, theta, X, p, delta, h):
        z = X @ theta
        zc = np.clip(z, np.log2(MIN_HALF_LIFE), np.log2(MAX_HALF_LIFE))
        h_hat = 2.0 ** zc
        p_hat = 2.0 ** (-delta / h_hat)
        inside = (z == zc).astype(float)  # no gradient where the half-life is clipped
        dp_dz = p_hat * LN2 ** 2 * delta / h_hat
        dh_dz = h_hat * LN2
        dloss_dz = (2 * (p_hat - p) * dp_dz + 2 * self.alpha * (h_hat - h) * dh_dz) * inside
        loss = np.mean((p - p_hat) ** 2 + self.alpha * (h - h_hat) ** 2) + self.lam * theta @ theta
        return loss, X.T @ dloss_dz / len(p) + 2 * self.lam * theta

    def fit(self, data):
        X, p, delta = features(data["right"], data["wrong"]), data["p"], data["delta_days"]
        h = target_half_life(p, delta)
        self.theta = adam(lambda t: self.loss_and_gradient(t, X, p, delta, h)[1], np.zeros(X.shape[1]), self.steps)
        return self

    def predict(self, data):
        z = features(data["right"], data["wrong"]) @ self.theta
        h = 2.0 ** np.clip(z, np.log2(MIN_HALF_LIFE), np.log2(MAX_HALF_LIFE))
        return recall(data["delta_days"], h), h


class HalfLifeRegressionLexeme(HalfLifeRegression):
    """Ablation: HLR plus one learned weight per word (the paper's full model). Unseen words get weight 0."""
    name = "hlr_lexeme"

    def fit(self, data):
        self.vocab = {word: i for i, word in enumerate(np.unique(data["lexeme"]))}
        index = np.array([self.vocab[word] for word in data["lexeme"]])
        X, p, delta = features(data["right"], data["wrong"]), data["p"], data["delta_days"]
        h = target_half_life(p, delta)
        base = X.shape[1]

        def gradient(params):
            theta, words = params[:base], params[base:]
            z = X @ theta + words[index]
            zc = np.clip(z, np.log2(MIN_HALF_LIFE), np.log2(MAX_HALF_LIFE))
            h_hat = 2.0 ** zc
            p_hat = 2.0 ** (-delta / h_hat)
            dloss_dz = (2 * (p_hat - p) * p_hat * LN2 ** 2 * delta / h_hat
                        + 2 * self.alpha * (h_hat - h) * h_hat * LN2) * (z == zc)
            word_grad = np.zeros_like(words)
            np.add.at(word_grad, index, dloss_dz)
            return np.concatenate([X.T @ dloss_dz, word_grad]) / len(p) + 2 * self.lam * params

        params = adam(gradient, np.zeros(base + len(self.vocab)), self.steps)
        self.theta, self.words = params[:base], params[base:]
        return self

    def predict(self, data):
        offset = np.array([self.words[self.vocab[w]] if w in self.vocab else 0.0 for w in data["lexeme"]])
        z = features(data["right"], data["wrong"]) @ self.theta + offset
        h = 2.0 ** np.clip(z, np.log2(MIN_HALF_LIFE), np.log2(MAX_HALF_LIFE))
        return recall(data["delta_days"], h), h


# ---- metrics ---------------------------------------------------------------------------------

def auc(labels, scores):
    """ROC-AUC via the rank-sum formula, with average ranks for ties."""
    labels = np.asarray(labels, bool)
    positives, negatives = labels.sum(), (~labels).sum()
    if not positives or not negatives:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores))
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_scores)) + 1]
    ends = np.r_[starts[1:], len(scores)]
    for start, end in zip(starts, ends):
        ranks[order[start:end]] = (start + end + 1) / 2
    return float((ranks[labels].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def spearman(a, b):
    def rank(x):
        r = np.empty(len(x))
        r[np.argsort(x, kind="mergesort")] = np.arange(len(x))
        return r
    ra, rb = rank(a), rank(b)
    return float(np.corrcoef(ra, rb)[0, 1])


def evaluate(p, p_hat, h_hat, delta):
    """MAE/MSE on p_recall; AUC for ranking sessions with p_recall >= 0.5 above those below;
    Spearman correlation of predicted vs observed-implied half-life (HLR-style models only)."""
    result = {"mae": float(np.mean(np.abs(p - p_hat))), "mse": float(np.mean((p - p_hat) ** 2)),
              "auc": auc(p >= 0.5, p_hat)}
    if not np.all(np.isnan(h_hat)):
        result["half_life_spearman"] = spearman(h_hat, target_half_life(p, delta))
    return result


def calibration(p, p_hat, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    index = np.clip(np.digitize(p_hat, edges) - 1, 0, bins - 1)
    return [{"bin": f"{edges[i]:.1f}-{edges[i + 1]:.1f}", "count": int((index == i).sum()),
             "mean_predicted": float(p_hat[index == i].mean()) if (index == i).any() else None,
             "mean_observed": float(p[index == i].mean()) if (index == i).any() else None}
            for i in range(bins)]


def bootstrap_mae_difference(users, err_a, err_b, resamples=1000, seed=0):
    """95% interval for MAE(a) - MAE(b), resampling whole learners (paired)."""
    ids, inverse = np.unique(users, return_inverse=True)
    sums_a, sums_b = np.bincount(inverse, err_a), np.bincount(inverse, err_b)
    counts = np.bincount(inverse)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(resamples):
        pick = rng.integers(0, len(ids), len(ids))
        n = counts[pick].sum()
        diffs.append(sums_a[pick].sum() / n - sums_b[pick].sum() / n)
    low, high = np.percentile(diffs, [2.5, 97.5])
    return {"difference": float(err_a.mean() - err_b.mean()), "ci95": [float(low), float(high)], "learners": len(ids)}
