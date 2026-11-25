from sklearn.linear_model import LogisticRegression


def make_logistic(c=1.0, seed=42):
    # Input summaries are imputed and scaled by the train-only shared encoder.
    return LogisticRegression(C=c, max_iter=500, random_state=seed, solver="lbfgs")
