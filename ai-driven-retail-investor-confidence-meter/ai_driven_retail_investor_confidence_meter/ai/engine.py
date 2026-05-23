from __future__ import annotations

import math

import numpy as np


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


class Engine:
    """TF-IDF + logistic classifier with token-level contribution attribution."""

    def __init__(self) -> None:
        pos = ["guaranteed moonshot buy now", "pump it to the moon urgent",
               "insider tip act fast easy money", "coordinated buy at open"]
        neg = ["quarterly earnings in line with guidance",
               "central bank held rates steady", "company reported revenue growth",
               "analyst maintains neutral rating"]
        corpus = pos + neg
        labels = [1] * len(pos) + [0] * len(neg)
        self.vec = TfidfVectorizer()
        X = self.vec.fit_transform(corpus)
        self.model = LogisticRegression(max_iter=500).fit(X, labels)
        self.vocab = self.vec.get_feature_names_out()

    def explain(self, features: dict) -> dict:
        text = str(features.get("text", ""))
        X = self.vec.transform([text])
        proba = float(self.model.predict_proba(X)[0, 1])
        row = X.toarray()[0]
        coef = self.model.coef_[0]
        contribs = []
        for i in range(len(self.vocab)):
            if row[i] > 0:
                contribs.append({"token": self.vocab[i],
                                 "contribution": round(float(row[i] * coef[i]), 4)})
        contribs.sort(key=lambda c: abs(c["contribution"]), reverse=True)
        top = contribs[0]["token"] if contribs else "n/a"
        return {"primitive": "nlp", "manipulation_probability": round(proba, 4),
                "flagged": proba >= 0.5,
                "summary": f"P(manipulative)={proba:.0%}; key token: {top}.",
                "attributions": contribs[:10]}
