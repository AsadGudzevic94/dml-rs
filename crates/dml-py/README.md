# dml-rs (Python)

Fast double / debiased machine learning for causal inference, implemented in Rust
and validated against [DoubleML](https://docs.doubleml.org).

```python
import dml_rs
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

# Partially linear regression with any scikit-learn learners
plr = dml_rs.PLR(ml_l=RandomForestRegressor(), ml_m=RandomForestRegressor(), n_folds=5, n_rep=3)
plr.fit(X, y, d)
print(plr.summary)

# Built-in Rust learners run the whole cross-fitting in parallel without the GIL
plr = dml_rs.PLR(ml_l=dml_rs.Lasso(alpha=0.01), ml_m=dml_rs.Lasso(alpha=0.01)).fit(X, y, d)

# Average treatment effect of a binary treatment
irm = dml_rs.IRM(ml_g=RandomForestRegressor(), ml_m=RandomForestClassifier(), score="ATE").fit(X, y, d)

# Joint inference over several treatments
plr.bootstrap(method="normal", n_boot=1000)
plr.confint(joint=True)
plr.p_adjust("romano-wolf")
```

See the [repository](https://github.com/AsadGudzevic94/dml-rs) for details.
