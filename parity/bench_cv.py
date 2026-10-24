"""PLR with cross-validated Lasso nuisances (5 folds, LassoCV with 5 inner folds) and IRM with
LassoCV / LogisticRegressionCV: DoubleML with scikit-learn against dml-rs, on the same data and
sample splits. Both sides use scikit-learn's defaults; the estimates agree."""

import time
import warnings

import doubleml as dml
import numpy as np
from doubleml.irm.datasets import make_irm_data
from doubleml.plm.datasets import make_plr_CCDDHNR2018
from sklearn.linear_model import LassoCV, LogisticRegressionCV
from sklearn.model_selection import KFold

import dml_rs

warnings.simplefilter("ignore")


def best_of(f, reps=3):
    times = []
    for _ in range(reps):
        start = time.perf_counter()
        result = f()
        times.append(time.perf_counter() - start)
    return result, min(times)


def logit_cv():
    return LogisticRegressionCV(scoring="neg_log_loss", l1_ratios=(0,), use_legacy_attributes=False)


for n, p in [(2_000, 50), (10_000, 20)]:
    np.random.seed(0)
    x, y, d = make_plr_CCDDHNR2018(n_obs=n, dim_x=p, return_type="array")
    folds = list(KFold(5, shuffle=True, random_state=0).split(x))

    def doubleml_plr():
        model = dml.DoubleMLPLR(dml.DoubleMLData.from_arrays(x, y, d), LassoCV(), LassoCV(), n_folds=5,
                                draw_sample_splitting=False)
        return model.set_sample_splitting([folds]).fit()

    ref, t_ref = best_of(doubleml_plr)
    ours, t_ours = best_of(lambda: dml_rs.PLR(dml_rs.LassoCV(), dml_rs.LassoCV()).fit(x, y, d, folds=folds))
    print(f"PLR LassoCV n={n} p={p}: DoubleML {t_ref:.2f} s (coef {ref.coef[0]:.6f}), "
          f"dml-rs {t_ours:.3f} s (coef {ours.coef_[0]:.6f}), {t_ref / t_ours:.0f}x")

np.random.seed(1)
x, y, d = make_irm_data(n_obs=5_000, dim_x=20, return_type="array")
folds = list(KFold(5, shuffle=True, random_state=0).split(x))


def doubleml_irm():
    model = dml.DoubleMLIRM(dml.DoubleMLData.from_arrays(x, y, d), LassoCV(), logit_cv(), n_folds=5,
                            draw_sample_splitting=False)
    return model.set_sample_splitting([folds]).fit()


ref, t_ref = best_of(doubleml_irm)
ours, t_ours = best_of(lambda: dml_rs.IRM(dml_rs.LassoCV(), dml_rs.LogisticRegressionCV()).fit(x, y, d, folds=folds))
print(f"IRM LassoCV + LogisticRegressionCV n=5000 p=20: DoubleML {t_ref:.2f} s (coef {ref.coef[0]:.6f}), "
      f"dml-rs {t_ours:.3f} s (coef {ours.coef_[0]:.6f}), {t_ref / t_ours:.0f}x")
