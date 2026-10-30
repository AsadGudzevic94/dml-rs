# Python API

All classes are available from the top-level package, e.g. `dml_rs.PLR`.

## Models

::: dml_rs.PLR
::: dml_rs.LPLR
::: dml_rs.PLPR
::: dml_rs.IRM
::: dml_rs.APO
::: dml_rs.APOS
::: dml_rs.CausalContrast
::: dml_rs.QTE
::: dml_rs.PotentialQuantiles
::: dml_rs.PQ
::: dml_rs.LPQ
::: dml_rs.CVAR
::: dml_rs.SSM
::: dml_rs.PLIV
::: dml_rs.IIVM
::: dml_rs.DID
::: dml_rs.DIDCS
::: dml_rs.StaggeredDID
::: dml_rs.DIDAggregation
::: dml_rs.RDFlex
::: dml_rs.AutoDML

## Functionals and Riesz representers

::: dml_rs.ATE
::: dml_rs.AverageDerivative
::: dml_rs.ShiftEffect
::: dml_rs.PolicyEffect
::: dml_rs.CustomFunctional
::: dml_rs.RieszLasso

## Tuning and learner parameters

PLR, LPLR, PLPR, IRM, APO, SSM, PLIV, IIVM, DID and DIDCS share these methods, and their `fit`
takes `store_models=True` to keep the fitted learners in `models_`.

::: dml_rs._tuning.TuningMixin
    options:
      show_root_heading: false
      members: ["params_names", "learner_names", "params", "set_ml_nuisance_params", "tune", "tune_ml_models"]

::: dml_rs.OptunaResult

## Data

::: dml_rs.DmlData
    options:
      members: ["from_arrays"]

::: dml_rs.datasets
    options:
      show_root_heading: false
      members_order: source
      filters: ["!^_"]

## Combining estimates

::: dml_rs.Framework
    options:
      filters: ["!^_"]
::: dml_rs.concat

## Heterogeneous effects

::: dml_rs.CausalForest
::: dml_rs.BLP
::: dml_rs.SLearner
::: dml_rs.TLearner
::: dml_rs.XLearner
::: dml_rs.RLearner
::: dml_rs.DRLearner
::: dml_rs.PolicyTree

## Learners

::: dml_rs.Ols
::: dml_rs.Lasso
::: dml_rs.LassoCV
::: dml_rs.LogisticRegression
::: dml_rs.LogisticRegressionCV
::: dml_rs.GridSearchRegressor
::: dml_rs.GridSearchClassifier
::: dml_rs.RandomForestRegressor
::: dml_rs.RandomForestClassifier
::: dml_rs.GradientBoostingRegressor
::: dml_rs.GradientBoostingClassifier
::: dml_rs.HistGradientBoostingRegressor
::: dml_rs.HistGradientBoostingClassifier
