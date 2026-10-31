//! With the `serde` feature, fitted learners and results serialize and come back unchanged.

use dml_core::learner::{Classifier, Regressor};
use dml_core::{
    AposFit, Blp, Bootstrap, CausalForestFit, DidAggregation, DmlData, DmlFit,
    GradientBoostingClassifier, GradientBoostingRegressor, HistGradientBoostingClassifier,
    HistGradientBoostingRegressor, InstrumentalForestFit, Lasso, LassoCV, LogisticRegression,
    LogisticRegressionCV, MultiArmCausalForestFit, Ols, Plr, PolicyTree, QteFit, QuantileForest,
    RandomForestClassifier, RandomForestRegressor, RdFlexFit, RegressionForest,
    SensitivityBenchmark, SensitivityResult, StaggeredDidFit,
};
use ndarray::{Array1, Array2};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{de::DeserializeOwned, Serialize};

fn round_trip<T: Serialize + DeserializeOwned>(value: &T) -> T {
    bincode::deserialize(&bincode::serialize(value).unwrap()).unwrap()
}

#[test]
fn every_learner_and_result_type_is_serializable() {
    fn check<T: Serialize + DeserializeOwned>() {}
    check::<Ols>();
    check::<Lasso>();
    check::<LassoCV>();
    check::<LogisticRegression>();
    check::<LogisticRegressionCV>();
    check::<RandomForestRegressor>();
    check::<RandomForestClassifier>();
    check::<GradientBoostingRegressor>();
    check::<GradientBoostingClassifier>();
    check::<HistGradientBoostingRegressor>();
    check::<HistGradientBoostingClassifier>();
    check::<RegressionForest>();
    check::<QuantileForest>();
    check::<DmlFit>();
    check::<Bootstrap>();
    check::<Blp>();
    check::<SensitivityResult>();
    check::<SensitivityBenchmark>();
    check::<PolicyTree>();
    check::<AposFit>();
    check::<QteFit>();
    check::<StaggeredDidFit>();
    check::<DidAggregation>();
    check::<RdFlexFit>();
    check::<CausalForestFit>();
    check::<InstrumentalForestFit>();
    check::<MultiArmCausalForestFit>();
}

fn data(n: usize) -> (Array2<f64>, Array1<f64>, Array1<f64>) {
    let mut rng = ChaCha8Rng::seed_from_u64(3);
    let x = Array2::from_shape_simple_fn((n, 3), || rng.gen::<f64>() * 2.0 - 1.0);
    let noise = |rng: &mut ChaCha8Rng| Array1::from_shape_simple_fn(n, || rng.gen::<f64>() - 0.5);
    let d: Array1<f64> =
        x.column(0).mapv(|v| v + 0.3) + x.column(1).mapv(|v| 0.5 * v) + noise(&mut rng);
    let y = &d * 0.5 + x.column(2).mapv(|v| v.sin()) + x.column(1) + noise(&mut rng);
    (x, y, d)
}

#[test]
fn fitted_learners_predict_the_same_after_a_round_trip() {
    let (x, y, d) = data(300);
    let labels = d.mapv(|v| f64::from(v > 0.3));
    let mut hist = HistGradientBoostingRegressor::new().max_iter(20);
    hist.fit(x.view(), y.view()).unwrap();
    assert_eq!(round_trip(&hist).predict(x.view()), hist.predict(x.view()));
    let mut forest = RegressionForest::new();
    forest.fit(x.view(), y.view()).unwrap();
    assert_eq!(
        round_trip(&forest).predict_with_variance(x.view()).unwrap(),
        forest.predict_with_variance(x.view()).unwrap()
    );
    let mut clf = RandomForestClassifier::new().n_estimators(20);
    clf.fit(x.view(), labels.view()).unwrap();
    assert_eq!(
        round_trip(&clf).predict_proba(x.view()),
        clf.predict_proba(x.view())
    );
}

#[test]
fn dml_results_survive_a_round_trip() {
    let (x, y, d) = data(400);
    let data = DmlData::new(y, d, x).unwrap();
    let fit = Plr::new(Ols::new(), Ols::new()).fit(&data).unwrap();
    let back = round_trip(&fit);
    assert_eq!(back.coef(), fit.coef());
    assert_eq!(back.se(), fit.se());
    assert_eq!(back.to_string(), fit.to_string());
    let params = dml_core::SensitivityParams::default();
    assert_eq!(
        back.sensitivity_analysis(&params).unwrap().to_string(),
        fit.sensitivity_analysis(&params).unwrap().to_string()
    );
}
