import doubleml
import pytest

# The tests compare with the pinned DoubleML (parity/requirements.txt). The scheduled upstream job
# runs them against DoubleML's main branch, where these results differ on purpose; anything else
# that fails there is a change to look at. See "Known differences" in docs/validation.md.
PINNED_DOUBLEML = "0.11.4"
DIFFERS_ON_DOUBLEML_MAIN = {
    "test_datasets.py::test_did_cs2021_panels_and_cross_sections": "keeps the never-treated units (#437)",
    "test_framework.py::test_arithmetic_matches_doubleml": "se of a negative multiple is fixed (#421)",
    "test_sensitivity_plot.py::test_contours_match_doubleml": "contours are no longer transposed (#415)",
    "test_staggered.py::test_staggered_did_agrees_with_doubleml": "new not-yet-treated controls before treatment (#393)",
    "test_tuning.py::test_tune_ml_models_matches_doubleml[PLR-IV]": "folds are reported, so trials can be pruned",
}


def pytest_collection_modifyitems(items):
    if doubleml.__version__ == PINNED_DOUBLEML:
        return
    for item in items:
        name = item.nodeid.split("/")[-1]
        for known, reason in DIFFERS_ON_DOUBLEML_MAIN.items():
            if name == known or name.startswith(known + "["):
                item.add_marker(pytest.mark.xfail(reason=reason, strict=False))
