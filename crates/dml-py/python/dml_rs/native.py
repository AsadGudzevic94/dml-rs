"""The low-level layer under the estimators: one function per model that fits from arrays and
returns the Rust result object.

The estimator classes (``dml_rs.PLR``, ``dml_rs.IRM``, ...) are the supported interface; these
functions and result classes are for code that needs the native objects directly. Their
signatures follow the Rust crate and may change in minor releases.
"""

from ._dml_rs import (
    AposResult,
    Blp,
    Bootstrap,
    CateResult,
    DmlResult,
    QteResult,
    RdFlexResult,
    fit_apo,
    fit_apos,
    fit_autodml,
    fit_cate,
    fit_did,
    fit_iivm,
    fit_irm,
    fit_lplr,
    fit_pliv,
    fit_plpr,
    fit_plr,
    fit_qte,
    fit_quantile,
    fit_rdflex,
    fit_ssm,
    fit_staggered_did,
)

__all__ = [
    "AposResult",
    "Blp",
    "Bootstrap",
    "CateResult",
    "DmlResult",
    "QteResult",
    "RdFlexResult",
    "fit_apo",
    "fit_apos",
    "fit_autodml",
    "fit_cate",
    "fit_did",
    "fit_iivm",
    "fit_irm",
    "fit_lplr",
    "fit_pliv",
    "fit_plpr",
    "fit_plr",
    "fit_qte",
    "fit_quantile",
    "fit_rdflex",
    "fit_ssm",
    "fit_staggered_did",
]
