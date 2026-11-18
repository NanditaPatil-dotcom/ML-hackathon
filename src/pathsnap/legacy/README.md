# Archived experiments

This package retains earlier standalone experiments and wrappers so their
implementation history is not lost. The current preprocessing, comparison,
and registered model implementations live in `pathsnap.data.preprocessing`,
`pathsnap.evaluation`, and `pathsnap.models`.

Do not import archived registry modules into the active model suite. In
particular, ExtraTrees is registered only by `pathsnap.models.extra_trees`.
