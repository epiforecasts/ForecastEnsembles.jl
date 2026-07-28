# PACKAGE-OWNED — scaffold writes this once and never overwrites it.
#
# If this file defines `JET_REPORT_FILTER` (a `report -> Bool` predicate; a
# report is kept when it returns `true`), the runner switches from `test_package`
# to `report_package` + filter and fails only on reports the predicate keeps.
#
# No filter is currently needed: the package has no by-design method-less hooks.
