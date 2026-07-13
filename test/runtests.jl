using TestItemRunner

# Every test is a `@testitem` discovered recursively under `test/`, including
# the managed quality suite in `test/package/`. Tag-based filters let CI split
# the fast package tests from the slower quality checks (JET, Aqua, doctests).
if "skip_quality" in ARGS
    @run_package_tests filter=ti -> !(:quality in ti.tags)
elseif "quality_only" in ARGS
    @run_package_tests filter=ti -> :quality in ti.tags
elseif "readme_only" in ARGS
    @run_package_tests filter=ti -> :readme in ti.tags
else
    @run_package_tests
end
