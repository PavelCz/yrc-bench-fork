import importlib

import matplotlib


# The analysis modules select TkAgg for interactive use. Keep this unit test
# headless without changing the production plotting backend.
def ignore_backend_selection(*args, **kwargs):
    del args, kwargs


matplotlib.use = ignore_backend_selection
paper_plot = importlib.import_module("analyzing.paper_plot")


def test_reference_anchors_use_endpoint_medians():
    novice, expert = paper_plot._median_reference_anchors(
        [1.0, 2.0, 100.0],
        [5.0, 6.0, 200.0],
    )

    assert novice == 2.0
    assert expert == 6.0


def test_filtered_metric_uses_unfiltered_novice_median():
    novice, expert = paper_plot._median_reference_anchors(
        [10.0, 20.0, 30.0],
        [40.0, 50.0, 60.0],
        [1.0, 3.0, 100.0],
    )

    assert novice == 3.0
    assert expert == 50.0
