import numpy as np

from forestfusion.data import make_synthetic_plot, make_synthetic_tree, render_depth_views
from forestfusion.data.segmentation import IGNORE_INDEX, SegmentationDataset
from forestfusion.data.transforms import normalize_tree, sample_points
from forestfusion.instance import instance_f1, segment_trees
from forestfusion.metrics import ConfusionMatrix

LABEL_MAP = {2: 0, 1: 1, 4: 2, 5: 3, 6: 4}


def test_sample_points_exact_count():
    rng = np.random.default_rng(0)
    assert len(sample_points(100, 40, rng)) == 40
    idx = sample_points(10, 40, rng)
    assert len(idx) == 40 and set(range(10)) <= set(idx.tolist())


def test_normalize_tree_unit_box_and_height():
    xyz, _ = make_synthetic_tree("spruce_like", np.random.default_rng(0), 2000, height=20.0)
    norm, h = normalize_tree(xyz + 100)
    assert np.abs(norm).max() <= 1.0 + 1e-9
    assert 18 < h < 23 and norm[:, 2].min() == 0


def test_render_views_shape_and_range():
    xyz, _ = make_synthetic_tree("birch_like", np.random.default_rng(0), 3000)
    views = render_depth_views(normalize_tree(xyz)[0], n_views=4, size=32, top_view=True)
    assert views.shape == (5, 32, 32)
    assert views.min() >= 0 and views.max() <= 1 and (views > 0).mean() > 0.02


def test_segmentation_dataset(tmp_path):
    plot = make_synthetic_plot(np.random.default_rng(0), size=20, n_trees=4, points_per_tree=1000)
    plot["classification"][:10] = 3  # "out-points" -> ignored
    np.savez(tmp_path / "p.npz", **plot)
    ds = SegmentationDataset([tmp_path / "p.npz"], LABEL_MAP, n_points=512, radius=3.0,
                             split="train", samples_per_epoch=5)
    s = ds[0]
    assert s["points"].shape == (512, 3) and s["labels"].shape == (512,)
    assert set(np.unique(s["labels"].numpy())) <= {0, 1, 2, 3, 4, IGNORE_INDEX}
    val = SegmentationDataset([tmp_path / "p.npz"], LABEL_MAP, n_points=512, radius=3.0, split="val")
    assert len(val) > 0
    freqs = val.class_frequencies(5)
    assert np.isclose(freqs.sum(), 1.0)


def test_confusion_matrix_metrics():
    cm = ConfusionMatrix(3)
    cm.update(np.array([0, 0, 1, 2, 2]), np.array([0, 1, 1, 2, 255]))
    assert cm.mat.sum() == 4  # ignore_index dropped
    np.testing.assert_allclose(cm.iou(), [0.5, 0.5, 1.0])
    assert cm.overall_accuracy() == 0.75


def test_instance_baseline_on_ground_truth_semantics():
    plot = make_synthetic_plot(np.random.default_rng(1), size=30, n_trees=6, points_per_tree=2000)
    sem = np.vectorize(lambda c: LABEL_MAP.get(c, 255))(plot["classification"])
    trees = segment_trees(plot["xyz"], sem, stem_class=2, tree_classes=(2, 3, 4), eps=0.6)
    res = instance_f1(trees, plot["treeID"])
    assert res["f1"] > 0.6  # oracle semantics on well-separated synthetic trees
