import importlib
from types import SimpleNamespace

import pytest


def test_chan_cap_optimizer_returns_solution_structure():
    np = pytest.importorskip("numpy")
    pytest.importorskip("scipy")

    from Implementations.c3e import ChanCapConEst

    data = SimpleNamespace(x=np.ones((8, 4)), num_edges=12)
    estimator = ChanCapConEst(data=data, eta=0.5)

    rounded_layers, dropout_schedules, channel_caps = estimator.optimize_weights(
        H=0.0,
        max_layers=2,
    )

    assert len(rounded_layers) == len(dropout_schedules) == len(channel_caps)
    assert rounded_layers
    assert all(isinstance(width, int) for width in rounded_layers[0])
    assert all(0.0 <= prob <= 1.0 for prob in dropout_schedules[0])


def test_chan_cap_optimizer_uses_soft_width_guard_not_hard_bound(monkeypatch):
    np = pytest.importorskip("numpy")

    from Implementations import c3e

    data = SimpleNamespace(x=np.ones((10, 4)), num_edges=15)
    estimator = c3e.ChanCapConEst(data=data, eta=0.5)
    width_guard = estimator.regularized_width_guard()
    observed_bounds = []
    observed_starts = []

    def fake_minimize(fun, x0, method, bounds, constraints, options):
        observed_starts.append(x0.copy())
        observed_bounds.append(bounds)
        return SimpleNamespace(success=True, x=x0.copy())

    monkeypatch.setattr(c3e, "minimize", fake_minimize)
    monkeypatch.setattr(estimator, "constraint", lambda widths, H=0.0: 1.0)
    monkeypatch.setattr(estimator, "objective", lambda widths: -float(np.sum(widths)))

    rounded_layers, _, channel_caps = estimator.optimize_weights(H=0.0, max_layers=2)

    assert observed_bounds
    assert all(
        bounds == [(2.0, c3e.OPTIMIZER_WIDTH_BOUND), (2.0, c3e.OPTIMIZER_WIDTH_BOUND)]
        for bounds in observed_bounds
    )
    assert any(np.allclose(start, np.full(2, 2.0)) for start in observed_starts)
    assert all(np.all(start <= width_guard) for start in observed_starts)
    assert rounded_layers
    assert channel_caps[0] > -0.5 * estimator.penalty


def test_chan_cap_optimizer_returns_candidates_inside_valid_window(monkeypatch):
    np = pytest.importorskip("numpy")

    from Implementations import c3e

    data = SimpleNamespace(x=np.ones((10, 4)), num_edges=15)
    estimator = c3e.ChanCapConEst(data=data, eta=0.5)

    def fake_minimize(fun, x0, method, bounds, constraints, options):
        return SimpleNamespace(success=True, x=np.full(len(x0), 10.0))

    def fake_constraint(widths, H=0.0):
        phi0 = 3.0 if len(widths) == 2 else 5.0
        return phi0 - H

    monkeypatch.setattr(c3e, "minimize", fake_minimize)
    monkeypatch.setattr(estimator, "constraint", fake_constraint)
    monkeypatch.setattr(estimator, "objective", lambda widths: -float(len(widths)))

    rounded_layers, _, channel_caps = estimator.optimize_weights(H=2.0, max_layers=4)

    assert rounded_layers == [[10, 10]]
    assert channel_caps == [2.0]


def test_chan_cap_optimizer_rejects_penalty_state_solutions(monkeypatch):
    np = pytest.importorskip("numpy")

    from Implementations import c3e

    data = SimpleNamespace(x=np.ones((10, 4)), num_edges=15)
    estimator = c3e.ChanCapConEst(data=data, eta=0.5)

    def fake_minimize(fun, x0, method, bounds, constraints, options):
        return SimpleNamespace(success=True, x=np.full(len(x0), 2.0))

    monkeypatch.setattr(c3e, "minimize", fake_minimize)
    monkeypatch.setattr(estimator, "objective", lambda widths: estimator.penalty)

    with pytest.raises(RuntimeError, match="non-penalty solution"):
        estimator.optimize_weights(H=0.0, max_layers=2)


def test_filter_penalty_solutions_drops_invalid_candidates():
    from Implementations.train_val_test import filter_penalty_solutions

    layers, dropouts, capacities, skipped = filter_penalty_solutions(
        rounded_layers=[[4, 4], [8, 8]],
        dropout_schedules=[[0.1, 0.1], [0.2, 0.2]],
        channel_caps=[-1_000_000.0, 3.5],
        penalty=1_000_000.0,
    )

    assert layers == [[8, 8]]
    assert dropouts == [[0.2, 0.2]]
    assert capacities == [3.5]
    assert skipped == 1


def test_create_masks_are_deterministic_and_disjoint():
    torch = pytest.importorskip("torch")
    Data = pytest.importorskip("torch_geometric.data").Data
    pytest.importorskip("torch_sparse")

    from Implementations.utility import create_masks

    data = Data(
        x=torch.ones(12, 3),
        y=torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]),
        num_nodes=12,
    )

    first = create_masks(data, train_per_class=1, num_val=3, num_test=3, seed=7)
    second = create_masks(data, train_per_class=1, num_val=3, num_test=3, seed=7)

    for left, right in zip(first, second):
        assert torch.equal(left, right)

    train_mask, val_mask, test_mask = first
    assert int(train_mask.sum()) == 3
    assert int(val_mask.sum()) == 3
    assert int(test_mask.sum()) == 3
    assert not torch.any(train_mask & val_mask)
    assert not torch.any(train_mask & test_mask)
    assert not torch.any(val_mask & test_mask)


def test_propagation_analyzer_constructs_from_tiny_pyg_graph():
    torch = pytest.importorskip("torch")
    Data = pytest.importorskip("torch_geometric.data").Data
    pytest.importorskip("scipy")

    from Implementations.propanalyzer import PropagationVarianceAnalyzer

    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    data = Data(edge_index=edge_index, num_nodes=3)

    analyzer = PropagationVarianceAnalyzer(data, method="gcn")

    assert analyzer.compute_variance() > 0.0


def test_apply_variance_guard_floors_shrinking_profile():
    np = pytest.importorskip("numpy")

    from Implementations.propanalyzer import apply_variance_guard

    guarded = apply_variance_guard(np.array([1.0, 0.2, 0.1]), 0.95)

    assert np.allclose(guarded, np.array([1.0, 0.95, 0.9025]))


def test_apply_variance_guard_leaves_flat_and_increasing_profiles_unchanged():
    np = pytest.importorskip("numpy")

    from Implementations.propanalyzer import apply_variance_guard

    values = np.array([1.0, 1.0, 1.2, 1.2])

    assert np.allclose(apply_variance_guard(values, 0.95), values)


def test_apply_variance_guard_can_be_disabled():
    np = pytest.importorskip("numpy")

    from Implementations.propanalyzer import apply_variance_guard

    values = np.array([1.0, 0.1, 0.01])

    assert np.allclose(apply_variance_guard(values, 0.0), values)


def test_apply_variance_guard_rejects_invalid_ratio():
    np = pytest.importorskip("numpy")

    from Implementations.propanalyzer import apply_variance_guard

    with pytest.raises(ValueError, match="variance guard ratio"):
        apply_variance_guard(np.array([1.0, 0.5]), 1.1)


def test_variance_guard_defaults_remain_enabled(monkeypatch):
    from Implementations.train_val_test import parse_args as parse_train_args
    from tools.inspect_cora_gat_c3e import parse_args as parse_gat_estimation_args
    from tools.run_cora_eta045_all_ops import parse_args as parse_all_ops_args

    monkeypatch.setattr("sys.argv", ["train_val_test.py"])
    assert parse_train_args().variance_guard_ratio == pytest.approx(0.95)

    monkeypatch.setattr("sys.argv", ["inspect_cora_gat_c3e.py"])
    assert parse_gat_estimation_args().variance_guard_ratio == pytest.approx(0.95)

    monkeypatch.setattr("sys.argv", ["run_cora_eta045_all_ops.py"])
    assert parse_all_ops_args().variance_guard_ratio == pytest.approx(0.95)


def test_activation_mode_aliases_normalize_to_canonical_values():
    from Model_factory.activations import activation_flags, normalize_activation_mode

    assert normalize_activation_mode("first-only") == "first-on"
    assert normalize_activation_mode("all") == "all-on"
    assert normalize_activation_mode("none") == "all-off"
    assert activation_flags("first-on", 3) == [True, False, False]
    assert activation_flags("all-on", 3) == [True, True, True]
    assert activation_flags("all-off", 3) == [False, False, False]


def test_attention_variance_stats_matches_tiny_sparse_matrix():
    torch = pytest.importorskip("torch")

    from Implementations.empirical_variance import attention_variance_stats

    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    alpha = torch.tensor([0.25, 0.75], dtype=torch.float32)

    stats = attention_variance_stats(edge_index, alpha, num_nodes=2)

    values = [0.0, 0.25, 0.75, 0.0]
    expected_mean = sum(values) / 4
    expected_mean_sq = sum(value * value for value in values) / 4
    expected_var = expected_mean_sq - expected_mean * expected_mean
    assert stats["heads"] == 1
    assert stats["per_head"] == pytest.approx([expected_var])
    assert stats["mean"] == pytest.approx(expected_var)
    assert stats["median"] == pytest.approx(expected_var)
    assert stats["max"] == pytest.approx(expected_var)
    assert stats["geometric_mean"] == pytest.approx(expected_var)


def test_attention_variance_stats_handles_multi_head_attention():
    torch = pytest.importorskip("torch")

    from Implementations.empirical_variance import attention_variance_stats

    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    alpha = torch.tensor([[0.25, 0.5], [0.75, 0.5]], dtype=torch.float32)

    stats = attention_variance_stats(edge_index, alpha, num_nodes=2)

    assert stats["heads"] == 2
    assert len(stats["per_head"]) == 2
    assert stats["max"] >= stats["median"] >= min(stats["per_head"])
    assert stats["geometric_mean"] > 0.0


def test_attention_variance_stats_rejects_non_finite_attention():
    torch = pytest.importorskip("torch")

    from Implementations.empirical_variance import attention_variance_stats

    edge_index = torch.tensor([[0], [1]], dtype=torch.long)
    alpha = torch.tensor([float("nan")], dtype=torch.float32)

    with pytest.raises(ValueError, match="finite"):
        attention_variance_stats(edge_index, alpha, num_nodes=2)


def test_load_dataset_does_not_require_ogb_for_cora(monkeypatch, tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    pytest.importorskip("tqdm")

    module = importlib.import_module("Implementations.train_val_test")
    sentinel = object()

    def fake_planetoid(root, name, transform):
        assert root == str(tmp_path)
        assert name == "Cora"
        return sentinel

    def fail_if_called(name):
        raise AssertionError(f"unexpected import_module call for {name}")

    monkeypatch.setattr(module, "Planetoid", fake_planetoid)
    monkeypatch.setattr(module, "import_module", fail_if_called)

    args = SimpleNamespace(dataset="Cora", data_root=tmp_path)

    assert module.load_dataset(args) is sentinel


def test_load_dataset_reports_missing_ogb(monkeypatch, tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    pytest.importorskip("tqdm")

    module = importlib.import_module("Implementations.train_val_test")

    def missing_ogb(name):
        raise ModuleNotFoundError("No module named 'ogb'")

    monkeypatch.setattr(module, "import_module", missing_ogb)

    args = SimpleNamespace(dataset="ogbn-arxiv", data_root=tmp_path)

    with pytest.raises(ModuleNotFoundError, match="ogb is required for ogbn-\\* datasets"):
        module.load_dataset(args)


def test_gdc_training_transform_uses_analyzer_defaults(monkeypatch):
    torch = pytest.importorskip("torch")
    Data = pytest.importorskip("torch_geometric.data").Data

    module = importlib.import_module("Implementations.train_val_test")
    captured = {}

    class FakeGDC:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def __call__(self, data):
            data.edge_attr = torch.ones(data.edge_index.size(1))
            data.was_gdc_transformed = True
            return data

    monkeypatch.setattr(module.T, "GDC", FakeGDC)
    data = Data(
        x=torch.ones(3, 2),
        edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
    )
    args = SimpleNamespace(prop_method="gdc")

    transformed = module.apply_training_graph_transform(data, args)

    assert transformed.was_gdc_transformed
    assert captured["self_loop_weight"] == 1.0
    assert captured["normalization_in"] == "sym"
    assert captured["normalization_out"] == "col"
    assert captured["diffusion_kwargs"] == {"method": "ppr", "alpha": 0.05}
    assert captured["sparsification_kwargs"] == {"method": "topk", "k": 64, "dim": 0}
    assert captured["exact"] is True


def test_model_validates_dropout_schedule_length_and_runs_gcn_forward():
    torch = pytest.importorskip("torch")
    Data = pytest.importorskip("torch_geometric.data").Data
    pytest.importorskip("torch_sparse")

    from Model_factory.model import Model

    with pytest.raises(ValueError, match="drop_probs length"):
        Model(prop_layer=[3, 4, 4], num_class=2, drop_probs=[0.1], conv_methods="gcn")

    model = Model(prop_layer=[3, 4], num_class=2, drop_probs=[0.0], conv_methods="gcn")
    data = Data(
        x=torch.randn(3, 3),
        edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long),
    )

    out = model(data)

    assert tuple(out.shape) == (3, 2)


def test_model_defaults_to_first_on_activation_and_supports_activation_kinds():
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_sparse")

    from Model_factory.model import Model

    model = Model(prop_layer=[3, 4, 4], num_class=2, drop_probs=[0.0, 0.0], conv_methods="gcn")

    assert model.activation_mode == "first-on"
    assert model.use_activations == [True, False]

    silu_model = Model(
        prop_layer=[3, 4, 4],
        num_class=2,
        drop_probs=[0.0, 0.0],
        conv_methods="gcn",
        activation_mode="all-on",
        activation_kind="silu",
    )
    gelu_model = Model(
        prop_layer=[3, 4, 4],
        num_class=2,
        drop_probs=[0.0, 0.0],
        conv_methods="gcn",
        activation_mode="all-off",
        activation_kind="gelu",
    )

    assert all(isinstance(module, torch.nn.SiLU) for module in silu_model.activations)
    assert all(isinstance(module, torch.nn.GELU) for module in gelu_model.activations)
    assert gelu_model.use_activations == [False, False]


def test_model_residual_projection_is_used_for_depth_greater_than_one():
    torch = pytest.importorskip("torch")
    Data = pytest.importorskip("torch_geometric.data").Data
    pytest.importorskip("torch_sparse")

    from Model_factory.model import Model

    model = Model(prop_layer=[3, 4, 4], num_class=2, drop_probs=[0.0, 0.0], conv_methods="gcn")
    calls = []
    model.residuals[0].register_forward_hook(lambda module, inputs, output: calls.append(True))
    data = Data(
        x=torch.randn(3, 3),
        edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long),
    )

    out = model(data)

    assert tuple(out.shape) == (3, 2)
    assert calls


def test_run_solution_preserves_original_training_error(monkeypatch, tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")

    from Implementations import train_val_test

    original_error = ValueError("boom")

    def explode(*args, **kwargs):
        raise original_error

    monkeypatch.setattr(train_val_test, "train", explode)

    args = SimpleNamespace(
        save_dir=tmp_path,
        prop_method="gcn",
        device="cpu",
        lr=1e-3,
        weight_decay=0.0,
        epochs=1,
        patience=1,
    )
    data = SimpleNamespace(x=torch.randn(4, 3))
    dataset = SimpleNamespace(num_classes=2)

    with pytest.raises(RuntimeError, match="Training failed for solution") as exc_info:
        train_val_test.run_solution(
            data=data,
            dataset=dataset,
            layers=[4],
            dropout=[0.0],
            channel_capacity=1.0,
            args=args,
        )

    assert exc_info.value.__cause__ is original_error
    assert tmp_path.exists()


def test_gat_candidate_snap_width_down_rounds_to_positive_multiple():
    from tools.run_cora_gat_c3e_candidates import snap_width_down

    assert snap_width_down(2053, 4) == 2052
    assert snap_width_down(1164, 4) == 1164
    with pytest.raises(ValueError, match="too small"):
        snap_width_down(3, 4)


def test_deep_residual_gat_validates_dropout_length():
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")

    from tools.run_cora_gat_c3e_candidates import DeepResidualGAT

    with pytest.raises(ValueError, match="dropouts length"):
        DeepResidualGAT(
            in_channels=3,
            layer_widths=[4, 4],
            num_classes=2,
            dropouts=[0.0],
            heads=2,
            attention_dropout=0.0,
        )


def test_deep_residual_gat_activation_modes():
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")

    from tools.run_cora_gat_c3e_candidates import DeepResidualGAT

    first_only = DeepResidualGAT(
        in_channels=3,
        layer_widths=[4, 4, 4],
        num_classes=2,
        dropouts=[0.0, 0.0, 0.0],
        heads=2,
        attention_dropout=0.0,
        activation_mode="first-only",
    )
    assert first_only.activation_mode == "first-on"
    assert [first_only.uses_activation(idx) for idx in range(3)] == [True, False, False]

    all_layers = DeepResidualGAT(
        in_channels=3,
        layer_widths=[4, 4, 4],
        num_classes=2,
        dropouts=[0.0, 0.0, 0.0],
        heads=2,
        attention_dropout=0.0,
        activation_mode="all",
    )
    assert all_layers.activation_mode == "all-on"
    assert [all_layers.uses_activation(idx) for idx in range(3)] == [True, True, True]

    no_layers = DeepResidualGAT(
        in_channels=3,
        layer_widths=[4, 4, 4],
        num_classes=2,
        dropouts=[0.0, 0.0, 0.0],
        heads=2,
        attention_dropout=0.0,
        activation_mode="none",
    )
    assert no_layers.activation_mode == "all-off"
    assert [no_layers.uses_activation(idx) for idx in range(3)] == [False, False, False]


def test_deep_residual_gat_defaults_to_first_on_activation():
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")

    from tools.run_cora_gat_c3e_candidates import DeepResidualGAT

    model = DeepResidualGAT(
        in_channels=3,
        layer_widths=[4, 4],
        num_classes=2,
        dropouts=[0.0, 0.0],
        heads=2,
        attention_dropout=0.0,
    )

    assert model.activation_mode == "first-on"
    assert [model.uses_activation(idx) for idx in range(2)] == [True, False]


def test_deep_residual_gat_activation_kinds_construct_expected_modules():
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")

    from tools.run_cora_gat_c3e_candidates import DeepResidualGAT, build_activation

    assert isinstance(build_activation("prelu", 4), torch.nn.PReLU)
    assert isinstance(build_activation("silu", 4), torch.nn.SiLU)
    assert isinstance(build_activation("gelu", 4), torch.nn.GELU)

    expected = {
        "prelu": torch.nn.PReLU,
        "silu": torch.nn.SiLU,
        "gelu": torch.nn.GELU,
    }
    for kind, module_type in expected.items():
        model = DeepResidualGAT(
            in_channels=3,
            layer_widths=[4, 4],
            num_classes=2,
            dropouts=[0.0, 0.0],
            heads=2,
            attention_dropout=0.0,
            activation_kind=kind,
        )
        assert all(isinstance(module, module_type) for module in model.activations)


def test_deep_residual_gat_rejects_invalid_activation_mode():
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")

    from tools.run_cora_gat_c3e_candidates import DeepResidualGAT

    with pytest.raises(ValueError, match="activation_mode"):
        DeepResidualGAT(
            in_channels=3,
            layer_widths=[4, 4],
            num_classes=2,
            dropouts=[0.0, 0.0],
            heads=2,
            attention_dropout=0.0,
            activation_mode="middle",
        )


def test_deep_residual_gat_rejects_invalid_activation_kind():
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")

    from tools.run_cora_gat_c3e_candidates import DeepResidualGAT

    with pytest.raises(ValueError, match="activation_kind"):
        DeepResidualGAT(
            in_channels=3,
            layer_widths=[4, 4],
            num_classes=2,
            dropouts=[0.0, 0.0],
            heads=2,
            attention_dropout=0.0,
            activation_kind="relu",
        )


def test_gat_candidate_cli_rejects_invalid_activation_mode(monkeypatch):
    from tools.run_cora_gat_c3e_candidates import parse_args

    monkeypatch.setattr(
        "sys.argv",
        ["run_cora_gat_c3e_candidates.py", "--activation-mode", "middle"],
    )

    with pytest.raises(SystemExit) as exc_info:
        parse_args()

    assert exc_info.value.code == 2


def test_gat_candidate_cli_rejects_invalid_activation_kind(monkeypatch):
    from tools.run_cora_gat_c3e_candidates import parse_args

    monkeypatch.setattr(
        "sys.argv",
        ["run_cora_gat_c3e_candidates.py", "--activation-kind", "relu"],
    )

    with pytest.raises(SystemExit) as exc_info:
        parse_args()

    assert exc_info.value.code == 2


def test_deep_residual_gat_tiny_forward_shape():
    torch = pytest.importorskip("torch")
    Data = pytest.importorskip("torch_geometric.data").Data

    from tools.run_cora_gat_c3e_candidates import DeepResidualGAT

    model = DeepResidualGAT(
        in_channels=3,
        layer_widths=[4, 4],
        num_classes=2,
        dropouts=[0.0, 0.0],
        heads=2,
        attention_dropout=0.0,
        activation_mode="first-only",
    )
    data = Data(
        x=torch.randn(3, 3),
        edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long),
    )

    out = model(data)

    assert tuple(out.shape) == (3, 2)


@pytest.mark.parametrize("activation_kind", ["silu", "gelu"])
def test_deep_residual_gat_tiny_forward_shape_for_new_activation_kinds(activation_kind):
    torch = pytest.importorskip("torch")
    Data = pytest.importorskip("torch_geometric.data").Data

    from tools.run_cora_gat_c3e_candidates import DeepResidualGAT

    model = DeepResidualGAT(
        in_channels=3,
        layer_widths=[4, 4],
        num_classes=2,
        dropouts=[0.0, 0.0],
        heads=2,
        attention_dropout=0.0,
        activation_mode="first-only",
        activation_kind=activation_kind,
    )
    data = Data(
        x=torch.randn(3, 3),
        edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long),
    )

    out = model(data)

    assert tuple(out.shape) == (3, 2)


def test_gat_candidate_training_records_activation_kind(tmp_path):
    torch = pytest.importorskip("torch")
    Data = pytest.importorskip("torch_geometric.data").Data

    from tools.run_cora_gat_c3e_candidates import Candidate, train_candidate

    data = Data(
        x=torch.randn(3, 3),
        y=torch.tensor([0, 1, 0], dtype=torch.long),
        edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long),
    )
    data.train_mask = torch.tensor([True, False, False])
    data.val_mask = torch.tensor([False, True, False])
    data.test_mask = torch.tensor([False, False, True])
    dataset = SimpleNamespace(num_classes=2)
    args = SimpleNamespace(
        seed=7,
        heads=2,
        attention_dropout=0.0,
        activation_mode="first-only",
        activation_kind="silu",
        lr=0.005,
        weight_decay=0.0,
        epochs=1,
        patience=1,
    )
    candidate = Candidate(
        candidate_id=1,
        depth=1,
        original_widths=[4],
        snapped_widths=[4],
        dropouts=[0.0],
        phi0=1.0,
        channel_capacity=1.0,
    )

    row = train_candidate(
        candidate=candidate,
        args=args,
        data=data,
        dataset=dataset,
        device="cpu",
        save_root=tmp_path,
    )
    checkpoint = torch.load(row["checkpoint"], map_location="cpu", weights_only=False)

    assert row["activation_mode"] == "first-on"
    assert row["activation_kind"] == "silu"
    assert checkpoint["activation_mode"] == "first-on"
    assert checkpoint["activation_kind"] == "silu"


def test_gat_activation_grid_expands_to_nine_activation_combinations():
    from tools.run_cora_gat_activation_grid import activation_combinations

    combos = activation_combinations()

    assert len(combos) == 9
    assert ("first-on", "prelu") in combos
    assert ("all-on", "silu") in combos
    assert ("all-off", "gelu") in combos


def test_gat_activation_grid_deduplicates_alias_modes():
    from tools.run_cora_gat_activation_grid import activation_combinations

    combos = activation_combinations(["first-on", "first-only", "all"], ["prelu"])

    assert combos == [("first-on", "prelu"), ("all-on", "prelu")]
