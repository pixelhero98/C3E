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
