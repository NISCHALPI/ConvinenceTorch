import pandas as pd
import pytest
import torch
from torchutils.skeletons import Register


@pytest.fixture()
def register():
    loss_fn = torch.nn.CrossEntropyLoss()
    return Register(
        metrics=["accuracy", "precision", "recall", "f1"], loss=loss_fn, epoch=10
    )


@pytest.mark.register()
def test_init(register):
    assert len(register.records) == 3  # 3 keys: 'train', 'valid', 'time'
    with pytest.raises(RuntimeError):
        _ = register.minimized_record  # minimized_record is empty initially
    assert len(register.metrics) == 5  # 6 metrics including cross entropy loss


@pytest.mark.register()
def test_check_metric(register):
    assert register._check_metric("accuracy")  # 'accuracy' is available
    with pytest.raises(RuntimeError):
        register._check_metric("L2")  # L2 is not available


@pytest.mark.register()
def test_eval_metric(register):
    y_pred = torch.tensor([[1, 0, 1], [0, 1, 0]]).T.float()
    y_true = torch.tensor([0, 0, 1])

    assert register._eval_metric(y_pred, y_true, "accuracy") == pytest.approx(1 / 3)
    assert register._eval_metric(y_pred, y_true, "precision") == pytest.approx(1 / 3)
    assert register._eval_metric(y_pred, y_true, "recall") == pytest.approx(1 / 3)
    assert register._eval_metric(y_pred, y_true, "f1") == pytest.approx(1 / 3)


@pytest.mark.register()
def test_record_batch(register):
    y_pred = torch.tensor([[1, 0, 1], [0, 1, 0]]).T.float()
    y_true = torch.tensor([0, 0, 1])
    epoch = 1
    register._record_batch(y_pred, y_true, epoch, where=True)  # record for training set
    assert register.records["train"]["accuracy"]["Epoch_1"] == [pytest.approx(1 / 3)]


@pytest.mark.register()
def test_record_train_time_per_epoch(register):
    epoch = 1
    time = 10.0
    register._record_train_time_per_epoch(epoch, time)
    assert register.records["time"]["Epoch_1"] == 10.0


@pytest.mark.register()
def test_minimize_per_epoch(register):
    y_pred = torch.tensor([[1, 0, 1], [0, 1, 0]]).T.float()
    y_true = torch.tensor([0, 0, 1])
    epoch = 1
    register._record_batch(y_pred, y_true, epoch, where=True)  # record for training set
    register._record_batch(
        y_pred, y_true, epoch, where=False
    )  # record for validation set
    register._record_train_time_per_epoch(epoch, 10.0)
    register._minimize_per_epoch()
    assert "Epoch_1" in register.minimized_record["train"].index
    assert "Epoch_1" in register.minimized_record["valid"].index


@pytest.mark.register()
def test_records_property(register):
    assert register.records == register._records


@pytest.mark.register()
def test_minimized_record_property(register):
    register._records_per_epoch = {
        "train": pd.DataFrame(
            {"time": [0, 1, 2], "accuracy": [0.2, 0.5, 0.8]},
            index=["Epoch_1", "Epoch_2", "Epoch_3"],
        )
    }
    assert register.minimized_record == register._records_per_epoch


@pytest.mark.register()
def test_getitem(register):
    register._records_per_epoch = {
        "train": pd.DataFrame(
            {"time": [0, 1, 2], "accuracy": [0.2, 0.5, 0.8]},
            index=["Epoch_1", "Epoch_2", "Epoch_3"],
        )
    }
    assert isinstance(register["train"], pd.DataFrame)


@pytest.mark.register()
def test_repr(register):
    assert repr(register) == "Register(train, valid, multiclass_strategy: micro)"


@pytest.mark.register()
def test_multiclass_classification(register):
    y_pred = [0, 1, 2, 3, 4, 5, 6, 7, 6]
    y_true = [0, 1, 2, 3, 4, 5, 6, 7, 8]

    y_true = torch.tensor(y_true)
    y_pred = torch.tensor(pd.get_dummies(y_pred).values).float()

    accuracy = register._eval_metric(y_pred, y_true, "accuracy")
    assert accuracy == pytest.approx(8 / 9)
