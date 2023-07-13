import pytest
import torch
from sklearn.datasets import make_moons
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchutils.trainer import NNtrainer


@pytest.fixture()
def moons():
    X, y = make_moons(n_samples=500, noise=0.05)

    X = torch.from_numpy(X).to(torch.float32)
    y = torch.from_numpy(y).to(torch.int64)

    dataset = TensorDataset(X, y)

    train, val = random_split(dataset, lengths=[0.8, 0.2])
    trainloader, valloader = DataLoader(train, batch_size=32), DataLoader(val)

    return trainloader, valloader


@pytest.mark.trainer()
@pytest.mark.parametrize("cycle", [1, 5])
def test_training_cycles(moons, cycle):
    trainloader, valloader = moons

    model = torch.nn.Sequential(
        torch.nn.Linear(2, 30),
        torch.nn.Tanh(),
        torch.nn.Linear(30, 30),
        torch.nn.Tanh(),
        torch.nn.Linear(30, 2),
    )
    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    trainer = NNtrainer(model=model, optimizer=optimizer, loss=loss)

    for _ in range(cycle):
        trainer.train(
            trainloader=trainloader,
            valloader=valloader,
            epoch=20,
            metrics=["accuracy"],
            record_loss=True,
            checkpoint_file="tests/save/train",
            validate_every_x_epoch=1,
        )

    trainer.plot_train_validation_metric_curve("accuracy")

    assert True


@pytest.mark.trainer()
@pytest.mark.parametrize(
    "cycle",
    argvalues=[
        1,
        6,
    ],
)
def test_training_from_checkpoint(moons, cycle):
    trainloader, valloader = moons

    model = torch.nn.Sequential(
        torch.nn.Linear(2, 30),
        torch.nn.Tanh(),
        torch.nn.Linear(30, 30),
        torch.nn.Tanh(),
        torch.nn.Linear(30, 2),
    )

    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    trainer = NNtrainer(model=model, optimizer=optimizer, loss=loss)

    for _ in range(cycle):
        trainer._regiser_buffer.clear()
        trainer.train_from_checkpoint(
            check_point="tests/save/train_checkpoint.testloader",
            trainloader=trainloader,
            valloader=valloader,
            epoch=25,
            metrics=["accuracy"],
            record_loss=True,
            checkpoint_file=f"tests/save/train{cycle}",
            validate_every_x_epoch=1,
        )

    # Tests
    assert trainer.register_buffer.size < 6

    trainer.plot_train_validation_metric_curve("accuracy")

    assert True
