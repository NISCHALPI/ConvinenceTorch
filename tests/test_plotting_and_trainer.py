import seaborn as sns
import torch
from torchutils.skeletons import Register
from sklearn.datasets import make_moons
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchutils.trainer import NNtrainer


def test_register_init():
    metrics = ["accuracy", "precision"]
    loss = torch.nn.CrossEntropyLoss()
    epoch = 10
    cycle = 3
    multiclass_reduction_strategy = "macro"

    register = Register(
        metrics=metrics,
        loss=loss,
        epoch=epoch,
        cycle=cycle,
        multiclass_reduction_strategy=multiclass_reduction_strategy,
    )

    assert register._metrics == metrics + [loss._get_name()]
    assert register._loss == loss
    assert register._epoch == epoch
    assert register.cycle == cycle
    assert register.multiclass_strategy == multiclass_reduction_strategy
    assert "accuracy" in register.available_metric
    assert "precision" in register.available_metric


def test_eval_metric():
    metrics = ["accuracy", "precision"]
    loss = torch.nn.CrossEntropyLoss()
    epoch = 10
    register = Register(metrics=metrics, loss=loss, epoch=epoch)

    y_pred = torch.rand(10, 2)
    y_true = torch.randint(0, 2, size=(10, 1))
    metric_name = "accuracy"

    metric_value = register._eval_metric(y_pred, y_true, metric_name)

    print(register.records)
    assert isinstance(metric_value, float)


def test_processing_and_plotting_on_moons():
    X, y = make_moons(n_samples=1000, noise=0.05)

    X = torch.from_numpy(X).to(torch.float32)
    y = torch.from_numpy(y).to(torch.int64)

    dataset = TensorDataset(X, y)

    train, val = random_split(dataset, lengths=[0.8, 0.2])
    trainloader, valloader = DataLoader(train), DataLoader(val)

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
    trainer.train(
        trainloader=trainloader,
        valloader=valloader,
        epoch=20,
        metrics=["accuracy"],
        record_loss=True,
        checkpoint_file="train",
        validate_every_x_epoch=1,
    )
    trainer.plot_train_validation_metric_curve("accuracy")
   
    assert True


def test_all():
    test_register_init()
    test_eval_metric()
    test_processing_and_plotting_on_moons()

if __name__ =='__main__':
    test_all()