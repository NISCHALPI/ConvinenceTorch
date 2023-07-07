import torch
from torch.utils.data import DataLoader
from src.trainer import NNtrainer  

# Create a mock model, optimizer, and loss function for testing
class MockModel(torch.nn.Module):
    def __init__(self):
        super(MockModel, self).__init__()
        self.linear = torch.nn.Linear(10, 1)

    def forward(self, x):
        return self.linear(x)

model = MockModel()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
loss_fn = torch.nn.MSELoss()

# Create mock data loaders for training and validation
train_data = torch.randn((100, 10))
train_labels = torch.randn((100, 1))
train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)

val_data = torch.randn((20, 10))
val_labels = torch.randn((20, 1))
val_dataset = torch.utils.data.TensorDataset(val_data, val_labels)
val_loader = DataLoader(val_dataset, batch_size=5, shuffle=False)

# Create an instance of NNtrainer
trainer = NNtrainer(model, optimizer, loss_fn)

# Test the train method
trainer.train(train_loader, val_loader, epoch=200, show_every_batch=2, early_stopping=True, eval_every_epoch=1, record_loss=True)

# Test the validate method
val_loss = trainer.validate(val_loader)
print("Validation Loss:", val_loss)

# Test the get_loss method
trainer.plot_train_validation_metric_curve()