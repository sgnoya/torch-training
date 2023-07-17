from typing import Any, Tuple

import torch
from ignite.contrib.handlers import TensorboardLogger, global_step_from_engine
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint
from ignite.metrics import Accuracy, Loss, RunningAverage
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.models import resnet18
from torchvision.transforms import Compose, Normalize, ToTensor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log_interval = 100


# network
class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()

        # Changed the output layer to output 10 classes instead of 1000 classes
        self.model = resnet18(num_classes=10)

        # Changed the input layer to take grayscale images for MNIST instaed of RGB images
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


model = Net().to(device)

# dataset
data_transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
train_loader = DataLoader(
    MNIST(download=True, root=".", transform=data_transform, train=True),
    batch_size=128,
    shuffle=True,
)
val_loader = DataLoader(
    MNIST(download=True, root=".", transform=data_transform, train=False),
    batch_size=256,
    shuffle=False,
)

# optimizer
optimizer = torch.optim.RMSprop(model.parameters(), lr=0.005)

# trainer


def train_step(engine: Engine, batch: Tuple) -> dict:
    model.train()
    optimizer.zero_grad()
    x, y = batch[0].to(device), batch[1].to(device)
    y_pred = model(x)
    loss = nn.functional.cross_entropy(y_pred, y)
    loss.backward()
    optimizer.step()
    loss_val: float = loss.item()
    return {"loss": loss_val}


trainer = Engine(train_step)
# pbar
pbar = ProgressBar(persist=True)
RunningAverage(output_transform=lambda x: x["loss"]).attach(trainer, "loss")

pbar.attach(trainer, ["loss"])


@trainer.on(Events.EPOCH_COMPLETED)
def log_training_results(trainer: Engine) -> None:
    train_evaluator.run(train_loader)
    metrics = train_evaluator.state.metrics
    print(
        f"Training Results - Epoch[{trainer.state.epoch}] Avg accuracy: {metrics['accuracy']:.2f} Avg loss: {metrics['loss']:.2f}"
    )


@trainer.on(Events.EPOCH_COMPLETED)
def log_validation_results(trainer: Engine) -> None:
    val_evaluator.run(val_loader)
    metrics = val_evaluator.state.metrics
    print(
        f"Validation Results - Epoch[{trainer.state.epoch}] Avg accuracy: {metrics['accuracy']:.2f} Avg loss: {metrics['loss']:.2f}"
    )


# evaluator
def validation_step(engine: Engine, batch: tuple) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    with torch.no_grad():
        x, y = batch[0].to(device), batch[1].to(device)
        y_pred = model(x)
        return y_pred, y


val_metrics = {"accuracy": Accuracy(), "loss": Loss(nn.CrossEntropyLoss())}

train_evaluator = Engine(validation_step)
val_evaluator = Engine(validation_step)
for name, metric in val_metrics.items():
    metric.attach(train_evaluator, name)

for name, metric in val_metrics.items():
    metric.attach(val_evaluator, name)


# checkpoint
def score_function(engine: Engine) -> Any:
    return engine.state.metrics["accuracy"]


model_checkpoint = ModelCheckpoint(
    "checkpoint",
    n_saved=2,
    filename_prefix="best",
    score_function=score_function,
    score_name="accuracy",
    global_step_transform=global_step_from_engine(trainer),
)

# logger
tb_logger = TensorboardLogger(log_dir="tb-logger")

tb_logger.attach_output_handler(
    trainer,
    event_name=Events.ITERATION_COMPLETED(every=100),
    tag="training",
    output_transform=lambda loss: {"batch_loss": loss},
)

for tag, evaluator in [("training", train_evaluator), ("validation", val_evaluator)]:
    tb_logger.attach_output_handler(
        evaluator,
        event_name=Events.EPOCH_COMPLETED,
        tag=tag,
        metric_names="all",
        global_step_transform=global_step_from_engine(trainer),
    )


trainer.run(train_loader, max_epochs=5)
