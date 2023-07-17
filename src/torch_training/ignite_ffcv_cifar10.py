from datetime import datetime
from typing import Any, Tuple

import ffcv.transforms as fftrans
import torch
from ffcv.fields import IntField, RGBImageField
from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder
from ffcv.loader import Loader, OrderOption
from ffcv.writer import DatasetWriter
from ignite.contrib.handlers import TensorboardLogger, global_step_from_engine
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint
from ignite.metrics import Accuracy, Loss, RunningAverage
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.models import resnet18

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log_interval = 100


# network
model = resnet18(num_classes=10).to(device)

# dataset


def write_dataset(name: str, dataset: Dataset) -> None:
    writer = DatasetWriter(
        fname=name,
        fields={
            "image": RGBImageField(
                max_resolution=256,
            ),
            "label": IntField(),
        },
    )

    # Write dataset
    writer.from_indexed_dataset(dataset=dataset)


train_dataset = CIFAR10(download=True, root=".", train=True)
test_dataset = CIFAR10(download=True, root=".", train=False)

write_dataset(name="tr.beton", dataset=train_dataset)
write_dataset(name="te.beton", dataset=test_dataset)

pipelines = {
    "image": [
        SimpleRGBImageDecoder(),
        fftrans.ToTensor(),
        fftrans.ToDevice(device, non_blocking=True),
        fftrans.ToTorchImage(),
        fftrans.Convert(torch.float32),
        transforms.Normalize(
            mean=(125.307, 122.961, 113.8575), std=(51.5865, 50.847, 51.255)
        ),
    ],
    "label": [
        IntDecoder(),
        fftrans.ToTensor(),
        fftrans.ToDevice(device, non_blocking=True),
        fftrans.Squeeze(),
    ],
}

train_loader = Loader(
    fname="tr.beton",
    batch_size=128,
    num_workers=8,
    order=OrderOption.RANDOM,
    pipelines=pipelines,
)
val_loader = Loader(
    fname="te.beton",
    batch_size=256,
    num_workers=8,
    order=OrderOption.SEQUENTIAL,
    pipelines=pipelines,
)

# optimizer
optimizer = torch.optim.RMSprop(model.parameters(), lr=0.005)

# trainer


def train_step(engine: Engine, batch: Tuple) -> dict:
    model.train()
    optimizer.zero_grad()
    x, y = batch[0], batch[1]
    y_pred = model(x)
    loss = nn.functional.cross_entropy(y_pred, y)
    loss.backward()
    optimizer.step()
    loss_val: float = loss.item()
    return {"loss": loss_val}


trainer = Engine(train_step)
RunningAverage(output_transform=lambda x: x["loss"]).attach(trainer, "loss")

# pbar
pbar = ProgressBar(persist=True)
pbar.attach(trainer, ["loss"])


# evaluator
def validation_step(engine: Engine, batch: tuple) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    with torch.no_grad():
        x, y = batch[0].to(device), batch[1].to(device)
        y_pred = model(x)
        return y_pred, y


val_metrics = {"accuracy": Accuracy(), "loss": Loss(nn.CrossEntropyLoss())}

evaluator = Engine(validation_step)
for name, metric in val_metrics.items():
    metric.attach(evaluator, name)


# checkpoint
def score_function(engine: Engine) -> Any:
    return engine.state.metrics["accuracy"]


# logger
@trainer.on(Events.EPOCH_COMPLETED)
def log_validation_results(trainer: Engine) -> None:
    evaluator.run(val_loader)
    metrics = evaluator.state.metrics
    print(
        f"Validation Results - Epoch[{trainer.state.epoch}] Avg accuracy: {metrics['accuracy']:.2f} Avg loss: {metrics['loss']:.2f}"
    )


dt_string = datetime.now().strftime("%Y-%m-%d--%H:%M:%S")
save_dir = f"tb-logger/cifar10_ffcv/{dt_string}"
tb_logger = TensorboardLogger(log_dir=save_dir)

tb_logger.attach_output_handler(
    trainer,
    event_name=Events.EPOCH_COMPLETED,
    tag="training",
    metric_names="all",
)


tb_logger.attach_output_handler(
    evaluator,
    event_name=Events.EPOCH_COMPLETED,
    tag="validation",
    metric_names="all",
    global_step_transform=global_step_from_engine(trainer),
)


trainer.run(train_loader, max_epochs=5)
