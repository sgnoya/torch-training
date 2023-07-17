from datetime import datetime
from typing import Tuple

import torch
from argparse_dataclass import ArgumentParser, dataclass
from ignite.contrib.handlers import TensorboardLogger, global_step_from_engine
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.engine import Engine, Events
from ignite.metrics import Accuracy, Loss, RunningAverage
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.models import resnet18
from torchvision.transforms import Compose, Normalize

from torch_training import Params

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class IgniteCifar10Trainer:
    def __init__(self) -> None:
        parser = ArgumentParser(Params)
        self.params = parser.parse_args()

        # data loader
        train_loader, val_loader = self.make_loader()

        # network
        model = resnet18(num_classes=self.params.num_classes).to(device)

        # optimizer
        optimizer = torch.optim.RMSprop(model.parameters(), lr=self.params.lr)

        # trainer
        def train_step(engine: Engine, batch: Tuple) -> dict:
            model.train()
            optimizer.zero_grad()
            x, y = batch[0].to(device), batch[1].to(device)
            y_pred = model(x)
            loss: torch.Tensor = nn.functional.cross_entropy(y_pred, y)
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
        def validation_step(
            engine: Engine, batch: tuple
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            model.eval()
            with torch.inference_mode():
                x, y = batch[0].to(device), batch[1].to(device)
                y_pred = model(x)
                return y_pred, y

        val_metrics = {"accuracy": Accuracy(), "loss": Loss(nn.CrossEntropyLoss())}
        evaluator = Engine(validation_step)
        for name, metric in val_metrics.items():
            metric.attach(evaluator, name)

        # logger
        @trainer.on(Events.EPOCH_COMPLETED)
        def log_validation_results(trainer: Engine) -> None:
            evaluator.run(val_loader)
            metrics = evaluator.state.metrics
            print(
                f"Validation Results - Epoch[{trainer.state.epoch}] Avg accuracy: {metrics['accuracy']:.2f} Avg loss: {metrics['loss']:.2f}"
            )

        save_dir = self.get_logdir_path()
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

        self.trainer = trainer
        self.train_loader = train_loader
        self.val_loader = val_loader

    def get_logdir_path(self) -> str:
        dt_string = datetime.now().strftime("%Y-%m-%d--%H:%M:%S")
        return f"tb-logger/cifar10/{dt_string}"

    def make_loader(self) -> Tuple[DataLoader, DataLoader]:
        # dataset
        data_transform = Compose(
            [
                transforms.ToTensor(),
                Normalize(
                    (125.307 / 255.0, 122.961 / 255, 113.8575 / 255),
                    std=(51.5865 / 255, 50.847 / 255, 51.255 / 255),
                ),
            ]
        )

        train_dataset = CIFAR10(
            download=True, root=".", train=True, transform=data_transform
        )
        test_dataset = CIFAR10(
            download=True, root=".", train=False, transform=data_transform
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=128,
            num_workers=8,
            shuffle=True,
        )
        val_loader = DataLoader(
            test_dataset,
            batch_size=256,
            num_workers=8,
            shuffle=False,
        )

        return train_loader, val_loader

    def run(self) -> None:
        self.trainer.run(self.train_loader, max_epochs=self.params.max_epochs)


if __name__ == "__main__":
    trainer = IgniteCifar10Trainer()
    trainer.run()
