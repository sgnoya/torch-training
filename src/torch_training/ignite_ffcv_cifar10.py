from datetime import datetime
from typing import Tuple

import ffcv.transforms as fftrans
import torch
from ffcv.fields import IntField, RGBImageField
from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder
from ffcv.loader import Loader, OrderOption
from ffcv.writer import DatasetWriter
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10

from torch_training import IgniteCifar10Trainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


class IgniteCifar10FFCVTrainer(IgniteCifar10Trainer):
    def get_logdir_path(self) -> str:
        dt_string = datetime.now().strftime("%Y-%m-%d--%H:%M:%S")
        return f"tb-logger/cifar10_ffcv/{dt_string}"

    def make_loader(self) -> Tuple[Loader, Loader]:
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
        return train_loader, val_loader


if __name__ == "__main__":
    trainer = IgniteCifar10FFCVTrainer()
    trainer.run()
