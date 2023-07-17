from argparse_dataclass import dataclass


@dataclass
class Params:
    max_epochs: int = 5
    lr: float = 0.005
    num_classes: int = 10
