from os import path, makedirs
from rich.console import Console

from utils import getDevice, getPlatform, hasGPU, disableWarnings
from lightning.pytorch.cli import LightningCLI
from module import myModule, myDataModule
from huggingface_hub import login
from transformers import AutoModel


CHECKPOINT_DIRECTORY = path.join(path.dirname(__file__), "checkpoints")


def cli_main() -> None:
    disableWarnings()
    console = Console()
    nok_prefix = "[bold red]->[/bold red]"

    try:
        makedirs(CHECKPOINT_DIRECTORY, exist_ok=True)
    except OSError as e:
        print(f"Error creating directory: {e}")

    # Test GPU availability
    platform = getPlatform()
    if not hasGPU(platform):
        console.print(f"{nok_prefix}[bold red]GPU is not available. Please make sure your system has a compatible GPU.[/bold red]")
        console.print("You either need a PC with an Nv  dia GPU or a Mac with Apple M1/M2/M3 GPU.")
        exit(1)

    device = getDevice(platform)
    console.print(f"\nPlatform Detected: [bold green]{platform}[/bold green] with device [bold green]{device}[/bold green]")

    LightningCLI(myModule, myDataModule)


if __name__ == "__main__":
    # cli_main()

    from module import myModule, myDataModule

    dm = myDataModule("deities-v0")
    dm.setup()

    print(dm.train_ds.classes)