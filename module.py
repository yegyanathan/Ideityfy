import math
import torch
import torchvision.transforms as T
from os import path
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn import CrossEntropyLoss
from torchmetrics.functional import accuracy
from timm import create_model, list_models
from timm.models.vision_transformer import VisionTransformer
from torchvision.datasets import ImageFolder

from utils import AverageMeter
from lightning import LightningDataModule, LightningModule
from huggingface_hub import PyTorchModelHubMixin, login
import torch.nn as nn
from lora import LoRA_qkv


PRE_SIZE = (256, 256)
IMG_SIZE = (224, 224)

STATS = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
DATASET_DIRECTORY = path.join(path.dirname(__file__), "datasets")
CHECKPOINT_DIRECTORY = path.join(path.dirname(__file__), "checkpoints")

TRANSFORMS = {
    "train": T.Compose([
        T.Resize(PRE_SIZE),
        T.RandomCrop(IMG_SIZE),
        T.ToTensor(),
        T.Normalize(**STATS)
    ]),
    "val": T.Compose([
        T.Resize(PRE_SIZE),
        T.CenterCrop(IMG_SIZE),
        T.ToTensor(),
        T.Normalize(**STATS)
    ])
}



class myDataModule(LightningDataModule):
    """
    Lightning DataModule for loading and preparing the image dataset.

    Args:
        ds_name (str): Name of the dataset directory.
        batch_size (int): Batch size for data loaders.
        num_workers (int): Number of workers for data loaders.
    """
    def __init__(self, ds_name: str = "deities", batch_size: int = 32, num_workers: int = 8):
        super(myDataModule, self).__init__()

        self.ds_path = path.join(DATASET_DIRECTORY, ds_name)
        assert path.exists(self.ds_path), f"Dataset {ds_name} not found in {DATASET_DIRECTORY}."

        self.ds_name = ds_name
        self.batch_size = batch_size
        self.num_workers = num_workers


    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_ds = ImageFolder(root=path.join(self.ds_path, 'train'), transform=TRANSFORMS['train'])
            self.val_ds = ImageFolder(root=path.join(self.ds_path, 'val'), transform=TRANSFORMS['val'])
            # Number of classes
            self.num_classes = len(self.train_ds.classes)          
    

    def train_dataloader(self) -> DataLoader:
        # Weighted Random sampler for imbalanced dataset
        class_samples = [0] * self.num_classes
        for _, (_, label) in enumerate(self.train_ds):
            class_samples[label] += 1
        weights = [1.0 / class_samples[label] for _, label in self.train_ds]
        self.sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
        return DataLoader(dataset=self.train_ds, batch_size=self.batch_size, 
                          sampler=self.sampler, num_workers=self.num_workers, persistent_workers=True)


    def val_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self.val_ds, batch_size=self.batch_size, 
                          shuffle=False, num_workers=self.num_workers, persistent_workers=True)
    



class myModule(LightningModule, PyTorchModelHubMixin):
    """
    Lightning Module for training and evaluating the Image classification model.

    Args:
        model_name (str): Name of the Vision Transformer model.
        num_classes (int): Number of classes in the dataset.
        freeze_flag (bool): Flag to freeze the base model parameters.
        use_lora (bool): Flag to use LoRA (Local Rank Adaptation) for fine-tuning.
        rank (int): Rank for LoRA if use_lora is True.
        learning_rate (float): Learning rate for the optimizer.
        weight_decay (float): Weight decay for the optimizer.
        push_to_hf (bool): Flag to push model to Huggingface Hub.
        commit_message (str): Commit message
        repo_id (str): Huggingface repo id
    """
    def __init__(self, 
                 model_name: str = "vit_tiny_patch16_224", 
                 num_classes: int = 25,
                 freeze_flag: bool = True,
                 use_lora: bool = False, 
                 rank: int = None, 
                 learning_rate: float = 3e-4, 
                 weight_decay: float = 2e-5,
                 push_to_hf: bool = True,
                 commit_message: str = "my model",
                 repo_id: str = "Yegiiii/ideityfy"
        ):
    
        super(myModule, self).__init__()
        self.save_hyperparameters()
        self.model_name = model_name
        self.num_classes = num_classes
        self.freeze_flag = freeze_flag
        self.rank = rank
        self.use_lora = use_lora
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.push_to_hf = push_to_hf
        self.commit_message = commit_message
        self.repo_id = repo_id
        
        assert model_name in list_models(), f"Timm model name {model_name} not available."
        timm_model = create_model(model_name, pretrained=True)
        assert isinstance(timm_model, VisionTransformer), f"{model_name} not a Vision Transformer."
        self.model = timm_model

        if freeze_flag:
            # Freeze the Timm model parameters
            self.freeze()

        if use_lora:
            # Add LoRA matrices to the Timm model
            assert freeze_flag, "Set freeze_flag to True for using LoRA fine-tuning."
            assert rank, "Rank can't be None."
            # self.model = LoRA_VisionTransformer(self.model, rank)
            self.add_lora()

        self.model.reset_classifier(num_classes)

        # Loss function
        self.criterion = CrossEntropyLoss()

        # Validation metrics
        self.top1_acc = AverageMeter()
        self.top3_acc = AverageMeter()
        self.top5_acc = AverageMeter()


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
    
    def on_fit_start(self) -> None:
        num_classes = self.trainer.datamodule.num_classes
        assert num_classes == self.num_classes, \
        f"Number of classes provided in the argument ({self.num_classes}) is not matching \
         the number of classes in the dataset ({num_classes})."


    def on_fit_end(self) -> None:
        if self.push_to_hf:
            login()
            self.push_to_hub(repo_id=self.repo_id, commit_message=self.commit_message)


    def configure_optimizers(self):
        optimizer = AdamW(params=filter(lambda param: param.requires_grad, self.model.parameters()), 
                          lr=self.learning_rate, weight_decay=self.weight_decay)
        
        scheduler = CosineAnnealingLR(optimizer, self.trainer.max_epochs, 1e-6)
        return ([optimizer], [scheduler])


    def shared_step(self, x: torch.Tensor, y: torch.Tensor):
        logits = self(x)
        loss = self.criterion(logits, y)   
        return logits, loss 


    def training_step(self, batch, batch_idx) -> torch.Tensor:
        x, y = batch
        _, loss = self.shared_step(x, y)

        self.log("train_loss", loss, prog_bar=True, logger=True, on_epoch=True)
        return loss


    def validation_step(self, batch, batch_idx) -> dict:
        x, y = batch
        logits, loss = self.shared_step(x, y)

        self.top1_acc(
            val=accuracy(logits, y, average="weighted", top_k=1, num_classes=self.num_classes))
        self.top3_acc(
            val=accuracy(logits, y, average="weighted", top_k=3, num_classes=self.num_classes))
        self.top5_acc(
            val=accuracy(logits, y, average="weighted", top_k=5, num_classes=self.num_classes))

        metric_dict = {
            "val_loss": loss, 
            "top1_acc": self.top1_acc.avg, 
            "top3_acc": self.top3_acc.avg, 
            "top5_acc": self.top5_acc.avg
        }
        
        self.log_dict(metric_dict, prog_bar=True, logger=True, on_epoch=True)
        return  metric_dict

    
    def on_validation_epoch_end(self) -> None:
        self.top1_acc.reset()
        self.top3_acc.reset()
        self.top5_acc.reset()


    def add_lora(self):
        self.w_As = []
        self.w_Bs = []

        for _, blk in enumerate(self.model.blocks):
            w_qkv_linear = blk.attn.qkv
            self.dim = w_qkv_linear.in_features
            lora_a_linear_q = nn.Linear(self.dim, self.rank, bias=False)
            lora_b_linear_q = nn.Linear(self.rank, self.dim, bias=False)
            lora_a_linear_v = nn.Linear(self.dim, self.rank, bias=False)
            lora_b_linear_v = nn.Linear(self.rank, self.dim, bias=False)
            self.w_As.append(lora_a_linear_q)
            self.w_Bs.append(lora_b_linear_q)
            self.w_As.append(lora_a_linear_v)
            self.w_Bs.append(lora_b_linear_v)
            blk.attn.qkv = LoRA_qkv(w_qkv_linear, lora_a_linear_q, 
                                    lora_b_linear_q, lora_a_linear_v, lora_b_linear_v)

        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)



if __name__ == "__main__":
    from torchinfo import summary
    
    module = myModule("vit_base_patch16_clip_224", rank=4, use_lora=True, freeze_flag=True)
    summary(module, (1, 3, 224, 224))

    # from datasets import load_dataset

    # dataset = load_dataset("Yegiiii/deities")
    # print(dataset)

