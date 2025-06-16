from torch.utils.data import Dataset
import torch
from torch.utils.data import DataLoader
from typing import Union, List, Sequence
from pytorch_lightning import LightningDataModule
from torch.utils.data import random_split


class ObservationDataset(Dataset):
    def __init__(self, observations: List[torch.Tensor]):
        assert all(
            observations[0].size(0) == observation.size(0)
            for observation in observations
        ), "Size mismatch between observations"
        self.observations = observations

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        return self.observations[idx], 0.0, f"observation_{idx}.png"


class ObservationDataModule(LightningDataModule):
    """
    PyTorch Lightning data module for observation datasets.
    Used for compatibility with pytorch_vae VAEDataset.
    """

    def __init__(
        self,
        train_dataset_torch: ObservationDataset,
        test_dataset_torch: ObservationDataset,
        train_batch_size: int = 8,
        val_batch_size: int = 8,
        test_batch_size: int = 8,
        patch_size: Union[int, Sequence[int]] = (256, 256),
        num_workers: int = 0,
        pin_memory: bool = False,
        train_dataset_name="coinrun",
        test_dataset_name=None,
        use_difficulty_sampling=False,
    ):
        super().__init__()
        # Not a fan of a lot of the naming here, but keeping it for compatibility.

        # Split into train and val datasets.
        train_size = int(len(train_dataset_torch) * 0.9)
        val_size = len(train_dataset_torch) - train_size
        self.train_dataset, self.val_dataset = random_split(
            train_dataset_torch, [train_size, val_size]
        )

        self.test_dataset = test_dataset_torch
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.train_dataset_name = train_dataset_name
        self.test_dataset_name = test_dataset_name
        self.use_difficulty_sampling = use_difficulty_sampling
        self.sampled_img_names = []
        self.sampled_img_losses = []
        self.difficulty_sampler = None

    def train_dataloader(self) -> DataLoader:
        if self.difficulty_sampler is not None:
            return DataLoader(
                self.train_dataset,
                batch_size=self.train_batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                sampler=self.difficulty_sampler,
            )
        else:
            return DataLoader(
                self.train_dataset,
                batch_size=self.train_batch_size,
                num_workers=self.num_workers,
                shuffle=True,
                pin_memory=self.pin_memory,
            )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            num_workers=self.num_workers,
            shuffle=False,  # Don't shuffle for test to get consistent results
            pin_memory=self.pin_memory,
        )

    def record_img_losses(self, img_names, losses):
        if isinstance(img_names, torch.Tensor):
            img_names = img_names.cpu().tolist()
        if isinstance(losses, torch.Tensor):
            losses = losses.cpu().tolist()
        self.sampled_img_names.extend(img_names)
        self.sampled_img_losses.extend(losses)

    def on_epoch_end(self):
        if self.difficulty_sampler is not None:
            self.difficulty_sampler.update_img_difficulties(
                self.sampled_img_names, self.sampled_img_losses
            )
        self.sampled_img_names = []
        self.sampled_img_losses = []
