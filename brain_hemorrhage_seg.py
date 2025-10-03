"""
Medical Imaging Segmentation Training Script
3D U-Net for Brain CT Hemorrhage Segmentation
"""

import os
import glob
import argparse
import torch
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from monai.data import Dataset
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd,
    ResizeWithPadOrCropd, ScaleIntensityRanged, AsDiscreted,
    EnsureTyped, MapTransform
)
from monai.networks.nets import UNet
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric


# ============================================================================
# Custom Transforms
# ============================================================================

class BrainWindowd(MapTransform):
    """Apply brain window transformation to CT images."""
    
    def __init__(self, keys, window_center=40, window_width=120):
        super().__init__(keys)
        self.center = window_center
        self.width = window_width

    def __call__(self, data):
        d = dict(data)
        low = self.center - self.width / 2
        high = self.center + self.width / 2
        for key in self.keys:
            img = d[key].astype(np.float32)
            img = (img - low) * (255.0 / (high - low))
            img = np.clip(img, 0, 255)
            d[key] = img
        return d

# ============================================================================
# Data Preparation
# ============================================================================

def prepare_data(data_dir, positive_only=True, test_size=25, random_state=42):
    """Prepare train/test splits with optional positive-only filtering."""
    
    ct_paths = sorted(glob.glob(os.path.join(data_dir, "ct_scans", "*.nii")))
    mask_paths = sorted(glob.glob(os.path.join(data_dir, "masks", "*.nii")))
    
    all_files = []
    labels = []
    
    for ct, mask in zip(ct_paths, mask_paths):
        mask_data = nib.load(mask).get_fdata()
        label = int(mask_data.sum() > 0)
        all_files.append({"image": ct, "label": mask})
        labels.append(label)
    
    labels = np.array(labels)
    print(f"Total: {len(all_files)} | Positive: {labels.sum()} | Negative: {len(labels)-labels.sum()}")
    
    # Stratified split
    train_files, test_files, train_labels, test_labels = train_test_split(
        all_files, labels, test_size=test_size, random_state=random_state, stratify=labels
    )
    
    # Filter positive only if requested
    if positive_only:
        train_files = [f for (i, f) in enumerate(train_files) if train_labels[i] != 0]
        print(f"Using positive cases only: {len(train_files)} train samples")
    
    print(f"Train: {len(train_files)} | Test: {len(test_files)}")
    print(f"Test distribution - Pos: {np.sum(test_labels)}, Neg: {len(test_labels)-np.sum(test_labels)}")
    
    return train_files, test_files


def get_transforms():
    """Get train and validation transforms."""
    
    base_transforms = [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Spacingd(keys=["image", "label"], pixdim=(0.5, 0.5, 5.0), mode=("bilinear", "nearest")),
        ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=(512, 512, 64)),
        BrainWindowd(keys=["image"], window_center=40, window_width=120),
        ScaleIntensityRanged(keys=["image"], a_min=0, a_max=255, b_min=0.0, b_max=1.0, clip=True),
        AsDiscreted(keys=["label"], threshold=0.5),
        EnsureTyped(keys=["image", "label"]),
    ]
    
    train_transforms = Compose(base_transforms)
    val_transforms = Compose(base_transforms)
    
    return train_transforms, val_transforms


# ============================================================================
# Model and Training Setup
# ============================================================================

def create_model(device):
    """Create 3D U-Net model."""
    
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)
    
    return model


def train_epoch(model, loader, optimizer, loss_fn, device):
    """Train for one epoch."""
    
    model.train()
    epoch_losses = []
    
    pbar = tqdm(loader, desc="Training", leave=False)
    for batch in pbar:
        inputs = batch["image"].to(device)
        labels = batch["label"].to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        
        epoch_losses.append(loss.item())
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    return float(np.mean(epoch_losses))


def validate(model, loader, loss_fn, dice_metric, device, threshold=0.25):
    """Validate model."""
    
    model.eval()
    val_losses = []
    
    with torch.no_grad():
        for batch in loader:
            inputs = batch["image"].to(device)
            labels = batch["label"].to(device)
            
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            val_losses.append(loss.item())
            
            # Compute Dice
            probs = torch.sigmoid(outputs)
            preds = (probs > threshold).float()
            dice_metric(y_pred=preds, y=labels)
    
    avg_val_loss = float(np.mean(val_losses))
    mean_dice = dice_metric.aggregate().item()
    dice_metric.reset()
    
    return avg_val_loss, mean_dice


# ============================================================================
# Main Training Loop
# ============================================================================

def train_model(
    train_files,
    test_files,
    data_dir,
    checkpoint_path="model_checkpoint.pth",
    max_epochs=300,
    batch_size=1,
    lr=1e-4,
    patience=20,
    resume=True,
):
    """Main training function."""
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Transforms
    train_transforms, val_transforms = get_transforms()
    
    # Split train into train/val
    train_split, val_split = train_test_split(train_files, test_size=0.2, random_state=42)
    
    # Datasets
    train_ds = Dataset(data=train_split, transform=train_transforms)
    val_ds = Dataset(data=val_split, transform=val_transforms)
    test_ds = Dataset(data=test_files, transform=val_transforms)
    
    # Loaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=2)
    
    # Model, loss, optimizer
    model = create_model(device)
    loss_fn = DiceCELoss(sigmoid=True, to_onehot_y=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10, verbose=True)
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    
    # Resume from checkpoint
    history = {"train": [], "val": []}
    start_epoch = 0
    best_val_loss = float("inf")
    
    if resume and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        if isinstance(checkpoint, dict) and "model_state" in checkpoint:
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            best_val_loss = checkpoint["best_loss"]
            history = checkpoint["history"]
            start_epoch = checkpoint["epoch"] + 1
            print(f"Resumed from epoch {start_epoch} with best_loss={best_val_loss:.4f}")
        else:
            model.load_state_dict(checkpoint)
            print("Loaded old checkpoint format (model weights only)")
    
    # Training loop
    early_stop_counter = 0
    
    for epoch in range(start_epoch, max_epochs):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        history["train"].append(train_loss)
        
        # Validate
        val_loss, val_dice = validate(model, val_loader, loss_fn, dice_metric, device)
        history["val"].append(val_loss)
        
        print(f"[Epoch {epoch+1}/{max_epochs}] "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f}")
        
        # Scheduler step
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            
            checkpoint = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_loss": best_val_loss,
                "history": history,
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"  >> Saved new best model (Val Loss={best_val_loss:.4f})")
        else:
            early_stop_counter += 1
            print(f"  >> No improvement. Early stop counter = {early_stop_counter}/{patience}")
        
        # Early stopping
        if early_stop_counter >= patience:
            print(">> Early stopping triggered!")
            break
    
    return model, history


# ============================================================================
# Argument Parser
# ============================================================================

def parse_args():
    """Parse command line arguments."""
    
    parser = argparse.ArgumentParser(
        description="Train 3D U-Net for Brain CT Hemorrhage Segmentation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data arguments
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/kaggle/input/cerebral-bleed-ct/ct-images-intracranial-hemorrhage",
        help="Path to data directory containing ct_scans and masks folders"
    )
    parser.add_argument(
        "--positive-only",
        action="store_true",
        default=True,
        help="Use only positive (hemorrhage) cases for training"
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=25,
        help="Number of samples to reserve for test set"
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="Fraction of training data to use for validation"
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    # Model arguments
    parser.add_argument(
        "--channels",
        type=int,
        nargs="+",
        default=[16, 32, 64, 128, 256],
        help="Channel dimensions for U-Net encoder/decoder"
    )
    parser.add_argument(
        "--num-res-units",
        type=int,
        default=2,
        help="Number of residual units in U-Net"
    )
    
    # Training arguments
    parser.add_argument(
        "--epochs",
        type=int,
        default=300,
        help="Maximum number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for training"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Early stopping patience (epochs without improvement)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.25,
        help="Probability threshold for binary segmentation"
    )
    
    # Checkpoint arguments
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="hemorrhage_seg_model.pth",
        help="Path to save/load model checkpoint"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Resume training from checkpoint if it exists"
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Start training from scratch (don't load checkpoint)"
    )
    
    # Output arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs",
        help="Directory to save outputs (plots, logs, etc.)"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="Number of workers for data loading"
    )
    
    # Preprocessing arguments
    parser.add_argument(
        "--window-center",
        type=int,
        default=40,
        help="Brain window center for CT preprocessing"
    )
    parser.add_argument(
        "--window-width",
        type=int,
        default=120,
        help="Brain window width for CT preprocessing"
    )
    parser.add_argument(
        "--spatial-size",
        type=int,
        nargs=3,
        default=[512, 512, 64],
        help="Target spatial size (H W D) for resizing"
    )
    parser.add_argument(
        "--pixdim",
        type=float,
        nargs=3,
        default=[0.5, 0.5, 5.0],
        help="Target voxel spacing (x y z) for resampling"
    )
    
    return parser.parse_args()


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Print configuration
    print("=" * 80)
    print("Training Configuration:")
    print("=" * 80)
    for arg, value in vars(args).items():
        print(f"{arg:20s}: {value}")
    print("=" * 80)
    
    # Prepare data
    train_files, test_files = prepare_data(
        args.data_dir,
        positive_only=args.positive_only,
        test_size=args.test_size,
        random_state=args.random_seed
    )
    
    # Train model
    model, history = train_model(
        train_files=train_files,
        test_files=test_files,
        data_dir=args.data_dir,
        checkpoint_path=args.checkpoint,
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        resume=args.resume,
    )
    
    # Plot training curves
    plot_path = os.path.join(args.output_dir, "training_curve.png")
    epochs = range(1, len(history["train"]) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history["train"], marker="o", label="Train Loss")
    plt.plot(epochs, history["val"], marker="s", label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(plot_path)
    print(f"Training curve saved to {plot_path}")
    
    print("\nTraining completed successfully!")