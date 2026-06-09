# train.py
import os
import argparse
import random
import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, f1_score

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

import sys

# sys.path.append("/kaggle/input/shuiling115/ConvNeXt")

from my_dataset import MyDataSet
from model_aug.model import densenet121 as create_model
from utils import read_split_data, create_lr_scheduler, get_params_groups, train_one_epoch, evaluate


# ----------------------- Grad-CAM Implementation -----------------------
class GradCAM:
    """Grad-CAM: visualize model decision regions"""
    def __init__(self, model, target_layer):
        self.model = model.eval()
        self.target_layer = target_layer
        self.feature_maps = None
        self.gradients = None

        # Register forward hook to save feature maps
        target_layer.register_forward_hook(self._save_feature_maps)
        # Register backward hook to save gradients (compatible with both old and new versions)
        try:
            target_layer.register_full_backward_hook(self._save_gradients)
        except AttributeError:
            target_layer.register_backward_hook(self._save_gradients_old)

    def _save_feature_maps(self, module, input, output):
        self.feature_maps = output.detach()

    def _save_gradients(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def _save_gradients_old(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate(self, input_image, target_class=None):
        """
        Generate Grad-CAM heatmap
        Args:
            input_image: tensor of shape (1, C, H, W), requires_grad=True
            target_class: target class index, if None use model prediction
        Returns:
            heatmap: numpy array of shape (H, W), values in [0,1]
        """
        self.model.zero_grad()
        output = self.model(input_image)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        one_hot = torch.zeros_like(output)
        one_hot[0][target_class] = 1
        output.backward(gradient=one_hot, retain_graph=False)

        gradients = self.gradients       # (1, C, H', W')
        feature_maps = self.feature_maps # (1, C, H', W')

        # Global average pooling of gradients -> channel weights
        weights = torch.mean(gradients, dim=[0, 2, 3])  # (C,)
        # Weighted sum of feature maps
        cam = torch.zeros(feature_maps.shape[2:], dtype=feature_maps.dtype, device=feature_maps.device)
        for i, w in enumerate(weights):
            cam += w * feature_maps[0, i, :, :]
        # ReLU and normalize
        cam = F.relu(cam)
        cam = cam / (cam.max() + 1e-8)
        return cam.cpu().numpy()
# --------------------------------------------------------------


def sample_data_by_ratio(images_path, images_label, ratio=0.1, seed=42):
    """
    Randomly sample data by ratio

    Args:
        images_path: list of image paths
        images_label: list of image labels
        ratio: sampling ratio (0.0-1.0)
        seed: random seed for reproducibility

    Returns:
        sampled image paths and labels
    """
    if ratio >= 1.0:
        return images_path, images_label

    random.seed(seed)

    combined = list(zip(images_path, images_label))
    random.shuffle(combined)

    total_samples = len(combined)
    sample_count = max(1, int(total_samples * ratio))

    sampled = combined[:sample_count]

    sampled_paths, sampled_labels = zip(*sampled)

    return list(sampled_paths), list(sampled_labels)


def seed_worker(worker_id):
    """
    Random seed initializer for DataLoader workers
    Note: This function must be global to be picklable on Windows
    """
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def set_seed(seed=42):
    """
    Set random seeds to ensure reproducibility

    Args:
        seed: random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU

    # Set CuDNN for determinism (may reduce performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"✓ Random seed set to: {seed}")
    print(f"  CuDNN deterministic: {torch.backends.cudnn.deterministic}")
    print(f"  CuDNN benchmark: {torch.backends.cudnn.benchmark}")


def plot_confusion_matrix(y_true, y_pred, class_names, save_dir='./results'):
    """
    Plot and save confusion matrix

    Args:
        y_true: true labels
        y_pred: predicted labels
        class_names: list of class names
        save_dir: save directory
    """
    os.makedirs(save_dir, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)

    # Normalize confusion matrix row-wise
    cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    cm_normalized = np.nan_to_num(cm_normalized)

    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']  # Use English font
    plt.rcParams['axes.unicode_minus'] = False

    plt.figure(figsize=(10, 8))

    ax = sns.heatmap(cm_normalized, annot=True, fmt='.3f', cmap='Blues',
                     xticklabels=class_names, yticklabels=class_names,
                     cbar_kws={'label': 'Normalized Ratio'})

    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Normalized Confusion Matrix')

    plt.tight_layout()

    save_path = os.path.join(save_dir, 'confusion_matrix.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Confusion matrix saved to: {save_path}")

    return cm


def compute_confusion_matrix(model, data_loader, device, num_classes, class_names=None):
    """
    Compute confusion matrix

    Args:
        model: model
        data_loader: data loader
        device: device
        num_classes: number of classes
        class_names: list of class names

    Returns:
        confusion matrix, classification report, all labels, all predictions
    """
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in data_loader:
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    if class_names is None:
        class_names = [str(i) for i in range(num_classes)]

    cm = confusion_matrix(all_labels, all_preds)

    report = classification_report(all_labels, all_preds,
                                   target_names=class_names,
                                   digits=4)

    return cm, report, all_labels, all_preds


def plot_individual_metrics(history, save_dir='./results'):
    """
    Plot individual metric curves

    Args:
        history: training history dictionary
        save_dir: save directory
    """
    os.makedirs(save_dir, exist_ok=True)

    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    epochs = range(1, len(history['train_loss']) + 1)

    # 1. Loss curves
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['train_loss'], 'b-', linewidth=2, label='Training Loss')
    plt.plot(epochs, history['val_loss'], 'r-', linewidth=2, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'loss_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Accuracy curve (Top-1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['train_acc'], 'b-', linewidth=2, label='Training Accuracy (Top-1)')
    plt.plot(epochs, history['val_acc_top1'], 'r-', linewidth=2, label='Validation Accuracy (Top-1)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy (Top-1)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'accuracy_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Precision curve (if exists)
    if 'val_precision' in history and history['val_precision']:
        plt.figure(figsize=(10, 6))
        if 'train_precision' in history and history['train_precision']:
            plt.plot(epochs, history['train_precision'], 'b-', linewidth=2, label='Training Precision')
        plt.plot(epochs, history['val_precision'], 'r-', linewidth=2, label='Validation Precision')
        plt.xlabel('Epoch')
        plt.ylabel('Precision')
        plt.title('Validation Precision (Macro Avg)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'precision_curve.png'), dpi=300, bbox_inches='tight')
        plt.close()

    # 4. F1-score curve (if exists)
    if 'val_f1' in history and history['val_f1']:
        plt.figure(figsize=(10, 6))
        if 'train_f1' in history and history['train_f1']:
            plt.plot(epochs, history['train_f1'], 'b-', linewidth=2, label='Training F1-score')
        plt.plot(epochs, history['val_f1'], 'r-', linewidth=2, label='Validation F1-score')
        plt.xlabel('Epoch')
        plt.ylabel('F1-score')
        plt.title('Training and Validation F1-score Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'f1_score_curve.png'), dpi=300, bbox_inches='tight')
        plt.close()


def visualize_training_results(history, save_dir='./results'):
    """
    Visualize training results

    Args:
        history: dictionary containing training history, format:
            {
                'train_loss': [...], 'train_acc': [...],
                'val_loss': [...], 'val_acc_top1': [...], 'val_acc_top3': [...],
                'train_f1': [...], 'val_f1': [...] (optional)
            }
        save_dir: results save directory
    """
    os.makedirs(save_dir, exist_ok=True)

    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    epochs = range(1, len(history['train_loss']) + 1)

    plot_individual_metrics(history, save_dir)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # 1. Loss curves
    axes[0].plot(epochs, history['train_loss'], 'b-', linewidth=2, label='Training Loss')
    axes[0].plot(epochs, history['val_loss'], 'r-', linewidth=2, label='Validation Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss Curves')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 2. Accuracy curves (Top-1)
    axes[1].plot(epochs, history['train_acc'], 'b-', linewidth=2, label='Training Accuracy (Top-1)')
    axes[1].plot(epochs, history['val_acc_top1'], 'r-', linewidth=2, label='Validation Accuracy (Top-1)')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training and Validation Accuracy Curves')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('Training Results Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()

    save_path = os.path.join(save_dir, 'training_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Training comparison figure saved to: {save_path}")

    history_path = os.path.join(save_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=4)
    print(f"✓ Training history saved to: {history_path}")

    print("\n" + "=" * 60)
    print("Detailed Training Report")
    print("=" * 60)

    best_val_acc_top1 = max(history['val_acc_top1'])
    best_epoch_top1 = history['val_acc_top1'].index(best_val_acc_top1) + 1

    if 'val_precision' in history and history['val_precision']:
        best_val_precision = max(history['val_precision'])
        best_precision_epoch = history['val_precision'].index(best_val_precision) + 1

    if 'val_f1' in history and history['val_f1']:
        best_val_f1 = max(history['val_f1'])
        best_f1_epoch = history['val_f1'].index(best_val_f1) + 1

    print(f"📊 Accuracy formula: Accuracy = (correct predictions) / (total samples)")
    print(f"📊 For multi-class tasks, Top-1 accuracy is the overall accuracy")
    print(f"🎯 Best validation accuracy (Top-1/Overall): {best_val_acc_top1:.2%} (epoch {best_epoch_top1})")

    if 'val_precision' in history and history['val_precision']:
        print(f"🎯 Best validation precision (macro avg): {best_val_precision:.4f} (epoch {best_precision_epoch})")

    print(f"\n📝 Final training accuracy (Top-1/Overall): {history['train_acc'][-1]:.2%}")
    print(f"📝 Final validation accuracy (Top-1/Overall): {history['val_acc_top1'][-1]:.2%}")

    if 'val_precision' in history and history['val_precision']:
        print(f"📝 Final validation precision (macro avg): {history['val_precision'][-1]:.4f}")

    print(f"\n📉 Final training loss: {history['train_loss'][-1]:.4f}")
    print(f"📉 Final validation loss: {history['val_loss'][-1]:.4f}")

    if 'val_f1' in history and history['val_f1']:
        print(f"\n🎯 Best validation F1-score: {best_val_f1:.4f} (epoch {best_f1_epoch})")
        print(f"📝 Final validation F1-score: {history['val_f1'][-1]:.4f}")

    overfit_gap = history['train_acc'][-1] - history['val_acc_top1'][-1]
    if overfit_gap > 0.15:
        print(f"\n⚠️ Possible overfitting (Top-1 accuracy gap: {overfit_gap:.2f})")
    elif overfit_gap < 0.05:
        print(f"\n✓ Good generalization (Top-1 accuracy gap: {overfit_gap:.2f})")
    else:
        print(f"\n○ Normal (Top-1 accuracy gap: {overfit_gap:.2f})")

    print("=" * 60)


def main(args):
    # Set random seed at the very beginning
    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"using {device} device.")

    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    if os.path.exists("./results") is False:
        os.makedirs("./results")

    tb_writer = SummaryWriter()

    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)

    # Read class names from generated class_indices.json
    if os.path.exists('class_indices.json'):
        with open('class_indices.json', 'r') as f:
            class_indices = json.load(f)
        # class_indices format: {"0": "class_name0", "1": "class_name1", ...}
        class_names = [class_indices[str(i)] for i in range(len(class_indices))]
    else:
        class_names = [f"Class_{i}" for i in range(args.num_classes)]

    # Use a percentage of data for training
    if args.data_ratio < 1.0:
        print(f"\nUsing {args.data_ratio * 100:.1f}% of data for training")
        print(f"Before sampling: {len(train_images_path)} training images, {len(val_images_path)} validation images")

        train_images_path, train_images_label = sample_data_by_ratio(
            train_images_path, train_images_label, args.data_ratio, seed=args.seed
        )

        val_images_path, val_images_label = sample_data_by_ratio(
            val_images_path, val_images_label, args.data_ratio, seed=args.seed
        )

        print(f"After sampling: {len(train_images_path)} training images, {len(val_images_path)} validation images")

    img_size = 224
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(img_size),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(int(img_size * 1.143)),
                                   transforms.CenterCrop(img_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    # Instantiate training dataset
    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=data_transform["train"])

    # Instantiate validation dataset
    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform["val"])

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print('Using {} dataloader workers every process'.format(nw))

    g = torch.Generator()
    g.manual_seed(args.seed)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_dataset.collate_fn,
                                               worker_init_fn=seed_worker,
                                               generator=g)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn,
                                             worker_init_fn=seed_worker,
                                             generator=g)

    model = create_model(num_classes=args.num_classes).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    model_source = f"{create_model.__module__}.{create_model.__name__}"
    print(f"Model source: {model_source}")

    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        checkpoint = torch.load(args.weights, map_location=device)
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            weights_dict = checkpoint["model"]
        else:
            weights_dict = checkpoint
        # Remove classifier-related keys
        for k in list(weights_dict.keys()):
            if "classifier" in k or "fc" in k:
                del weights_dict[k]
        print(model.load_state_dict(weights_dict, strict=False))

    if args.freeze_layers:
        for name, para in model.named_parameters():
            # Freeze all layers except the classifier
            if "classifier" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))

    pg = get_params_groups(model, weight_decay=args.wd)
    optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=args.wd)
    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs, warmup=True, warmup_epochs=5)

    # Initialize training history
    training_history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc_top1': [],
        'val_precision': [],
        'val_f1': []
    }

    best_acc = 0.
    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch,
                                                lr_scheduler=lr_scheduler,
                                                mixup_alpha=args.mixup_alpha)

        val_loss, val_acc_top1, val_precision, val_f1 = evaluate(model=model,
                                                                 data_loader=val_loader,
                                                                 device=device,
                                                                 epoch=epoch,
                                                                 num_classes=args.num_classes)

        training_history['train_loss'].append(train_loss)
        training_history['train_acc'].append(train_acc)
        training_history['val_loss'].append(val_loss)
        training_history['val_acc_top1'].append(val_acc_top1)
        training_history['val_precision'].append(val_precision)
        training_history['val_f1'].append(val_f1)

        tags = ["train_loss", "train_acc", "val_loss", "val_acc_top1", "val_precision", "val_f1"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc_top1, epoch)
        tb_writer.add_scalar(tags[4], val_precision, epoch)
        tb_writer.add_scalar(tags[5], val_f1, epoch)

        if best_acc < val_acc_top1:
            torch.save(model.state_dict(), "./weights/best_model.pth")
            best_acc = val_acc_top1

    print("\n" + "=" * 60)
    print("Training completed, generating visualization results...")
    print("=" * 60)

    visualize_training_results(training_history, save_dir='./results')

    print("\n" + "=" * 60)
    print("Computing confusion matrix...")
    print("=" * 60)

    best_model_path = "./weights/best_model.pth"
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        print(f"Loaded best model: {best_model_path}")
    else:
        print("Best model not found, using final model for confusion matrix computation")

    cm, report, all_labels, all_preds = compute_confusion_matrix(
        model=model,
        data_loader=val_loader,
        device=device,
        num_classes=args.num_classes,
        class_names=class_names
    )

    print("\nClassification Report:")
    print(report)

    plot_confusion_matrix(
        y_true=all_labels,
        y_pred=all_preds,
        class_names=class_names,
        save_dir='./results'
    )

    total_val_samples = len(val_dataset)
    final_top1_acc = training_history['val_acc_top1'][-1]

    report_path = os.path.join('./results', 'classification_report.txt')
    with open(report_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("Training Configuration\n")
        f.write("=" * 60 + "\n")
        f.write(f"Epochs: {args.epochs}\n")
        f.write(f"Batch size: {args.batch_size}\n")
        f.write(f"Learning rate: {args.lr}\n")
        f.write(f"Weight decay: {args.wd}\n")
        f.write(f"Number of classes: {args.num_classes}\n")
        f.write(f"Data path: {args.data_path}\n")
        f.write(f"Pretrained weights: {args.weights if args.weights else 'None'}\n")
        f.write(f"Freeze layers: {args.freeze_layers}\n")
        f.write(f"Device: {args.device}\n")
        f.write(f"Random seed: {args.seed}\n")
        f.write(f"Mixup alpha: {args.mixup_alpha}\n")
        f.write(f"Model source: {model_source}\n")
        f.write("\n" + "=" * 60 + "\n")
        f.write("Training Results Summary\n")
        f.write("=" * 60 + "\n")
        f.write(f"Best validation accuracy (Top-1): {best_acc:.2%}\n")
        f.write(f"Final training accuracy: {training_history['train_acc'][-1]:.2%}\n")
        f.write(f"Final validation accuracy (Top-1): {training_history['val_acc_top1'][-1]:.2%}\n")
        f.write(f"Final validation precision (macro avg): {training_history['val_precision'][-1]:.4f}\n")
        f.write(f"Final training loss: {training_history['train_loss'][-1]:.4f}\n")
        f.write(f"Final validation loss: {training_history['val_loss'][-1]:.4f}\n")
        f.write(f"Number of samples: train={len(train_images_path)}, val={len(val_images_path)}\n")
        f.write(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M\n")

        f.write("\n" + "=" * 60 + "\n")
        f.write("Detailed Classification Report\n")
        f.write("=" * 60 + "\n")
        f.write(report)

        f.write(f"\n\n" + "=" * 60 + "\n")
        f.write("Accuracy Detailed Analysis\n")
        f.write("=" * 60 + "\n")
        f.write(f"   Total validation samples: {total_val_samples}\n")
        f.write(f"   Top-1 correct predictions: {int(final_top1_acc * total_val_samples)}\n")
        f.write(f"   Top-1 accuracy (overall): {final_top1_acc:.2%}\n")
        f.write(f"\nAccuracy formula: Accuracy = (correct predictions) / (total samples)\n")
        f.write("\nConfusion matrix:\n")
        f.write(np.array2string(cm, separator=', '))

    print(f"✓ Classification report saved to: {report_path}")

    # ---------- Generate Grad-CAM interpretability figures ----------
    print("\n" + "=" * 60)
    print("Generating Grad-CAM interpretability figures...")
    print("=" * 60)

    target_layer = model.lsk_att
    gradcam = GradCAM(model, target_layer)
    model.eval()

    samples_per_class = {}
    with torch.no_grad():
        for images, labels in val_loader:
            for i in range(len(labels)):
                true_label = labels[i].item()
                if true_label not in samples_per_class:
                    samples_per_class[true_label] = (images[i].cpu(), true_label)
                if len(samples_per_class) == args.num_classes:
                    break
            if len(samples_per_class) == args.num_classes:
                break

    os.makedirs('./results/gradcam', exist_ok=True)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    for class_idx, (img_tensor, label) in samples_per_class.items():
        img_show = img_tensor * std + mean
        img_show = torch.clamp(img_show, 0, 1)
        img_np = img_show.permute(1, 2, 0).cpu().numpy()

        input_tensor = img_tensor.unsqueeze(0).to(device)
        input_tensor.requires_grad = True

        heatmap = gradcam.generate(input_tensor, target_class=None)

        from PIL import Image
        heatmap_pil = Image.fromarray(np.uint8(255 * heatmap))
        heatmap_resized = heatmap_pil.resize((img_np.shape[1], img_np.shape[0]), resample=Image.BILINEAR)
        heatmap = np.array(heatmap_resized) / 255.0

        cmap = plt.get_cmap('jet')
        heatmap_colored = cmap(heatmap)[..., :3]
        superimposed = heatmap_colored * 0.4 + img_np * 0.6
        class_name = class_names[label] if class_names else str(label)
        save_path = f'./results/gradcam/class_{class_name}.png'
        plt.imsave(save_path, np.clip(superimposed, 0, 1))
        print(f"  Grad-CAM for {class_name} saved to {save_path}")

    print("✓ All Grad-CAM interpretability figures generated.")
    # ------------------------------------------------------------


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=8,
                        help='Number of output classes, must match the actual dataset')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Total training epochs, each epoch traverses the entire training set once')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size, affects training speed and GPU memory usage')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate, controls parameter update step size')
    parser.add_argument('--wd', type=float, default=0.1,
                        help='Weight decay (L2 regularization), used to prevent overfitting')

    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed to ensure reproducibility')

    parser.add_argument('--data-ratio', type=float, default=1,
                        help='Proportion of data to use, e.g., 0.1 means using 10%% of the data')

    parser.add_argument('--mixup-alpha', type=float, default=0.,
                        help='Mixup hyperparameter, >0 enables mixup (e.g., 0.2)')

    # Dataset path: adjust to your ISIC 2019 data location
    parser.add_argument('--data-path', type=str,
                        default=r"isic2019",
                        help="Path to the ISIC 2019 dataset folder (contains train/val subfolders or images)")
    parser.add_argument('--weights', type=str, default='',
                        help='Initial weights path')
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='Device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)
