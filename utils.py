# utils.py
import os
import sys
import json
import pickle
import random
import math
import numpy as np

import torch
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

import matplotlib.pyplot as plt


# ---------- Mixup function ----------
def mixup_data(x, y, alpha=1.0, device='cuda'):
    """Returns mixed inputs, pairs of targets, and lambda"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
# -------------------------------------


def read_split_data(root: str, val_rate: float = 0.2):
    """
    Read data and split into training and validation sets

    Args:
        root: dataset root directory
        val_rate: validation set ratio

    Returns:
        image paths and labels for training and validation sets
    """
    random.seed(0)  # ensure reproducible results
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    # Check if train/test folders exist
    train_dir = os.path.join(root, 'train')
    test_dir = os.path.join(root, 'test')

    # If train/test folders exist, use them directly
    if os.path.exists(train_dir) and os.path.exists(test_dir):
        print("Detected train/test folder structure, using directly...")

        # Generate class names and corresponding indices (based on training set)
        flower_class = [cla for cla in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, cla))]
        flower_class.sort()
        class_indices = dict((k, v) for v, k in enumerate(flower_class))
        json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
        with open('class_indices.json', 'w') as json_file:
            json_file.write(json_str)

        train_images_path = []  # store all training image paths
        train_images_label = []  # store corresponding indices for training images
        val_images_path = []  # store all validation image paths
        val_images_label = []  # store corresponding indices for validation images
        every_class_num = []  # store total number of samples per class
        supported = [".jpg", ".JPG", ".png", ".PNG"]  # supported file extensions

        # Read training set
        for cla in flower_class:
            cla_path = os.path.join(train_dir, cla)
            images = [os.path.join(train_dir, cla, i) for i in os.listdir(cla_path)
                      if os.path.splitext(i)[-1] in supported]
            images.sort()
            image_class = class_indices[cla]
            every_class_num.append(len(images))

            for img_path in images:
                train_images_path.append(img_path)
                train_images_label.append(image_class)

        # Read test set (as validation set)
        for cla in flower_class:
            cla_path = os.path.join(test_dir, cla)
            if not os.path.exists(cla_path):
                continue
            images = [os.path.join(test_dir, cla, i) for i in os.listdir(cla_path)
                      if os.path.splitext(i)[-1] in supported]
            images.sort()
            image_class = class_indices[cla]

            for img_path in images:
                val_images_path.append(img_path)
                val_images_label.append(image_class)

        print("{} images were found in the dataset.".format(sum(every_class_num) + len(val_images_path)))
        print("{} images for training.".format(len(train_images_path)))
        print("{} images for validation.".format(len(val_images_path)))
        assert len(train_images_path) > 0, "number of training images must greater than 0."
        assert len(val_images_path) > 0, "number of validation images must greater than 0."

        plot_image = True
        if plot_image:
            # Plot bar chart of class distribution
            plt.bar(range(len(flower_class)), every_class_num, align='center')
            # Replace x-axis numbers with class names
            plt.xticks(range(len(flower_class)), flower_class)
            # Add value labels on bars
            for i, v in enumerate(every_class_num):
                plt.text(x=i, y=v + 5, s=str(v), ha='center')
            plt.xlabel('image class')
            plt.ylabel('number of images')
            plt.title('skin diseases class distribution')
            plt.show()

        return train_images_path, train_images_label, val_images_path, val_images_label

    # Original code (if no train/test folders)
    # Traverse folders, each folder corresponds to one class
    flower_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    # Sort to ensure consistent order across platforms
    flower_class.sort()
    # Generate class names and corresponding indices
    class_indices = dict((k, v) for v, k in enumerate(flower_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_images_path = []  # store all training image paths
    train_images_label = []  # store corresponding indices for training images
    val_images_path = []  # store all validation image paths
    val_images_label = []  # store corresponding indices for validation images
    every_class_num = []  # store total number of samples per class
    supported = [".jpg", ".JPG", ".png", ".PNG"]  # supported file extensions
    # Traverse each folder
    for cla in flower_class:
        cla_path = os.path.join(root, cla)
        # Get all supported file paths
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
        images.sort()
        image_class = class_indices[cla]
        every_class_num.append(len(images))
        # Randomly sample validation samples
        val_path = random.sample(images, k=int(len(images) * val_rate))

        for img_path in images:
            if img_path in val_path:  # if in validation set, add to validation
                val_images_path.append(img_path)
                val_images_label.append(image_class)
            else:  # otherwise add to training set
                train_images_path.append(img_path)
                train_images_label.append(image_class)

    print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for training.".format(len(train_images_path)))
    print("{} images for validation.".format(len(val_images_path)))
    assert len(train_images_path) > 0, "number of training images must greater than 0."
    assert len(val_images_path) > 0, "number of validation images must greater than 0."

    plot_image = True
    if plot_image:
        # Plot bar chart of class distribution
        plt.bar(range(len(flower_class)), every_class_num, align='center')
        # Replace x-axis numbers with class names
        plt.xticks(range(len(flower_class)), flower_class)
        # Add value labels on bars
        for i, v in enumerate(every_class_num):
            plt.text(x=i, y=v + 5, s=str(v), ha='center')
        plt.xlabel('image class')
        plt.ylabel('number of images')
        plt.title('skin diseases class distribution')
        plt.show()

    return train_images_path, train_images_label, val_images_path, val_images_label


def plot_data_loader_image(data_loader):
    """
    Plot images from data loader

    Args:
        data_loader: data loader
    """
    batch_size = data_loader.batch_size
    plot_num = min(batch_size, 4)

    json_path = './class_indices.json'
    assert os.path.exists(json_path), json_path + " does not exist."
    json_file = open(json_path, 'r')
    class_indices = json.load(json_file)

    for data in data_loader:
        images, labels = data
        for i in range(plot_num):
            # [C, H, W] -> [H, W, C]
            img = images[i].numpy().transpose(1, 2, 0)
            # Reverse normalization
            img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
            label = labels[i].item()
            plt.subplot(1, plot_num, i + 1)
            plt.xlabel(class_indices[str(label)])
            plt.xticks([])  # remove x-axis ticks
            plt.yticks([])  # remove y-axis ticks
            plt.imshow(img.astype('uint8'))
        plt.show()


def write_pickle(list_info: list, file_name: str):
    """
    Write list information to a pickle file

    Args:
        list_info: list information
        file_name: file name
    """
    with open(file_name, 'wb') as f:
        pickle.dump(list_info, f)


def read_pickle(file_name: str) -> list:
    """
    Read list information from a pickle file

    Args:
        file_name: file name

    Returns:
        list information
    """
    with open(file_name, 'rb') as f:
        info_list = pickle.load(f)
        return info_list


def train_one_epoch(model, optimizer, data_loader, device, epoch, lr_scheduler, mixup_alpha=0.0):
    """
    Train for one epoch, support mixup data augmentation

    Args:
        model: model
        optimizer: optimizer
        data_loader: data loader
        device: device
        epoch: current epoch
        lr_scheduler: learning rate scheduler
        mixup_alpha: mixup hyperparameter, if >0 enables mixup

    Returns:
        average loss and training accuracy (Top-1)
    """
    model.train()

    loss_function = torch.nn.CrossEntropyLoss()

    accu_loss = torch.zeros(1).to(device)  # accumulated loss
    accu_num = torch.zeros(1).to(device)  # accumulated number of correct predictions (using original labels)
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        sample_num += images.shape[0]

        # Apply mixup (if enabled)
        if mixup_alpha > 0:
            mixed_images, labels_a, labels_b, lam = mixup_data(images, labels, alpha=mixup_alpha, device=device)
            outputs = model(mixed_images)
            loss = mixup_criterion(loss_function, outputs, labels_a, labels_b, lam)
            # Still use original labels to compute accuracy (for display only)
            pred_classes = torch.max(outputs, dim=1)[1]
            accu_num += lam * torch.eq(pred_classes, labels_a).sum() + (1 - lam) * torch.eq(pred_classes, labels_b).sum()
        else:
            outputs = model(images)
            pred_classes = torch.max(outputs, dim=1)[1]
            accu_num += torch.eq(pred_classes, labels).sum()
            loss = loss_function(outputs, labels)

        loss.backward()
        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}, lr: {:.5f}".format(
            epoch,
            accu_loss.item() / (step + 1),
            accu_num.item() / sample_num,
            optimizer.param_groups[0]["lr"]
        )

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()
        # update lr
        lr_scheduler.step()  # commented due to ReduceLROnPlateau

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


@torch.no_grad()
def evaluate(model, data_loader, device, epoch, num_classes=None):
    """
    Evaluation function, returns loss, Top-1 accuracy (overall accuracy), precision (macro avg), and F1-score (macro avg)

    Args:
        model: model
        data_loader: data loader
        device: device
        epoch: current epoch
        num_classes: number of classes

    Returns:
        avg_loss: average loss
        accuracy_top1: Top-1 accuracy (overall accuracy)
        precision: precision (macro average)
        f1: F1-score (macro average)

    Notes:
        Overall accuracy formula: Accuracy = (number of correct predictions) / (total number of samples)
        For multi-class tasks, Top-1 accuracy equals overall accuracy
    """
    loss_function = torch.nn.CrossEntropyLoss()

    model.eval()

    # Initialize statistics
    correct_top1 = torch.zeros(1).to(device)  # number of correct Top-1 predictions
    total_loss = torch.zeros(1).to(device)  # accumulated loss
    total_samples = 0  # total number of samples

    # Lists for precision, F1-score and overall accuracy
    all_preds_top1 = []  # Top-1 predictions
    all_labels = []  # true labels

    data_loader = tqdm(data_loader, desc=f"[valid epoch {epoch}]", file=sys.stdout)

    for step, data in enumerate(data_loader):
        images, labels = data
        batch_size = images.shape[0]
        total_samples += batch_size

        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)

        # Compute loss
        loss = loss_function(outputs, labels)
        total_loss += loss.detach() * batch_size  # accumulate loss multiplied by batch size

        # 1. Compute Top-1 accuracy (overall accuracy)
        # Formula: Accuracy = (number of correct predictions) / (total number of samples)
        _, preds_top1 = torch.max(outputs, 1)
        correct_top1 += torch.eq(preds_top1, labels).sum()

        # Collect predictions for precision and F1-score
        all_preds_top1.extend(preds_top1.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        # Update progress bar display
        data_loader.set_description(
            f"[valid epoch {epoch}] loss: {total_loss.item() / total_samples:.3f}, "
            f"acc(top1): {correct_top1.item() / total_samples:.3f}"
        )

    # Compute final metrics
    avg_loss = total_loss.item() / total_samples
    accuracy_top1 = correct_top1.item() / total_samples  # overall accuracy = Top-1 accuracy

    all_preds_top1 = np.array(all_preds_top1)
    all_labels = np.array(all_labels)

    # Compute macro average precision
    precision = precision_score(all_labels, all_preds_top1, average='macro', zero_division=0)
    # Compute macro average F1-score
    f1 = f1_score(all_labels, all_preds_top1, average='macro', zero_division=0)

    # Print evaluation results
    print(f"\n[valid epoch {epoch}] Evaluation results:")
    print(f"  Top-1 accuracy (overall accuracy): {accuracy_top1:.2%}")
    print(f"  Precision (macro avg): {precision:.4f}")
    print(f"  F1-score (macro avg): {f1:.4f}")

    return avg_loss, accuracy_top1, precision, f1


def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3,
                        end_factor=1e-6):
    """
    Create a learning rate scheduler

    Args:
        optimizer: optimizer
        num_step: number of steps per epoch
        epochs: total number of epochs
        warmup: whether to use warmup
        warmup_epochs: number of warmup epochs
        warmup_factor: warmup factor
        end_factor: final learning rate factor

    Returns:
        learning rate scheduler
    """
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        """
        Returns a learning rate multiplier based on step number.
        Note: PyTorch calls lr_scheduler.step() once before training starts.
        """
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # lr multiplier goes from warmup_factor -> 1 during warmup
            return warmup_factor * (1 - alpha) + alpha
        else:
            current_step = (x - warmup_epochs * num_step)
            cosine_steps = (epochs - warmup_epochs) * num_step
            # lr multiplier goes from 1 -> end_factor after warmup
            return ((1 + math.cos(current_step * math.pi / cosine_steps)) / 2) * (1 - end_factor) + end_factor

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)


def get_params_groups(model: torch.nn.Module, weight_decay: float = 1e-5):
    """
    Get parameter groups for weight decay

    Args:
        model: model
        weight_decay: weight decay coefficient

    Returns:
        list of parameter groups
    """
    # Record trainable weight parameters
    parameter_group_vars = {"decay": {"params": [], "weight_decay": weight_decay},
                            "no_decay": {"params": [], "weight_decay": 0.}}

    # Record corresponding weight names
    parameter_group_names = {"decay": {"params": [], "weight_decay": weight_decay},
                             "no_decay": {"params": [], "weight_decay": 0.}}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights

        if len(param.shape) == 1 or name.endswith(".bias"):
            group_name = "no_decay"
        else:
            group_name = "decay"

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)

    print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
    return list(parameter_group_vars.values())