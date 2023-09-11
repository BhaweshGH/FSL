from torch import Tensor
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torchvision
import numpy as np
from typing import List
from prototypical_networks import PrototypicalNetworks
from tqdm import tqdm
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def plot_images(images: Tensor, title: str, images_per_row: int):
    """
    Plot images in a grid.
    Args:
        images: 4D mini-batch Tensor of shape (B x C x H x W)
        title: title of the figure to plot
        images_per_row: number of images in each row of the grid
    """
    plt.figure()
    plt.title(title)
    plt.imshow(
        torchvision.utils.make_grid(images, nrow=images_per_row).permute(1, 2, 0)
    )
    plt.show()

def sliding_average(value_list: List[float], window: int) -> float:
    """
    Computes the average of the latest instances in a list
    Args:
        value_list: input list of floats (can't be empty)
        window: number of instances to take into account. If value is 0 or greater than
            the length of value_list, all instances will be taken into account.

    Returns:
        average of the last window instances in value_list

    Raises:
        ValueError: if the input list is empty
    """
    if len(value_list) == 0:
        raise ValueError("Cannot perform sliding average on an empty list.")
    return np.asarray(value_list[-window:]).mean()


def evaluate_on_one_task(
        support_images: Tensor,
        support_labels: Tensor,
        query_images: Tensor,
        query_labels: Tensor,
        model: PrototypicalNetworks
) -> [int, int]:
    """
    Returns the number of correct predictions of query labels, and the total number of predictions.
    """
    return (
            torch.max(
                model(support_images.cuda(), support_labels.cuda(), query_images.cuda())
                .detach()
                .data,
                1,
            )[1]
            == query_labels.cuda()
        ).sum().item(), len(query_labels)


def evaluate(data_loader: DataLoader, model: PrototypicalNetworks):
    # We'll count everything and compute the ratio at the end
    total_predictions = 0
    correct_predictions = 0
    # eval mode affects the behaviour of some layers (such as batch normalization or dropout)

    # no_grad() tells torch not to keep in memory the whole computational graph (it's more lightweight this way)

    model.eval()
    with torch.no_grad():
        for episode_index, (
                support_images,
                support_labels,
                query_images,
                query_labels,
                class_ids,
        ) in tqdm(enumerate(data_loader), total=len(data_loader)):
            correct, total = evaluate_on_one_task(
                support_images, support_labels, query_images, query_labels, model
            )

            total_predictions += total
            correct_predictions += correct
    print(
        f"Model tested on {len(data_loader)} tasks. Accuracy: {(100 * correct_predictions / total_predictions):.2f}%"
    )


