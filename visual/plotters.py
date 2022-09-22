import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE


def plot_tracker(tracker):
    plt.figure(figsize=(10, 10))
    plt.plot(tracker)
    plt.legend(["Training", "Validation"])
    plt.xlabel("Optimisation Steps")
    plt.ylabel("Model Error")
    plt.show()


def plot_images(*input_images):
    if type(input_images) == np.ndarray:
        input_images = (input_images,)

    num_images = len(input_images)
    num_rows = int(num_images / 2) + (num_images % 2 > 0)
    fig, ax = plt.subplots(num_rows, 2, figsize=(10, 10))
    for i, image in enumerate(input_images):
        ax[i].imshow(image)
        plt.axis("off")
    plt.show()


def plot_latents(latents, labels):
    if latents.shape[1] > 2:
        latents = TSNE().fit_transform(latents)
    labels = np.array([np.array2string(l) for l in labels])
    classes = np.flip(np.sort(np.unique(labels)))

    plt.figure(figsize=(10, 10))
    for c in classes:
        class_latents = latents[np.argwhere(labels == c).squeeze(), :]
        plt.scatter(class_latents[:, 0], class_latents[:, 1])
        plt.axis("off")
        plt.legend(classes)
    plt.show()
