import jax.numpy as jnp


# TODO: Add scaling, normalization, etc.
class ImageDataset:
    """
    A class representing an image dataset.

    Args:
        images (jnp.ndarray): The array of images.
        labels (jnp.ndarray | None, optional): The array of labels. Defaults to None.
        name (str | None, optional): The name of the dataset. Defaults to None.

    Raises:
        ValueError: If the number of images and labels do not match.
        ValueError: If the images do not have shape (N, H, W).

    Attributes:
        images (jnp.ndarray): The array of images.
        labels (jnp.ndarray | None): The array of labels.
        name (str | None): The name of the dataset.

    Methods:
        __repr__(): Returns a string representation of the ImageDataset object.
        __len__(): Returns the number of images in the dataset.
        __getitem__(idx: int): Returns the image and label at the specified index.

    """

    def __init__(
        self,
        images: jnp.ndarray,
        labels: jnp.ndarray | None = None,
        name: str | None = None,
    ) -> None:
        if labels is not None and images.shape[0] != labels.shape[0]:
            raise ValueError("Number of images and labels must match")

        if images.ndim != 3:
            raise ValueError("Images must have shape (N, H, W)")

        self.images = images
        self.labels = labels
        self.name = name

    def __repr__(self) -> str:
        return f"ImageDataset(name={self.name}, shape={self.images.shape}, labels={self.labels is not None})"

    def __len__(self) -> int:
        return self.images.shape[0]

    def __getitem__(self, idx: int) -> tuple[jnp.ndarray, jnp.ndarray | None]:
        return self.images[idx], self.labels[idx] if self.labels is not None else None
