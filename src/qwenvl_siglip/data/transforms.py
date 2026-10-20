import torchvision.transforms as T


def build_transform(size: int = 384) -> T.Compose:
    """Squash-resize (no crop) + SigLIP normalisation (mean = std = 0.5).

    Not cropping keeps normalised box coordinates valid after resizing.
    """
    return T.Compose(
        [
            T.Resize((size, size), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize([0.5] * 3, [0.5] * 3),
        ]
    )
