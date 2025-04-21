from torchvision.transforms import v2
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10


def get_dataloader(
    data_root="./data/CIFAR10",
    batch_size=64,
    num_workers=4
):
    train_transforms = v2.Compose([
        v2.RandomCrop(32, padding=4),
        v2.AugMix(),
        v2.RandomHorizontalFlip(0.5),
        v2.ToTensor(),
        v2.Normalize((0.4914, 0.4822, 0.4465),
                     (0.2023, 0.1994, 0.2010))
    ])

    test_transforms = v2.Compose([
        v2.ToTensor(),
        v2.Normalize((0.4914, 0.4822, 0.4465),
                     (0.2023, 0.1994, 0.2010))
    ])

    train_set = CIFAR10(
        root=data_root,
        train=True, 
        transform=train_transforms,
        download=True
    )

    test_set = CIFAR10(
        root=data_root,
        train=False,
        transform=test_transforms,
        download=True
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        num_workers=num_workers
    )

    return train_loader, test_loader



if __name__ == "__main__":
    train_loader, test_loader = get_dataloader(1, 0)
    print(len(train_loader), len(test_loader))

    for image, label in train_loader:
        print(image.size(), image.mean(), image.std(), label)
        quit()

