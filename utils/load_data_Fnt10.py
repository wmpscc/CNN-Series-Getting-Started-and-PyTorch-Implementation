import torch
from torchvision import transforms, datasets


def load_data_Fnt10(input_size=227, batch_size=32):
    train_transforms = transforms.Compose([
        # transforms.Resize(input_size),
        transforms.RandomRotation(10),
        transforms.RandomResizedCrop(input_size),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((.5, .5, .5), (.5, .5, .5))
    ])

    val_transforms = transforms.Compose([
        transforms.Resize(input_size),
        # transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((.5, .5, .5), (.5, .5, .5))
    ])

    train_dir = "../dataset/Fnt10"
    train_datasets = datasets.ImageFolder(train_dir, train_transforms)
    train_dataloader = torch.utils.data.DataLoader(train_datasets, batch_size=batch_size, shuffle=True, num_workers=2)

    val_dir = "../dataset/Fnt10"
    val_datasets = datasets.ImageFolder(val_dir, val_transforms)
    val_dataloader = torch.utils.data.DataLoader(val_datasets, batch_size=batch_size, shuffle=True, num_workers=2)
    return train_dataloader, val_dataloader


if __name__ == "__main__":
    train_dataloader, val_dataloader = load_data_Fnt10()
    for img, lables in train_dataloader:
        print(lables)
