import torch
import yaml

from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

from preprocessing import CustomDataset
from models.simplemodel import SmallCNN


def main():
    config = yaml.safe_load(open("config/default.yaml"))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = transforms.Compose([
        transforms.Resize(tuple(config['dataset']['img_size'])),
        transforms.ToTensor()
    ])
    target_transform = None
    train_path = config['dataset']['path']+"train"
    test_path = config['dataset']['path']+"val"

    trainds = CustomDataset(path=train_path,
                            class_names=config['dataset']['class_names'],
                            transform=transform,
                            target_transform=target_transform
                            )
    val_split = int(0.2 * len(trainds))
    trainds, valds = random_split(trainds, [len(trainds) - val_split, val_split])
    testds = CustomDataset(path=test_path,
                            class_names=config['dataset']['class_names'],
                            transform=transform,
                            target_transform=target_transform
                            )
    traindl = DataLoader(trainds, batch_size=config['dataset']['batch_size'])
    valdl = DataLoader(valds, batch_size=config['dataset']['batch_size'])
    testdl = DataLoader(testds, batch_size=config['dataset']['batch_size'])

    model = SmallCNN(config['model'])
    print(model)

    
    

    return

if __name__=="__main__":
    main()