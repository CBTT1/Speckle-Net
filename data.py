import os

from torch.utils.data import Dataset
from utils import *
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

transform = transforms.Compose([
    transforms.ToTensor()
])

class MyDataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.name = os.listdir(os.path.join(path, 'test_psf'))

    def __len__(self):
        return len(self.name)

    def __getitem__(self, index):
        segment_name = self.name[index]  # xx.png
        segment_path = os.path.join(self.path, 'test', segment_name)
        image_path = os.path.join(self.path, 'test_psf', segment_name)
        segment_image = keep_image_size_open(segment_path, size=(64, 64), reverse=False)
        segment_image.save('./data/test_gd.png')
        image = keep_image_size_open(image_path, size=(64, 64), reverse=False)
        image.save('./data/test_size.png')
        return transform(image), transform(segment_image)


if __name__ == '__main__':
    # writer = SummaryWriter("logs")
    data = MyDataset('D:\\Code\\pytorch-unet-master\\data')
    # writer.add_image("ground_truth", data[0][1], global_step=1)
    # writer.add_image("scattered", data[0][0], global_step=1)
    # writer.close()
    print(data[0][0].shape)
    print(data[0][1].shape)
