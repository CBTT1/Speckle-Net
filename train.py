import os

from torch import nn, optim
import torch
from torch.utils.data import DataLoader
from data import *
from net import *
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
weight_path = 'params/unet.pth'
data_path = r'data'
save_path = 'train_image'

def pearson_correlation(x, y):
    mean_x = torch.mean(x)
    mean_y = torch.mean(y)
    xm = x - mean_x
    ym = y - mean_y
    r_num = torch.sum(xm * ym)
    r_den = torch.sqrt(torch.sum(xm ** 2)) * torch.sqrt(torch.sum(ym ** 2))
    r = r_num / r_den
    return r

class myNPCCLoss(nn.Module):
    def __init__(self):
        super(myNPCCLoss, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, preds, targets):
        mse = self.mse_loss(preds, targets)
        pccs = pearson_correlation(preds, targets)
        npcc = 1 - pccs ** 2
        # 结合MSE和NPCC
        loss = mse + npcc
        # loss = npcc
        return loss

if __name__ == '__main__':
    train_size = int(0.6 * len(MyDataset(data_path)))
    print(train_size)
    test_size = len(MyDataset(data_path)) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(MyDataset(data_path), [train_size, test_size])
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True, drop_last=True)
    net = UNet().to(device)
    if os.path.exists(weight_path):
        net.load_state_dict(torch.load(weight_path))
        print('successful load weight！')
    else:
        print('not successful load weight')

    learning_rate = 0.001
    opt = optim.Adam(net.parameters(), lr=learning_rate)
    # loss_fun=nn.BCELoss()
    loss_fun = myNPCCLoss()

    writer = SummaryWriter("logs/logs_train")
    epoch = 10
    total_train_step = 0
    for epoch_number in range(epoch):
        for i, (image, segment_image) in enumerate(train_dataloader):
            image, segment_image = image.to(device), segment_image.to(device)

            out_image = net(image)
            train_loss = loss_fun(out_image, segment_image)

            opt.zero_grad()
            train_loss.backward()

            opt.step()
            writer.add_scalar("train_loss",train_loss.item(), total_train_step)
            total_train_step += 1

            if i % 10 == 0:
                print(f'{epoch_number}-{i}-train_loss===>>{train_loss.item()}')
                writer.add_scalar("train_loss", train_loss.item(), i)
                writer.add_images(f"Epoch:{epoch_number}", image, i)
                writer.add_images(f"Epoch:{epoch_number}_out", out_image, i)
                writer.add_images(f"Epoch:{epoch_number}_gd", segment_image, i)
            if i % 50 == 0:
                torch.save(net.state_dict(), weight_path)

            _image = image[0]
            _segment_image = segment_image[0]
            _out_image = out_image[0]

            img = torch.stack([_image, _segment_image, _out_image], dim=0)
            save_image(img, f'{save_path}/{i}.png')
        total_test_loss = 0
        total_test_step = 0
        with torch.no_grad():
            for i, (image, segment_image) in enumerate(test_dataloader):
                image, segment_image = image.to(device), segment_image.to(device)
                out_image = net(image)
                loss = loss_fun(out_image, segment_image)
                total_test_loss += loss.item()
        print(f"整体测试集上的LOSS:{total_test_loss}")
        writer.add_scalar("test_loss", total_test_loss, total_test_step)
        total_test_step += 1


    writer.close()