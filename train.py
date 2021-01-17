import argparse
import os
import shutil
import torch
import math
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torchvision
from dataloaders.dataset_paired_rgb import PairedRGBDataset, ValPairedRGBDataset
from networks.denoising_rgb import DenoiseNet

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
def psnr(org, ref, max_value=255):
   mse = torch.mean((org -ref) ** 2, dim=[1,2,3])
   psnr = 20 * torch.log10(max_value / (torch.sqrt(mse)))
   return torch.mean(psnr)

def evaluate(model, data, criterion):
    model.eval()
    epoch_loss = 0
    for batch in tqdm(data):
        x, y, _, _ = batch
        x = x.to(device)
        y = y.to(device)
        y_hat = model(x)
        # done
        loss = criterion(y_hat, y)
        epoch_loss += loss.item()
    return epoch_loss / len(data)



def train(config):
    # done
    model = DenoiseNet()
    model = model.to(device)
    if config.pretrain_dir is not None:
        model.load_state_dict(torch.load(config.pretrain_dir))
    # todo: done
    train_dataset = ValPairedRGBDataset(config.train_data, (config.crop, config.crop))
    train_data = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True)
    val_dataset = ValPairedRGBDataset(config.val_data, (config.crop, config.crop))
    val_data = torch.utils.data.DataLoader(val_dataset, batch_size=config.batch_size - 1, shuffle=True, num_workers=config.num_workers, pin_memory=True)
    # todo: done
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay, betas=(0.9, 0.999))
    criterion = torch.nn.L1Loss(reduction='mean')
    lowest_loss = math.inf
    for epoch in range(config.num_epochs):
        print('Epoch', epoch)
        epoch_loss = 0
        for batch in tqdm(train_data):
            x, y, _, _ = batch
            x = x.to(device)
            y = y.to(device)
            # print(x.shape)
            # print(y.shape)
            y_hat = model(x)
            # done
            loss = criterion(y_hat, y)
            optimizer.zero_grad()
            loss.backward()
            if config.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm(model.parameters(), config.grad_clip_norm)
            optimizer.step()
            epoch_loss += loss.item()
        writer.add_image('noisy', x[0], epoch)
        writer.add_image('target', y[0], epoch)
        writer.add_image('predict', y_hat[0], epoch)
        print('train_loss', epoch_loss)
        writer.add_scalar('train', epoch_loss, epoch)
        torch.cuda.empty_cache()
        val_loss = evaluate(model, val_data, criterion)
        if val_loss < lowest_loss:
            lowest_loss = val_loss
            torch.save(model.state_dict(), os.path.join(
                config.snapshot_folder, "best.pth"))
            print('val_loss', val_loss)
            writer.add_scalar('validation', val_loss, epoch)
        if (epoch % config.model_saved_freq) == 0:
            torch.save(model.state_dict(), os.path.join(
                config.snapshot_folder, "Epoch" + str(epoch) + '.pth'))

def test(config):
    model = SKUNet()
    model = model.to(device)
    model.load_state_dict(torch.load(config.pretrain_dir))
    val_dataset = PairedDataset(os.path.join(config.val_data, config.low_folder) , os.path.join(config.val_data, config.high_folder))
    val_data = torch.utils.data.DataLoader(val_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True)
    i = 0
    epoch_loss = 0
    for x,y in tqdm(val_data):
        x = x.to(device)
        y = y.to(device)
        y_hat, _ = model(x)            
        # done
        for j in range(y_hat.shape[0]):
            torchvision.utils.save_image(y_hat[j], os.path.join(config.snapshot_folder, "img_{}_enhanced.png".format(i)))
            torchvision.utils.save_image(y[j], os.path.join(config.snapshot_folder, "img_{}.png".format(i)))
            i += 1
        loss = psnr(y_hat, y, 1)
        epoch_loss += loss.item()
    print('Loss:', epoch_loss / len(val_data))
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Input Parameters
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--train_data', type=str, required=True)
    parser.add_argument('--val_data', type=str, required=True)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--grad_clip_norm', type=float, default=0.1)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--output_display_freq', type=int, default=10)
    parser.add_argument('--model_saved_freq', type=int, default=10)
    parser.add_argument('--crop', type=int, default=512)
    parser.add_argument('--log_dir', type=str,
                        default="ss512/log")
    parser.add_argument('--snapshot_folder', type=str,
                        default="ss512")
    parser.add_argument('--pretrain_dir', type=str)
    config = parser.parse_args()

    if not os.path.exists(config.snapshot_folder):
        os.mkdir(config.snapshot_folder)
    # else:
    #     choice = input("Do you want to remove {}?".format(
    #         config.snapshot_folder))
    #     if choice == 'y':
    #         shutil.rmtree(config.snapshot_folder)
    #         os.mkdir(config.snapshot_folder)
    if os.path.exists(config.log_dir):
        shutil.rmtree(config.log_dir)
    writer = SummaryWriter(config.log_dir)
    if config.test:
        test(config)
    else:
        train(config)