import argparse
import torch
import torch.nn as nn
import torchvision as tv
import torchvision.transforms as T
from torch.utils.data import DataLoader
import cv2
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from tqdm import tqdm
#import wandb

from model.Unet import UNet
from diffuser import DDPM,SMLD
from TURsample import TUR_sample
from gendata import eda,swissroll
import numpy as np 

class AverageMeter:
    def __init__(self, name=None):
        self.name = name
        self.reset()

    def reset(self):
        self.sum = self.count = self.avg = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


@torch.no_grad()
def plot(ddpm_model, num_cls, ws, save_dir, epoch):

    ddpm_model.eval()
    num_samples = 4 * num_cls
    
    for w_i, w in enumerate(ws):

        pred, pred_arr = ddpm_model.sample(num_samples, (1,28,28), num_cls, w)
        real = torch.tensor(pred.shape).cuda()
        #print(pred.shape, real.shape)
        #combined = torch.cat([real, pred])
        grid = tv.utils.make_grid(pred, nrow = 10)

        grid_arr = grid.squeeze().detach().cpu().numpy()
        print(f'Grid Array Shape: {grid_arr.shape}')
        cv2.imwrite(f'{save_dir}/pred_epoch_{epoch}_w_{w}.png', grid_arr.transpose(1,2,0))
    
    ddpm_model.train()
    return grid_arr.transpose(1,2,0)

"""
train :学習の1 step(epoch)

"""
def train(unet:UNet, ddpm_model:DDPM, loader, opt, criterion, scaler, num_cls, save_dir, ws, epoch,
                    n_generate_sample=500,TUR_samplenum=1000,img_size=(1,28,28),isTURsample=True,
                    logfilename="log/TUR_sample.csv", init_every_sample=True,skip=1):

    obs=lambda x:torch.mean(x)
    #obs2=lambda x:torch.torch.var(x)

    unet.train()
    ddpm_model.train()

    #wandb.log({'Epoch': epoch })

    loop = tqdm(loader, position = 0, leave = True)
    loss_ = AverageMeter()

    for idx, (img, class_lbl) in enumerate(loop):

        img = img.cuda(non_blocking = True)
        lbl = class_lbl.cuda(non_blocking = True)

        opt.zero_grad(set_to_none = True)

        with torch.cuda.amp.autocast_mode.autocast():

            noise, x_t, ctx, timestep, ctx_mask = ddpm_model(img, lbl)
            pred = unet(x_t.half(), ctx, timestep.half(), ctx_mask.half())
            loss = criterion(noise, pred)

        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()



        loss_.update(loss.detach(), img.size(0))

        if idx % 200 == 0:
            #wandb.log({'Loss': loss_.avg })
            print(loss_.avg)
        

    if epoch % 1 == 0:




      ddpm_model.eval()
      
      with torch.no_grad():
            n_sample = 4*num_cls
            #普通のサンプリング
            for w_i, w in enumerate(ws):
                x1, xis = ddpm_model.sample(n_sample, img_size, num_cls,w)

            #ある学習ステップでサンプリングしたときのTURの左辺と右辺
            if(isTURsample):
                TUR_samples=TUR_sample(epoch,ddpm_model,img_size,obs,
                                       n_generate_sample=n_generate_sample,TUR_samplenum=TUR_samplenum,
                                       init_every_sample=init_every_sample,
                                       logfilename=logfilename,skp=skip)

      fig, ax = plt.subplots(nrows = n_sample // num_cls, ncols = num_cls, sharex = True, sharey = True, figsize = (10, 4))

      def animate_plot(i, xis):

        plots = []

        for row in range(n_sample // num_cls):

          for col in range(num_cls):

            ax[row, col].clear()
            ax[row, col].set_xticks([])
            ax[row, col].set_yticks([])

            plots.append(ax[row, col].imshow(-xis[i, (row*num_cls) + col, 0], cmap = 'gray', vmin = (-xis[i]).min(), vmax = (-xis[i]).max()))
        
        return plots

      ani = FuncAnimation(fig, animate_plot, fargs = [xis], interval = 200, blit = False, repeat = True, frames = xis.shape[0])
      ani.save(f'{save_dir}/epoch_{epoch}.gif', dpi = 100, writer = PillowWriter(fps = 5))
      print('GIF Saved!')

      torch.save(ddpm_model.state_dict(), os.path.join(save_dir, f'ddpm.pth'))
      torch.save(unet.state_dict(), os.path.join(save_dir, f'unet.pth'))
    
    print("#sample @ epoch%d"%(epoch))

    return TUR_samples


if __name__ == '__main__':
#    wandb.init(project = 'MinDiffusion')
    

    parser = argparse.ArgumentParser(description='学習、生成段階におけるエントロピー生成率と変数のゆらぎを出力する')
    parser.add_argument('-n', '--num_epochs',default=10,type=int)  
    parser.add_argument('-s', '--sample_num',default=1000,type=int)      
    parser.add_argument('-i', '--tradition',action="store_true")
    parser.add_argument('-t', '--type',default="DDPM") #"SMLD"
    parser.add_argument('-l', '--learningrate',default=1e-4,type=float)#lr = 1e-4  5e-6
    parser.add_argument('-skip', '--skip',default=10,type=int)
    parser.add_argument('-save_dir', '--save_dir',default="result")
    parser.add_argument('-sche', '--scheduler_type',default="linear")
    parser.add_argument('-bs', '--batchsize',default=64,type=int)
    parser.add_argument('-ds', '--dataset',default="mnist")    

    args = parser.parse_args()
#    print(args)
    num_epochs = args.num_epochs
    diftype=args.type
    init_every_sample=not args.tradition
    TUR_samplenum=args.sample_num
    lr =args.learningrate
    skip=args.skip
    save_dir =args.save_dir
    scheduler_type=args.scheduler_type
    batchsize=args.batchsize
    dataset=args.dataset
    suffix=diftype


    logfilename="log/TUR_log_skip{}sample{}epoch{}_{}_lr{}_{}".format(skip,TUR_samplenum,num_epochs,scheduler_type,lr,suffix)
    if (init_every_sample):
        logfilename+=".csv"
    else:
        logfilename+="_tradition.csv"
                
    tr = T.Compose([T.ToTensor()])
    if(dataset=="mnist"):
        ds = tv.datasets.MNIST('data', True, transform = tr, download = True)
        img_size=(1,28,28)
        num_cls = 10
    if(dataset=="kmnist"):    
        ds = tv.datasets.KMNIST('data', True, transform = tr, download = True)        
        img_size=(1,28,28)
        num_cls = 10
    elif(dataset=="cifar10"):
        ds = tv.datasets.CIFAR10('data', True, transform = tr, download = True)
        img_size=(3,32,32) #        128,1,3,3
        num_cls = 10
    elif(dataset=="cifar100"):
        ds = tv.datasets.CIFAR100('data', True, transform = tr, download = True)
        img_size=(3,32,32)
        num_cls = 100    
    elif(dataset=="imagenet"):
        ds = tv.datasets.imagenet('data', True, transform = tr, download = True)
        img_size=(3,28,28)
    elif(dataset=="eda"):          
        ddim=5
        img_size=(1,4,4)
        ds=eda(ddim,4*4).reshape(img_size)
    elif(dataset=="swissroll"):          
        N=6000
        img_size=(1,28,28)
        ddim=28*28
        num_cls=10
        d,label=swissroll(N,ddim)
        label=np.array(label)
        label=((label-min(label))/(max(label)-min(label))*num_cls).astype("int")
        ds = torch.utils.data.TensorDataset( torch.Tensor(d),  torch.Tensor(label))
    else:
        print("unsupported dataset {}".format(dataset))

#    print(ds.shape)
    loader = DataLoader(ds, batch_size = batchsize, shuffle = True, num_workers = 0)

    unet = UNet(img_size[0], 128, num_cls).cuda()

    if(diftype=="SMLD"):
        ddpm_model = SMLD(unet, (1e-4, 0.02),scheduler_type=scheduler_type).cuda()        
    else:
        ddpm_model = DDPM(unet, (1e-4, 0.02),scheduler_type=scheduler_type).cuda()

    opt = torch.optim.Adam(list(ddpm_model.parameters()) + list(unet.parameters()), lr =lr)
    criterion = nn.MSELoss()

    scaler = torch.cuda.amp.grad_scaler.GradScaler()
    
    ws = [0.0, 0.5, 1.0]
    
    for epoch in range(num_epochs):
        train(unet, ddpm_model, loader, opt, criterion, scaler, num_cls, save_dir, ws, epoch,
              n_generate_sample=500,TUR_samplenum=TUR_samplenum,img_size=img_size,
              logfilename=logfilename, init_every_sample=init_every_sample,
              skip=skip)


