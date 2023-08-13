import torch
import torch.nn as nn
import torchvision as tv
import torchvision.transforms as T
from torch.utils.data import DataLoader
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from tqdm import tqdm
#import wandb

from model.Unet import UNet
from ddpm import DDPM


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

def calc_meanvar(_r,TUR_samplenum):
    _mean=torch.sum(_r[0])/TUR_samplenum
    mean2=_mean*_mean
    _var = (_r[1]/TUR_samplenum -mean2) *TUR_samplenum/(TUR_samplenum-1)

    mean=_mean.detach().cpu().numpy()
    var=_var.detach().cpu().numpy()
    rhs=2*mean2.detach().cpu().numpy()/var
    return mean,var,rhs

def increment_1_2(_r,r):
    _r0 = _r[0]+torch.sum(r)
    _r1 = _r[1]+torch.sum(r*r)
    return [_r0,_r1]    

#ある学習ステップでサンプリングしたときのTURの左辺と右辺        
def TUR_sample(epoch,ddpm_model,img_size,obs,
               guide_w=0.0,n_generate_sample=500,TUR_samplenum=1000):
                """
                ddpm_model: 拡散モデル 
                img_size:データ(画像)サイズ
                obs: 右辺(rhs)で取得する統計量の関数
                guide_w: guide=promptの重み
                n_generate_sample: 総生成ステップ数
                TUR_samplenum: 平均<>(∫dxP(x))を計算するためのサンプリング数
                """
                #unet.train()
 
                num_samples=1
                c_i = torch.arange(0, 1).cuda()                   
                c_i = c_i.repeat(int(num_samples / c_i.shape[0]))
                ctx_mask = torch.zeros_like(c_i).cuda()
                c_i = c_i.repeat(2)

                ctx_mask = ctx_mask.repeat(2)
                ctx_mask[num_samples:] = 1.0

                TUR_lhs=[]
                TUR_rhs=[]
                for t in range(n_generate_sample - 1, 0, -1):
                    t_is = torch.tensor([t / n_generate_sample]).cuda()
                    t_is=t_is.repeat(num_samples, 1, 1, 1)
                    t_is=t_is.repeat(2, 1, 1, 1)
                    
                    x= torch.randn(img_size).cuda().repeat(2, 1, 1, 1)
                    xo= torch.randn(img_size).cuda().repeat(2, 1, 1, 1)
                    beta=ddpm_model.sqrt_beta_t[t]
                    #compute LHS of TUR: entoropy production= j^T B^{-1} j/P = Ai^2P+2D(nabra_i Ai)P-(nabla_i^2 logP)P
                    _lhs=0
                    for i in range(TUR_samplenum):
                        z = torch.randn(num_samples,*img_size).cuda()
                        #print("{},xo size {}".format(i,xo.size()))
 
                        eps = ddpm_model.model(xo, c_i, t_is, ctx_mask)
                        eps1 = eps[:num_samples]
                        eps2 = eps[num_samples:]
                        eps = (1 + guide_w)*eps1 - guide_w*eps2                        
                        score=eps
                        x= ddpm_model.sample1(xo,t,z,c_i,ctx_mask,eps)
                        Ai=ddpm_model.A(x,t)
                        #with torch.cuda.amp.autocast_mode.autocast():
                            #score = unet(x.half(), c_i, timestep.half(), ctx_mask.half())
                            #score=ddpm_model.model(xo,c_i,t_is,ctx_mask)
                        #    score.backward()
                        v=torch.flatten(Ai-beta*score)
                        v2=torch.dot(v,v)
                        _lhs+= v2/(2*beta)
                        xo=x.repeat(2, 1, 1, 1)
                    TUR_lhs.append(_lhs.detach().cpu().numpy()/TUR_samplenum)

                    #compute RHS of TUR: <R>^2/Var<R> of variable R=obs cdot dx
                    x= torch.randn(img_size).cuda() 
                    xo= torch.randn(img_size).cuda() 
                    _r=[0,0]
                    _rd=[0,0]
                    _rx=[0,0]
                    _r0=[0,0]
                    for i in range(TUR_samplenum):
                        z = torch.randn(num_samples,*img_size).cuda()                        
                        x= ddpm_model.sample1(xo,t,z,c_i,ctx_mask,eps)
                        #stratnovich obs
                        rd=obs((x+xo)/2)*(x-xo)
                        r=obs((x+xo)/2)
                        rx=obs(x)
                        r0=x[0,0,0,0]

                        _r[0] += torch.sum(r)
                        _r[1] += torch.sum(r*r)
                        _rd[0] += torch.sum(rd)
                        _rd[1] += torch.sum(rd*rd)
                        _rx[0] += torch.sum(rx)
                        _rx[1] += torch.sum(rx*rx)
                        _r0[0] += torch.sum(r0)
                        _r0[1] += torch.sum(r0*r0)

                        xo  =  x

                    #mean,var,rhs=calc_meanvar(_r,TUR_samplenum)
                    #mean_d,var_d,rhs_d=calc_meanvar(_rd,TUR_samplenum)
                    #mean_x,var_x,rhs_x=calc_meanvar(_rd,TUR_samplenum)
                    rhs=calc_meanvar(_r,TUR_samplenum)
                    rhs_d=calc_meanvar(_rd,TUR_samplenum)
                    rhs_x=calc_meanvar(_rx,TUR_samplenum)
                    rhs_0=calc_meanvar(_r0,TUR_samplenum)
                    TUR_rhs.append([rhs,rhs_d,rhs_x,rhs_0])
                    
                #[ [TUR_lhs[i],TUR_lhs[i][0][0], ]for i in range(n_generate_sample)]
                with open("TUR_log.csv","a") as fp:
                    for i in range(n_generate_sample-1):
                        fp.write("{},{},{},".format(epoch,i,TUR_lhs[i]))
                        for r in TUR_rhs[i]:
                            for j in r:
                                fp.write("{},".format(j))
                            fp.write("{},".format(TUR_lhs[i]/r[2]))
                        fp.write("\n")
                return [TUR_lhs,TUR_rhs]

"""
train :学習の1 step(epoch)

"""
def train(unet:UNet, ddpm_model:DDPM, loader, opt, criterion, scaler, num_cls, save_dir, ws, epoch):
    img_size=(1,28,28)
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
                TUR_samples=TUR_sample(epoch,ddpm_model,img_size,obs)

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

isTURsample=True

def main():

    num_cls = 10
    num_epochs = 20
    save_dir = 'result'
    unet = UNet(1, 128, num_cls).cuda()
    ddpm_model = DDPM(unet, (1e-4, 0.02)).cuda()


    tr = T.Compose([T.ToTensor()])
    dataset = tv.datasets.MNIST('data', True, transform = tr, download = True)
    loader = DataLoader(dataset, batch_size = 64, shuffle = True, num_workers = 0)

    opt = torch.optim.Adam(list(ddpm_model.parameters()) + list(unet.parameters()), lr = 1e-4)
    criterion = nn.MSELoss()

    scaler = torch.cuda.amp.grad_scaler.GradScaler()
    
    ws = [0.0, 0.5, 1.0]
    
    with open("TUR_log.csv","w") as fp:
        fp.write("epoch,gen_step,TUR_lhs,TUR_rhs,LHS/RHS\n")

    for epoch in range(num_epochs):
        train(unet, ddpm_model, loader, opt, criterion, scaler, num_cls, save_dir, ws, epoch)


if __name__ == '__main__':

#    wandb.init(project = 'MinDiffusion')
    main()
