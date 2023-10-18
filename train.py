import argparse
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
    _r1 = _r[1]+torch.sum(r)*torch.sum(r)  #torch.sum(r*r)
    return [_r0,_r1]    

debug=False
isTURsample=True
get_LHS=True
get_RHS=True

#ある学習ステップでサンプリングしたときのTURの左辺と右辺        
def TUR_sample(epoch:int,ddpm_model:DDPM,img_size,obs,
               guide_w=0.0,n_generate_sample=500,TUR_samplenum=1000,init_every_sample=True,skp=1,
               logfilename="TUR_log.csv"):
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

                x= torch.randn(img_size).cuda().repeat(2, 1, 1, 1)
                xo= torch.randn(img_size).cuda()

                #generating step
                for t in range(n_generate_sample - 1, 0, -skp):
                    xpath=[]

                    if(get_LHS):
                        t_is = torch.tensor([t / n_generate_sample]).cuda()
                        t_is=t_is.repeat(num_samples, 1, 1, 1)
                        t_is=t_is.repeat(2, 1, 1, 1)
                        if(init_every_sample):                    
                            x= torch.randn(img_size).cuda().repeat(2, 1, 1, 1)
                            xo= torch.randn(img_size).cuda()
                        D=ddpm_model.betas[t]/2
                        #compute LHS of TUR: entoropy production= j^T B^{-1} j/P = Ai^2P+2D(nabra_i Ai)P-(nabla_i^2 logP)P
                        _lhs=0
                        _lhs2=0
                        _lhsh=0                                                
                        _scores=0
                        _Ai=0                        
                        for i in range(TUR_samplenum):
                            z = torch.randn(num_samples,*img_size).cuda()
    
                            eps = ddpm_model.model(xo.repeat(2, 1, 1, 1), c_i, t_is, ctx_mask)
                            eps1 = eps[:num_samples]
                            eps2 = eps[num_samples:]
                            eps = (1 + guide_w)*eps1 - guide_w*eps2                        
                            score=eps
                            x= ddpm_model.sample1(xo.repeat(2, 1, 1, 1),t,z,c_i,ctx_mask,eps)
                            Ai=ddpm_model.A(x,t)
                            
                            v=torch.flatten(Ai-D*score)
                            v2=torch.dot(v,v)
                            _lhs+= v2/D

                            v1=torch.flatten(Ai-2*D*score)
                            v12=torch.dot(v1,v1)
                            _lhs2+= v12/D

                            vh=torch.flatten(Ai-D*score/2)
                            vh2=torch.dot(vh,vh)
                            _lhsh+= vh2/D

                            _scores+=torch.sum(score)
                            _Ai+=torch.sum(Ai)
                            xo=x
                            if(debug and t<=40):
                                xpath.append(torch.flatten(x).detach().cpu().numpy())

                        TUR_lhs.append([_lhs.detach().cpu().numpy()/TUR_samplenum,
                                        _lhs2.detach().cpu().numpy()/TUR_samplenum,
                                        _lhsh.detach().cpu().numpy()/TUR_samplenum,
                                        _scores.detach().cpu().numpy()/TUR_samplenum,
                                        _Ai.detach().cpu().numpy()/TUR_samplenum,
                                        D.detach().cpu().numpy()/TUR_samplenum
                                        ])
                    if(debug and t<=40):
                        for x in xpath:
                            with open("lastxpath_epoch{}t{}.csv".format(epoch,t),"a") as fp:
                                np.savetxt(fp,x)

                    #compute RHS of TUR: <R>^2/Var<R> of variable R=obs cdot dx
                    if(get_RHS):
                        if(init_every_sample):                    
                            x= torch.randn(img_size).cuda() 
                            xo= torch.randn(img_size).cuda() 
                        _rd=[0,0]
                        _rF=[0,0]                        
                        _rF2=[0,0]                        

                        rds=[]

                        for i in range(TUR_samplenum):
                            z = torch.randn(num_samples,*img_size).cuda()                        
                            x= ddpm_model.sample1(xo,t,z,c_i,ctx_mask,eps)
                            xe=(x+xo)/2
                            #stratnovich obs
                            rd=torch.mean(xe)*(x-xo)

                            score=ddpm_model.model(xe.repeat(2, 1, 1, 1), c_i, t_is, ctx_mask)[0]
                            Ai=ddpm_model.A(xe,t)
                            F=torch.flatten(Ai/D-score)
                            rF=torch.dot(F,torch.flatten(x-xo))
                            rF2=torch.sum(F)*(x-xo)

                            _rd=increment_1_2(_rd,rd)
                            _rF=increment_1_2(_rF,rF)
                            _rF2=increment_1_2(_rF2,rF2)                            
#                            if(debug):
#                                rds.append([torch.sum(rd).detach().cpu().numpy(),(torch.sum(rd)*torch.sum(rd)).detach().cpu().numpy()])
                            
                            xo  =  x

                        #mean,var,rhs
                        rhss=[
                        calc_meanvar(_rd,TUR_samplenum),
                        calc_meanvar(_rF,TUR_samplenum),
                        calc_meanvar(_rF2,TUR_samplenum)
                        ]
                        TUR_rhs.append(rhss)
                        
                        # var<
                        for i,r in enumerate(rhss):
                            #assert(r[1]>=0)
                            if(debug and r[1]<0):
                                with open("rd_trace{}.csv".format(i),"a") as fp:
                                    fp.write("#{},{}\n".format(epoch,t))
                                    mean=0
                                    v=0
                                    for i in range(len(rds)):
                                        fp.write("{},{}\n".format(rds[i][0],rds[i][1]))
                                        mean+=rds[i][0]
                                        v+=rds[i][1]
                                    mean=mean/len(rds)
                                    var=v/len(rds)-mean*mean
                                    fp.write("#mean,var(pytorch){},{}\n".format(rhs_d[0],rhs_d[1]))
                                    fp.write("#mean,var(numpy){},{}\n".format(mean,var))
                                exit()

                #[ [TUR_lhs[i],TUR_lhs[i][0][0], ]for i in range(n_generate_sample)]

                if(epoch==0):
                    with open(logfilename,"w") as fp:
                        fp.write("epoch,gen_step,")
                        fp.write("TUR_lhs,")
                        fp.write("TUR_lhsx2,TUR_lhs_h,score,Ai,D,")
                        for r in TUR_rhs[0]:
                            fp.write("mean,var,rhs,LHS/RHS,")
                        fp.write("\n")


                with open(logfilename,"a") as fp:
                    for i in range(n_generate_sample//skp):
                        if(get_LHS):
                                fp.write("{},{},".format(epoch,i))
                                for l in TUR_lhs[i]:
                                    fp.write("{},".format(l))

                        if(get_RHS):
                            for r in TUR_rhs[i]:
                                #mean,var,rhs,lhs/rhs
                                for j in r:
                                    fp.write("{},".format(j))
                                fp.write("{},".format(TUR_lhs[i][0]/r[2]))
                        fp.write("\n")

                        

                return [TUR_lhs,TUR_rhs]


"""
train :学習の1 step(epoch)

"""
def train(unet:UNet, ddpm_model:DDPM, loader, opt, criterion, scaler, num_cls, save_dir, ws, epoch,
                    n_generate_sample=500,TUR_samplenum=1000,
                    logfilename="TUR_sample.csv", init_every_sample=True,skip=1):
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
    num_cls = 10

    parser = argparse.ArgumentParser(description='コマンドライン引数の例')
    parser.add_argument('-n', '--num_epochs')  

    num_epochs = 3
    save_dir = 'result'
    unet = UNet(1, 128, num_cls).cuda()
    ddpm_model = DDPM(unet, (1e-4, 0.02)).cuda()

    skip=2
    init_every_sample=True
    TUR_samplenum=1000
    if (init_every_sample):
        logfilename="TUR_log_betas_skip{}sample{}epoch{}.csv".format(skip,TUR_samplenum,num_epochs)
    else:
        logfilename="TUR_log_betas_skip{}sample{}epoch{}_tradition.csv".format(skip,TUR_samplenum,num_epochs)
                
    tr = T.Compose([T.ToTensor()])
    dataset = tv.datasets.MNIST('data', True, transform = tr, download = True)
    loader = DataLoader(dataset, batch_size = 64, shuffle = True, num_workers = 0)

    opt = torch.optim.Adam(list(ddpm_model.parameters()) + list(unet.parameters()), lr = 1e-4)
    criterion = nn.MSELoss()

    scaler = torch.cuda.amp.grad_scaler.GradScaler()
    
    ws = [0.0, 0.5, 1.0]
    

    for epoch in range(num_epochs):
        train(unet, ddpm_model, loader, opt, criterion, scaler, num_cls, save_dir, ws, epoch,
              n_generate_sample=500,TUR_samplenum=TUR_samplenum,
              logfilename=logfilename, init_every_sample=init_every_sample,
              skip=skip)


