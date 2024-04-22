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
from ddpm import DDPM,SMLD

tdot=torch.dot
tsum=torch.sum
tmean=torch.mean
tflatten=torch.flatten

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

class RHS:
    def __init__(self) -> None:
        self.ave =0
        self.f2 =0
        self.aveA =0
        self.aveB =0
        self.varD =0

    def inclement(self,f,df,s,F,A,D):
        """
        F: force
        f: tensor
        s: scalar like f(x) cdot dx
        """
        self.ave  = self.ave+s
        self.f2   = self.f2+s*s
        self.aveA = self.aveA+tdot(f,F)
        self.aveB = self.aveB+tdot(A,f)+D*torch.sum(df)
        self.varD = self.varD+tdot(f,f)*D
        
    def meanvar(self,TUR_samplenum):
#        var=_var.detach().cpu().numpy()
#        meanB=_meanB.detach().cpu().numpy()
#        mean2=(_mean*_mean).detach().cpu().numpy()
#        mean2B=(_meanB*_meanB).detach().cpu().numpy()
#        mean=_mean.detach().cpu().numpy()
        mean=self.ave/TUR_samplenum
        f2  = self.f2/TUR_samplenum
        meanA=self.aveA/TUR_samplenum
        meanB=self.aveB/TUR_samplenum
        varD = self.varD/TUR_samplenum

        mean2=(mean*mean)
        mean2A=(meanA*meanA)
        mean2B=(meanB*meanB)
        var = (f2 -mean2)*TUR_samplenum/(TUR_samplenum-1)/2
        rhs=2*mean2/var
        if(varD!=0):
            rhsA=2*mean2A/varD
            rhsB=2*mean2B/varD
        else:
            rhsA=0
            rhsB=0

        return mean,meanA,meanB,var,varD,rhs,rhsA,rhsB
    
def RHS_printheader(fp,i:int):
        fp.write("mean.{},meanA(F).{},meanB(df).{},var.{},varD.{},rhs.{},rhsA(F).{},rhsB(df).{},".format(i,i,i,i,i,i,i,i))

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
                    D=ddpm_model.betas[t]/2
                    if(get_LHS):
                        t_is = torch.tensor([t / n_generate_sample]).cuda()
                        t_is=t_is.repeat(num_samples, 1, 1, 1)
                        t_is=t_is.repeat(2, 1, 1, 1)
                        if(init_every_sample):                    
                            x= torch.randn(img_size).cuda().repeat(2, 1, 1, 1)
                            xo= torch.randn(img_size).cuda()
                        #compute LHS of TUR: entoropy production= j^T B^{-1} j/P = Ai^2P+2D(nabra_i Ai)P-(nabla_i^2 logP)P
                        _lhs=0
                        _scores=0
                        _Ai=0                        
                        for i in range(TUR_samplenum):
                            z = torch.randn(num_samples,*img_size).cuda()
                            xo=xo.repeat(2, 1, 1, 1)
                            score = ddpm_model.model(xo, c_i, t_is, ctx_mask)[0]
                            Ai=ddpm_model.A(xo,t,score)
                            v=torch.flatten(Ai-D*score)
                            v2=torch.dot(v,v)
                            _lhs+= v2/D
                            _scores+=torch.sum(score)
                            _Ai+=torch.sum(Ai)
                            #next step
                            xo=ddpm_model.sample1(xo,t,z,c_i,ctx_mask,score) #diffusion step xo=[2,1,28,28],x=[1,1,28,28]
                            if(debug and t<=40):
                                xpath.append(torch.flatten(xo).detach().cpu().numpy())

                        TUR_lhs.append([_lhs.detach().cpu().numpy()/TUR_samplenum,
                                        v2.detach().cpu().numpy()/TUR_samplenum,
                                        _scores.detach().cpu().numpy()/TUR_samplenum,
                                        _Ai.detach().cpu().numpy()/TUR_samplenum,
                                        D.detach().cpu().numpy() 
                                        ])
                    if(debug and t<=40):
                        for x in xpath:
                            with open("lastxpath/lastxpath_epoch{}t{}.csv".format(epoch,t),"a") as fp:
                                np.savetxt(fp,x)

                    #compute RHS of TUR: <R>^2/Var<R> of variable R=obs cdot dx
                    #
                    if(get_RHS):
                        if(init_every_sample):                    
                            x= torch.randn(img_size).cuda() 
                            xo= torch.randn(img_size).cuda() 
                        
                        RHSS=[RHS() for r in range(6)]
                        rds=[]

                        for i in range(TUR_samplenum):
                            z = torch.randn(num_samples,*img_size).cuda()
                            xo2=xo.repeat(2, 1, 1, 1)
                            score=ddpm_model.model(xo2, c_i, t_is, ctx_mask)[0]
                            x= ddpm_model.sample1(xo2,t,z,c_i,ctx_mask,score)
                            #stratnovich obs
                            xe=(x+xo)/2
                            dx=x-xo
                            #等式条件用でもある
                            #avef=<f(AP-D∇P)>=∫dx Pf*(A-Dscore) =: <f*F>
                            #=∫dx Pf(A-D∇P/P)=∫dx PfA-fD∇P=∫dx PfA+DP∇f=<fA+D∇f> )
                            #score=ddpm_model.model(xo, c_i, t_is, ctx_mask)[0]
                            Ai=ddpm_model.A(xo,t,score)
                            F=tflatten(Ai-D*score)
                            A=tflatten(Ai)
                            #f
                            score_e=ddpm_model.model(xe.repeat(2, 1, 1, 1), c_i, t_is, ctx_mask)[0]
                            Ae=ddpm_model.A(xe,t,score_e)
                            Fe=tflatten(Ae-D*score_e)

                            xe=tflatten(xe)
                            dx=tflatten(dx)
                            xe2=xe*xe
                            xe3=xe2*xe
                            xf=tflatten(xo)
                            xf2=xf*xf
                            xf3=xf*xf2

                            fs=[
                                tflatten(torch.ones(x.shape).cuda()),
                                xf,
            -                   xf2,
                                xe,
                                xe2,
                                F
                            ]
                            dfs=[
                                tflatten(torch.zeros(x.shape).cuda()),
                                tflatten(torch.ones(x.shape).cuda()),
                                2*xf,
                                tflatten(torch.ones(x.shape).cuda()),
                                2*xe,
                                F
                            ]

                            sc=[
                                torch.sum(dx),
                                tdot(xe,dx),
                                tdot(xe2,dx),
                                tdot(xe,dx),
                                tdot(xe2,dx),
                                tdot(Fe,dx),
                            ]

                            for i,f in enumerate(fs):
                                if(i==3 or i==4):
                                    RHSS[i].inclement(f,dfs[i],sc[i],Fe,tflatten(Ae),D)
                                else:
                                    RHSS[i].inclement(f,dfs[i],sc[i],F,A,D)

#                            if(debug):
#                                rds.append([torch.sum(rd).detach().cpu().numpy(),(torch.sum(rd)*torch.sum(rd)).detach().cpu().numpy()])
                            xo  =  x

                        #mean,var,rhs
                        rhs_v=[r.meanvar(TUR_samplenum) for r in RHSS]
                        TUR_rhs.append(rhs_v)
                        
                        for i,r in enumerate(rhs_v):
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
                        fp.write("TUR_lhs,(Ai-D*score)^2,score,Ai,D,")
                        for i,r in enumerate(TUR_rhs[0]):
                            RHS_printheader(fp,i)
                        fp.write("\n")

                with open(logfilename,"a") as fp:
                    for i in range(n_generate_sample//skp):
                        if(get_LHS):
                                fp.write("{},{},".format(epoch,i))
                                for l in TUR_lhs[i]:
                                    fp.write("{},".format(l))

                        if(get_RHS):
                            for r in TUR_rhs[i]:
                                #mean,meanB,var,rhs,rhsB,LHS/RHS
                                for j in r:
                                    fp.write("{},".format(j))
#                                fp.write("{},".format(TUR_lhs[i][0]/r[3]))
                        fp.write("\n")

                        

                return [TUR_lhs,TUR_rhs]


"""
train :学習の1 step(epoch)

"""
def train(unet:UNet, ddpm_model:DDPM, loader, opt, criterion, scaler, num_cls, save_dir, ws, epoch,
                    n_generate_sample=500,TUR_samplenum=1000,
                    logfilename="log/TUR_sample.csv", init_every_sample=True,skip=1):
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

    parser = argparse.ArgumentParser(description='学習、生成段階におけるエントロピー生成率と変数のゆらぎを出力する')
    parser.add_argument('-n', '--num_epochs',default=10,type=int)  
    parser.add_argument('-s', '--sample_num',default=1000,type=int)      
    parser.add_argument('-i', '--tradition',action="store_true")
    parser.add_argument('-t', '--type',default="DDPM") #"SMLD"
    parser.add_argument('-l', '--learningrate',default=1e-4,type=float)#lr = 1e-4  5e-6
    parser.add_argument('-skip', '--skip',default=10,type=int)
    parser.add_argument('-save_dir', '--save_dir',default="result")
    parser.add_argument('-sche', '--scheduler_type',default="linear")

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

    suffix=diftype

    unet = UNet(1, 128, num_cls).cuda()
    
    if(diftype=="SMLD"):
        ddpm_model = SMLD(unet, (1e-4, 0.02),scheduler_type=scheduler_type).cuda()        
    else:
        ddpm_model = DDPM(unet, (1e-4, 0.02),scheduler_type=scheduler_type).cuda()


    if (init_every_sample):
        logfilename="log/TUR_log_skip{}sample{}epoch{}_{}_lr{}_{}.csv".format(skip,TUR_samplenum,num_epochs,scheduler_type,lr,suffix)
    else:
        logfilename="log/TUR_log_skip{}sample{}epoch{}_{}_tradition_lr{}_{}.csv".format(skip,TUR_samplenum,num_epochs,scheduler_type,lr,suffix)
                
    tr = T.Compose([T.ToTensor()])
    dataset = tv.datasets.MNIST('data', True, transform = tr, download = True)
    loader = DataLoader(dataset, batch_size = 64, shuffle = True, num_workers = 0)

    opt = torch.optim.Adam(list(ddpm_model.parameters()) + list(unet.parameters()), lr =lr)
    criterion = nn.MSELoss()

    scaler = torch.cuda.amp.grad_scaler.GradScaler()
    
    ws = [0.0, 0.5, 1.0]
    

    for epoch in range(num_epochs):
        train(unet, ddpm_model, loader, opt, criterion, scaler, num_cls, save_dir, ws, epoch,
              n_generate_sample=500,TUR_samplenum=TUR_samplenum,
              logfilename=logfilename, init_every_sample=init_every_sample,
              skip=skip)


