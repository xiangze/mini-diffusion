import torch
import numpy as np

tdot=torch.dot
tsum=torch.sum
tmean=torch.mean
tflatten=torch.flatten
from  diffuser import Diffuser


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


get_LHS=True
get_RHS=True


#ある学習ステップでサンプリングしたときのTURの左辺と右辺        
def TUR_sample(epoch:int,ddpm_model:Diffuser,img_size,obs,
               guide_w=0.0,n_generate_sample=500,TUR_samplenum=1000,init_every_sample=True,skp=1,
               logfilename="TUR_log.csv",debug=False):
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
