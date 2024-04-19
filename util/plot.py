import matplotlib.pyplot as plt
import numpy as np

#プロット用関数
def plot_generates(df,generate_num=500,epochs=20,fmt="o-"):
    plt.figure(figsize=(30,20))
    plt.plot([0]*generate_num,color='red')
    for i in range(epochs):
        q=df[i*generate_num:(i+1)*generate_num].values
        plt.plot(q,fmt,label="epoch"+str(i))
    plt.legend()        
        
def plot_generates_log(df,generate_num=500,epochs=20,fmt="o-",ignorenegative=False):
    plt.figure(figsize=(30,20))
    plt.plot([0]*generate_num,color='red')
    for i in range(epochs):
        q=df[i*generate_num:(i+1)*generate_num].values
        lq=np.log(q)
        if(ignorenegative):
            np.nan_to_num(lq,nan=0.,copy=False)
        plt.plot(lq,fmt,label="epoch"+str(i))
    plt.legend()    
    
    
def plot_per_epoch(df,islog=True,offset=0,generate_num=500,fmt="o-"):
    plt.figure(figsize=(15,10))
    interval=generate_num//10 #50
    for i in range(10):        
        start=i*interval+offset
        q=df[start::generate_num].values
        if(islog):
            q=np.log(q)       
        plt.plot(q,fmt,label=str(start))
        plt.legend(loc='upper right')
    plt.legend()    
    plt.show()