import numpy as np
class L2_Reg:
    def Loss(self,W):
        return np.sum(np.dot(W.T,W))
    def Derivative(self,W):
        return 2*W

class L1_Reg:
    def Loss(self,W):
        return np.sum(np.abs(W))
    def Derivative(self,W):
        return (W>0)

class None_Reg:
    def Loss(self,W):
        return 0
    def Derivative(self,W):
        return 0
    
class regression:
    def __init__(self,regularization="None"):
        reg_map={"L2":L2_Reg(),"L1":L1_Reg(),"None":None_Reg()}
        self.reg_function=reg_map[regularization]
        self.Loss_history={"train":[],"val":[]}
    def Train(self,x,y,calc_val=0,x_val=None,y_val=None,lr=0.000001,omega=0.1,batch_size=32,epoch=5,v=1):
        n,m=x.shape
        self.W=np.random.rand(m+1,1)
        x=np.concatenate([x,np.ones((n,1))],axis=1)
        if(calc_val!=0):
            x_val=np.concatenate([x_val,np.ones((x_val.shape[0],1))],axis=1)
        for ep in range(epoch):
            if(v==1):
                print("+"*5,ep,"+"*5)
            Loss_epoch={"train":[],"val":[]}
            for i in range(0,n,batch_size):
                x_batch=x[i:i+batch_size,:]
                y_batch=y[i:i+batch_size]
                y_pred=np.dot(x_batch,self.W).reshape(y_batch.shape)
                train_loss=self.Loss(y_pred,y_batch)+(omega*self.reg_function.Loss(self.W))
                Loss_epoch["train"].append(train_loss)
                if(calc_val!=0):
                    y_val_pred=np.dot(x_val,self.W).reshape(y_val.shape)
                    valid_loss=self.Loss(y_val_pred,y_val)+(omega*self.reg_function.Loss(self.W))
                    Loss_epoch["val"].append(valid_loss)
                w_grad=(2*np.dot(x_batch.T,(y_pred-y_batch).reshape(-1,1))/batch_size)+(omega*self.reg_function.Derivative(self.W))
                self.W-=lr*w_grad
            self.Loss_history["train"].append(np.mean(Loss_epoch["train"]))
            self.Loss_history["val"].append(np.mean(Loss_epoch["val"]))
            if(v==1):
                print("Train Loss : ",self.Loss_history["train"][-1])
                print("Valid Loss : ",self.Loss_history["val"][-1])
                print("="*10)
    def Loss(self,y_pred,y):
        return np.mean((y_pred-y)**2)
    def pred(self,test):
        test=np.concatenate([test,np.ones((test.shape[0],1))],axis=1)
        return np.dot(test,self.W).reshape(-1)
