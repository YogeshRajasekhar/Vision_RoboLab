from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints
import numpy as np
from transforms import Transforms

class UKF:
    def __init__(self,dh,ini_T,dim_x=6,dim_z=6,alpha=0.1,beta=2.0,kappa=1.0,ini_state=np.zero(6),ini_uncertanity=0.1):
        self.model=Transforms(dh=dh,ini_T=ini_T)
        self.ukf=UKF(dim_x=dim_x,dim_z=dim_z,fx=self.fx,hx=self.hx,dt=0.01,points=MerweScaledSigmaPoints(n=6,alpha=alpha,beta=beta,kappa=kappa))
        self.ukf.x=ini_state
        self.ukf.P *=ini_uncertanity

    def fx(self,x,dt,u=None):
        if u is not None:
            res=x+(self.model.Jacobian(u[:6])@u[6:].reshape(6,1)).flatten()
        else:
            res=x
        return res
    
    def hx(self,x):
        return x
    
    def predict(self,u=None):
        self.ukf.fx = lambda x, dt: self.fx(x, dt, u)
        self.ukf.predict()

    def update(self,z):
        self.ukf.update(z)

    def get_x(self):
        return self.ukf.x.copy()