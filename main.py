import wave
import struct
import sys
import csv
import numpy as np
import pandas as pd
from vmdpy import VMD
import sys, os, os.path
from sklearn.decomposition import PCA
import librosa
import soundfile as sf

N=100000
data, sr = sf.read('input.wav')
df=pd.DataFrame(data)
sig1=df.iloc[:,0]
sig2=df.iloc[:,1]
alpha=2000       # moderate bandwidth constraint
tau=0           # noise-tolerance (no strict fidelity enforcement)
K=5              # 5 modes
DC=0             # no DC part imposed
init=1           # initialize omegas uniformly
tol=1e-7
l=len(sig1)
k=l//N
msum=pd.DataFrame()
nsum=pd.DataFrame()
pca = PCA(n_components=5)
for x in range(k+1):
	if x==k:
		temp1=sig1[k*N:l,]
		temp2=sig2[k*N:l,]
		u,u_hat,omega=VMD(temp1,alpha,tau,K,DC,init,tol)
		u1,u_hat1,omega1=VMD(temp2,alpha,tau,K,DC,init,tol)
		u=u.transpose()
		u1=u1.transpose()
		Xt=pca.fit_transform(u)
		Xt2=pca.fit_transform(u1)
		tcol=Xt[:,0]+Xt[:,1]
		msum=pd.concat([msum,pd.DataFrame(tcol)],axis=0)
		tcol=Xt2[:,0]+Xt2[:,1]
		nsum=pd.concat([nsum,pd.DataFrame(tcol)],axis=0)
		print(x)
	else:
		temp1=sig1[(x*N):(N*(x+1)),]
		temp2=sig2[(x*N):(N*(x+1)),]
		u,u_hat,omega=VMD(temp1,alpha,tau,K,DC,init,tol)
		u1,u_hat1,omega1=VMD(temp2,alpha,tau,K,DC,init,tol)
		u=u.transpose()
		u1=u1.transpose()
		Xt=pca.fit_transform(u)
		Xt2=pca.fit_transform(u1)
		tcol=Xt[:,0]+Xt[:,1]
		msum=pd.concat([msum,pd.DataFrame(tcol)],axis=0)
		tcol=Xt2[:,0]+Xt2[:,1]
		nsum=pd.concat([nsum,pd.DataFrame(tcol)],axis=0)
		print(x)
df3=pd.DataFrame()
df3=pd.concat([pd.DataFrame(msum),pd.DataFrame(nsum)],axis=1)
sf.write('stereo_file1.wav',df3,sr)
print("done")

