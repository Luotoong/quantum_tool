#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from decimal import Decimal
from qibo import hamiltonians, models
from qibo import hamiltonians
from qibo.symbols import X, Y, Z
from qibo.models import Circuit
from qibo import gates
import matplotlib.pyplot as plt


# In[2]:


def bellst(c,n,m,p):
    c.add(gates.H(n)) 
    c.add(gates.PauliNoiseChannel(n, p/3, p/3, p/3))
    
    c.add(gates.CNOT(n, m))   
    c.add(gates.PauliNoiseChannel(n, p/3, p/3, p/3))
    c.add(gates.PauliNoiseChannel(m, p/3, p/3, p/3))
    return c
#bell 的产生，将|00>初态变成bell

def ibellst(c,n,m,p):

    c.add(gates.CNOT(n, m))
    c.add(gates.PauliNoiseChannel(n, p/3, p/3, p/3))
    c.add(gates.PauliNoiseChannel(m, p/3, p/3, p/3))
    
    c.add(gates.H(n))
    c.add(gates.PauliNoiseChannel(n, p/3, p/3, p/3))
    
    return c
#对演化的末态做个操作，使得bell态的信息容易读取，|00>,|10>,|01>,|11>
#==========================================================================
def RZZ(J,dt,c,q1,q2,p):
    c.add(gates.CNOT(q1, q2 ))
    c.add(gates.PauliNoiseChannel(q1, p/3, p/3, p/3))
    c.add(gates.PauliNoiseChannel(q2, p/3, p/3, p/3)) 
    
    c.add(gates.RZ(q2, theta=-2*J*dt))  
    c.add(gates.PauliNoiseChannel(q2, p/3, p/3, p/3))
    
    c.add(gates.CNOT(q1, q2 ))
    c.add(gates.PauliNoiseChannel(q1, p/3, p/3, p/3))
    c.add(gates.PauliNoiseChannel(q2, p/3, p/3, p/3))
    

    return c
#构造RZZgate======================================================================


# In[3]:


def Hmix_dt_evl_up(c,NL,J,hz,hx,dt,p):
    for i in range(int(NL/2)):
        c=RZZ(J,dt,c,2*i,2*i+1,p)
    for i in range(int((NL-1)/2)):
        c=RZZ(J,dt,c,2*i+1,2*i+2,p) 
        
    for i in range(NL):
        c.add(gates.RZ(i, theta=-2*hz*dt ))
        c.add(gates.PauliNoiseChannel(i, p/3, p/3, p/3))
    for i in range(NL):
        c.add(gates.RX(i, theta=-2*hx*dt ))
        c.add(gates.PauliNoiseChannel(i, p/3, p/3, p/3))
    return c
#===============================================================
def Hmix_dt_evl_down(c,NL,J,hz,hx,dt,p):
    for i in range(int(NL/2)):
        c=RZZ(J,dt,c,NL+2*i,NL+2*i+1,p)
    for i in range(int((NL-1)/2)):
        c=RZZ(J,dt,c,NL+2*i+1,NL+2*i+2,p) 
        
    for i in range(NL):
        c.add(gates.RZ(NL+i, theta=-2*hz*dt ))
        c.add(gates.PauliNoiseChannel(NL+i, p/3, p/3, p/3))
    for i in range(NL):
        c.add(gates.RX(NL+i, theta=-2*hx*dt ))
        c.add(gates.PauliNoiseChannel(NL+i, p/3, p/3, p/3))
    return c


# In[4]:


def time_evl_T(c,NL,J,hz,hx,T,step,p):
    dt=T/step
    for i in range(step):
        c = Hmix_dt_evl_up(c,NL,J,hz,hx,-dt,p)
        c = Hmix_dt_evl_down(c,NL,J,hz,hx,dt,p)
    return c              


# In[5]:



#计算平均算符长度========================
def oplbar(result, NL):

    Lbar=NL
    Lbar2=NL
    A = result.state()

    A00 = np.zeros((4,4))
    A00[0,0] = 1
    for ii in range(NL):

        Lbar= Lbar - np.trace(np.dot(A00,rcdm2(A,ii+1,ii+1+NL)))
        
        
    return Lbar
#计算平均算符长度========================


# In[6]:


def time_evl_get_Lbar(NL,J,hx,hz,T,p):
    step = int(T/0.1)
    c=Circuit(2*NL, density_matrix=True) 
    for i in range(NL):
        c=bellst(c,i,i+NL,p)
    
#=====================================================
    c.add(gates.X(int(NL/2)))    #t=0时的算符\sigma_x
    c.add(gates.PauliNoiseChannel(int(NL/2), p/3, p/3, p/3))
#=====================================================

    c=time_evl_T(c,NL,J,hz,hx,T,step,p)

#===================================================
    for i in range(NL):
        c=ibellst(c,i,i+NL,p)
    
    result = c()
    Lbar = oplbar(result, NL)
    return Lbar
    #print(result)
    #print(c.draw())


# In[7]:
def srcdm(A,n):
    Ln = int( np.log2( A.shape[0]))
    B = 1j*np.zeros((int(A.shape[0]/2),int(A.shape[0]/2)))
    for i in range(int(A.shape[0]/2)):
        for k in range(int(A.shape[0]/2)):
            B[i,k] = A[i,k]+ A[i+2**(Ln-n),k+2**(Ln-n)]
    return B
##约化密度矩阵
# In[8]:

def rcdm2(A,n,m):
    Ln = int( np.log2( A.shape[0]) )
    nn = 0
    for ii in range(Ln):
        nn = nn + 1
    
        if (ii+1) != n:
            if (ii+1) != m:
                A = srcdm(A, nn)
                nn = nn - 1
    return A
            
# In[9]:

import time
time_zstart=time.time() #计时开始

NL=6 #自旋链的长度
J=1
hx=1
hz=0
#dt=0.1
Tmax=10
deltaT=0.5
p=0.00001


# In[8]:


OPLbarf=str('OPLbar')+str(',T=')+str(Tmax)+str(',N=')+str(NL)+str(',hz=')+str(hz)+str(',p=')+str(p)+str('.dat')

file1=open(OPLbarf, mode='a+')



# In[9]:


Lbart = np.zeros((int(Tmax/deltaT)+1))
Lbart[0] = 1
file1.writelines(str(0))
file1.writelines(',')
file1.writelines(str(1))
file1.writelines('\n')
for i in range(int(Tmax/deltaT)):
    T = deltaT*(i+1)
    print(T)
    Lbart[i+1] = time_evl_get_Lbar(NL,J,hx,hz,T,p)
    file1=open(OPLbarf, mode='a+')
    file1.writelines(str(T))
    file1.writelines(',')
    file1.writelines(str(Lbart[i+1]))
    file1.writelines('\n')
    time_zend=time.time()  #计时结束
    print('计算演化花费的时间',time_zend-time_zstart) #s输出时间
    file1.close()
# In[12]:


file1.close()
time_zend=time.time()  #计时结束
print('计算演化花费的时间',time_zend-time_zstart) #s输出时间
plt.plot(Lbart)

