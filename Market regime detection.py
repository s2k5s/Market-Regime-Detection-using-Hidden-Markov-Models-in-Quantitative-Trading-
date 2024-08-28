import numpy as np


def take_input():
    A_values=[float(x) for x in input().split()]
    A=np.array(A_values).reshape(2, 2)
    pi_values=[float(x) for x in input().split()]
    pi=np.array(pi_values)
    T=int(input())
    Y_values=[float(input()) for _ in range(T)]
    Y=np.array(Y_values)
    meui=np.array([0, 0])
    sigi=np.array([1, 1])
    
    return A,pi,T,Y,meui,sigi

def normal(y, meu, sigma):
    x = (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * ((y - meu) / sigma) ** 2)
    return x

def alpha(T,pimat,Ymat,meu,sigmasq,Amat):
    alphamat=np.zeros((T,len(pimat)))
    for s in range (len(pimat)):
        alphamat[0][s]=pimat[s]*normal(Ymat[0],meu[s],np.sqrt(sigmasq[s]))
    for t in range(1,T):
        for s in range(len(pimat)):
            for sprime in range (len(pimat)):
                alphamat[t][s]=alphamat[t][s]+alphamat[t-1][sprime]*Amat[sprime][s]*normal(Ymat[t],meu[s],np.sqrt(sigmasq[s]))
          
    return alphamat

def beta(T,pimat,Ymat,meu,sigmasq,Amat):
    betamat=np.zeros((T,len(pimat)))
    for s in range (len(pimat)):
        betamat[T-1][s]=1
    for t in range(T-2,-1,-1):
        for s in range(len(pimat)):
            for sprime in range(len(pimat)):
                betamat[t][s]=betamat[t][s]+betamat[t+1][sprime]*Amat[s][sprime]*normal(Ymat[t+1],meu[sprime],np.sqrt(sigmasq[sprime]))
    
    return betamat

def gamma(matalpha,matbeta):
    gammamat=np.zeros((len(matalpha),len(matalpha[0])))
    
    for t in range(0,len(matalpha)):
        for s in range(len(matalpha[0])):
            norm=0
            for k in range(len(matalpha[0])):
               norm+=matalpha[t][k]*matbeta[t][k] 
               
            
            gammamat[t][s]=(matalpha[t][s]*matbeta[t][s])/norm
    
    return gammamat

def zhi(matalpha, matbeta, Amat, Ymat, pimat, meu, sigmasq):
    T, K = len(matalpha), len(pimat)
    zhimat = np.zeros((T-1 , K, K))
    
    for t in range(T -1):
        for s in range(K):
            for k in range(K):
                norm = 0
                for w in range(K):
                    for q in range(K):
                        norm += matalpha[t][w] * Amat[w][q] * normal(Ymat[t+1], meu[q], np.sqrt(sigmasq[q])) * matbeta[t+1][q]
                        
                zhimat[t][s][k] = (matalpha[t][s] * Amat[s][k] * normal(Ymat[t+1], meu[k], np.sqrt(sigmasq[k])) * matbeta[t+1][k]) / norm
              
    return zhimat


def enhance(xhimatrix,Ymatrix,gammamatrix,meumatrix):
    pistar=np.array(gammamatrix[0])
    row=len(gammamatrix)
    column=len(gammamatrix[0])
    astar=np.zeros((column,column))
    numeratormat=np.sum(xhimatrix,axis=0)
    denominatormat_com=np.sum(gammamatrix,axis=0)
    den_A=denominatormat_com-gammamatrix[T-1]#sum from 0 to T-2
    for i in range(column):
        for j in range(column):
            astar[i][j]=numeratormat[i][j]/den_A[i]
    meunum=np.matmul(Ymatrix,gammamatrix)
    meuden=denominatormat_com
    meustar=np.zeros(column)
    for i in range(column):
        meustar[i]=meunum[i]/meuden[i]
    sigmasqstar=np.zeros(column)
    for i in range(column):
        num=0
        for t in range(row):
            num+=gammamatrix[t][i]*(Ymatrix[t]-meumatrix[i])**2
            
        sigmasqstar[i]=num/meuden[i]
    
    return pistar,astar,meustar,sigmasqstar

def convergance(init_A,init_meu,init_sigmasq,init_pi,epsilon,Y,T):
    A0=init_A
    pi0=init_pi
    meu0=init_meu
    sigmasq0=init_sigmasq
    alpha0=alpha(T,pi0,Y,meu0,sigmasq0,A0)
    beta0=beta(T,pi0,Y,meu0,sigmasq0,A0)
    gamma0=gamma(alpha0,beta0)
    xie0=zhi(alpha0,beta0,A0, Y,pi0,meu0, sigmasq0)
    pistar,astar,meustar,sigmasqstar=enhance(xie0,Y,gamma0,meu0)
    
    while(np.max(np.abs(np.array(pistar-pi0)))>epsilon or np.max(np.abs(np.array(astar-A0)))>epsilon or np.max(np.abs(np.array(meustar-meu0)))>epsilon or np.max(np.abs(np.array(sigmasqstar-sigmasq0)))>epsilon):

        meu0=meustar
        sigmasq0=sigmasqstar
        A0=astar
        pi0=pistar
        alpha0=alpha(T,pi0,Y,meu0,sigmasq0,A0)
        beta0=beta(T,pi0,Y,meu0,sigmasq0,A0)
        gamma0=gamma(alpha0,beta0)
        
        xie0=zhi(alpha0,beta0,A0, Y,pi0,meu0, sigmasq0)
        pistar,astar,meustar,sigmasqstar=enhance(xie0,Y,gamma0,meu0)
        
    return pistar,astar,meustar,sigmasqstar

def regime(gammaf):
    return np.argmax(gammaf,axis=1)

A, pi, T, Y ,meui,sigi= take_input()

pif,Af,meuf,sigf=convergance(A,meui,sigi,pi,1e-8,Y,T)

alphaf=alpha(T,pif,Y,meuf,sigf,Af)
betaf=beta(T,pif,Y,meuf,sigf,Af)
gammaf=gamma(alphaf,betaf)
xief=zhi(alphaf,betaf,Af,Y,pif,meuf,sigf)
market_regime=regime(gammaf)

bullstate=np.argmin(meuf)
bearstate=1-bullstate
for i in range(T):
    
    if(market_regime[i]==bullstate):
        print("Bear")
    else:
        print("Bull")
