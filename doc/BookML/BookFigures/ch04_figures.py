from common import *
rng=np.random.default_rng(2024)

# 1. GD on an ill-conditioned quadratic: contours + paths
lam=np.array([1.0,12.0]); H=np.diag(lam); kappa=lam.max()/lam.min()
def path(eta,gam=0.0,steps=60,x0=np.array([9.0,1.6])):
    x=x0.copy(); v=np.zeros(2); P=[x.copy()]
    for _ in range(steps):
        g=H@x
        if gam: v=gam*v+eta*g; x=x-v
        else: x=x-eta*g
        P.append(x.copy())
        if np.abs(x).max()>1e4: break
    return np.array(P)
gx,gy=np.meshgrid(np.linspace(-10,10,300),np.linspace(-3,3,300))
Z=0.5*(lam[0]*gx**2+lam[1]*gy**2)
fig,axes=plt.subplots(1,3,figsize=(10.5,3.0),sharex=True,sharey=True)
for ax,(eta,gam,t) in zip(axes,[(0.02,0.0,r"$\eta=0.02$ (too small)"),
                                (0.15,0.0,r"$\eta=0.15$ (near the bound)"),
                                (0.15,0.85,r"$\eta=0.15,\ \gamma=0.85$")]):
    ax.contour(gx,gy,Z,levels=np.logspace(-1,2.4,16),colors="0.75",linewidths=0.7)
    P=path(eta,gam); ax.plot(P[:,0],P[:,1],"o-",ms=2.5,lw=1.1,color="crimson")
    ax.plot(0,0,"k*",ms=10); ax.set_title(t,fontsize=9.5); ax.set_xlabel("$x_0$")
axes[0].set_ylabel("$x_1$")
fig.suptitle(r"$\kappa=%.0f$;  stability bound $2/\lambda_{\max}=%.3f$"%(kappa,2/lam.max()),fontsize=10,y=1.04)
save(fig,4,"gd_paths_conditioning")

# 2. stability bound and convergence rate
lams=np.array([0.05,1.0,5.0]); Hd=np.diag(lams)
lmax,lmin=lams.max(),lams.min(); kap=lmax/lmin
etas=np.linspace(0.005,0.45,180); iters=[]
for eta in etas:
    e=np.ones(3); n=0
    while np.abs(e).max()>1e-6 and n<20000:
        e=e-eta*(Hd@e); n+=1
        if np.abs(e).max()>1e8: n=20000; break
    iters.append(n)
fig,ax=plt.subplots(figsize=(5.4,3.4))
ax.semilogy(etas,iters,lw=1.8)
ax.axvline(2/lmax,ls="--",c="crimson"); ax.text(2/lmax*0.99,4e3,r"$2/\lambda_{\max}$",rotation=90,color="crimson",fontsize=9,ha="right")
ax.axvline(2/(lmax+lmin),ls=":",c="k"); ax.text(2/(lmax+lmin)*1.03,20,r"$\eta^*$",fontsize=9)
ax.set_xlabel(r"learning rate $\eta$"); ax.set_ylabel("iterations to $10^{-6}$")
ax.set_title(r"$\kappa=%.0f$"%kap,fontsize=10)
save(fig,4,"learning_rate_bound")

# 3. momentum: rate vs kappa
ks=np.logspace(0.3,4,60)
fig,ax=plt.subplots(figsize=(5.4,3.4))
ax.loglog(ks,1/np.log(1/((ks-1)/(ks+1))),label=r"gradient descent, $\propto\kappa$")
ax.loglog(ks,1/np.log(1/((np.sqrt(ks)-1)/(np.sqrt(ks)+1))),label=r"momentum, $\propto\sqrt{\kappa}$")
ax.set_xlabel(r"condition number $\kappa$"); ax.set_ylabel("iterations (arb. units)"); ax.legend()
save(fig,4,"momentum_rate")

# 4. optimiser comparison on the least-squares problem
n=100; xx=2.0*rng.random((n,1)); yy=4.0+3.0*xx+rng.normal(size=(n,1)); X=np.c_[np.ones((n,1)),xx]
te=np.linalg.pinv(X.T@X)@X.T@yy
def cost(t): return float(np.mean((X@t-yy)**2))
def run(method,eta,epochs=100,seed=7):
    r=np.random.default_rng(seed); p=X.shape[1]; th=r.normal(size=(p,1))
    ch=np.zeros((p,1)); rr=np.zeros((p,1)); m=np.zeros((p,1)); t=0; hist=[]
    for e in range(epochs):
        idx=r.permutation(n)
        for b in [idx[i:i+10] for i in range(0,n,10)]:
            t+=1; Xb,yb=X[b],yy[b]; g=(2.0/len(b))*Xb.T@(Xb@th-yb)
            if method=="plain": u=eta*g
            elif method=="momentum": ch=eta*g+0.9*ch; u=ch
            elif method=="AdaGrad": rr+=g*g; u=eta*g/(np.sqrt(rr)+1e-8)
            elif method=="RMSProp": rr=0.99*rr+0.01*g*g; u=eta*g/(np.sqrt(rr)+1e-8)
            elif method=="Adam":
                m=0.9*m+0.1*g; rr=0.999*rr+0.001*g*g
                u=eta*(m/(1-0.9**t))/(np.sqrt(rr/(1-0.999**t))+1e-8)
            th-=u
        hist.append(cost(th))
    return hist
fig,axes=plt.subplots(1,2,figsize=(9.4,3.4),sharey=True)
for ax,eta in zip(axes,[0.01,0.5]):
    for meth in ["plain","momentum","AdaGrad","RMSProp","Adam"]:
        ax.semilogy(np.array(run(meth,eta))-cost(te)+1e-12,label=meth,lw=1.4)
    ax.set_xlabel("epoch"); ax.set_title(rf"$\eta={eta}$",fontsize=10)
axes[0].set_ylabel(r"$C(\theta)-C(\hat\theta)$"); axes[0].legend(fontsize=8)
save(fig,4,"optimiser_comparison")
