from common import *
import warnings; warnings.filterwarnings("ignore")
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split,KFold,cross_val_score
from mpl_toolkits.mplot3d import Axes3D

def franke(x,y):
    return (0.75*np.exp(-(0.25*(9*x-2)**2)-0.25*((9*y-2)**2))+0.75*np.exp(-((9*x+1)**2)/49.0-0.1*(9*y+1))
            +0.50*np.exp(-(9*x-7)**2/4.0-0.25*((9*y-3)**2))-0.20*np.exp(-(9*x-4)**2-(9*y-7)**2))
def dm(x,y,deg):
    x,y=np.ravel(x),np.ravel(y); c=[]
    for t in range(deg+1):
        for b in range(t+1): c.append(x**(t-b)*y**b)
    return np.column_stack(c)

# 1. Franke surface
gx,gy=np.meshgrid(np.linspace(0,1,80),np.linspace(0,1,80))
fig=plt.figure(figsize=(9.2,3.4))
ax=fig.add_subplot(121,projection="3d")
ax.plot_surface(gx,gy,franke(gx,gy),cmap="viridis",linewidth=0,antialiased=True)
ax.set_xlabel("$x$"); ax.set_ylabel("$y$"); ax.set_zlabel("$f$"); ax.set_title("Franke function",fontsize=10)
ax2=fig.add_subplot(122)
r=np.random.default_rng(2024); xs,ys=r.random(100),r.random(100)
zs=franke(xs,ys)+0.2*r.normal(size=100)
c=ax2.contourf(gx,gy,franke(gx,gy),levels=20,cmap="viridis")
ax2.scatter(xs,ys,c="w",s=14,edgecolors="k",linewidths=0.5)
ax2.set_xlabel("$x$"); ax2.set_ylabel("$y$"); ax2.set_title(r"$n=100$ samples, $\sigma=0.2$",fontsize=10)
fig.colorbar(c,ax=ax2,shrink=0.85)
save(fig,3,"franke_function")

# 2. the central experiment: test MSE vs degree for OLS/Ridge/Lasso
x,y=r.random(100),r.random(100); z=franke(x,y)+0.2*np.random.default_rng(7).normal(size=100)
maxd=14; lam=np.logspace(-5,1,25); kf=KFold(5,shuffle=True,random_state=2024)
res={"OLS":[],"Ridge":[],"Lasso":[]}; nnz=[]
for d in range(1,maxd+1):
    X=dm(x,y,d)[:,1:]
    Xtr,Xte,ztr,zte=train_test_split(X,z,test_size=0.2,random_state=2024)
    p=make_pipeline(StandardScaler(),LinearRegression()); p.fit(Xtr,ztr)
    res["OLS"].append(np.mean((zte-p.predict(Xte))**2))
    for nm,M in (("Ridge",Ridge),("Lasso",Lasso)):
        cv=[-cross_val_score(make_pipeline(StandardScaler(),M(alpha=l,max_iter=5000)),Xtr,ztr,cv=kf,
                             scoring="neg_mean_squared_error").mean() for l in lam]
        b=make_pipeline(StandardScaler(),M(alpha=lam[int(np.argmin(cv))],max_iter=5000)); b.fit(Xtr,ztr)
        res[nm].append(np.mean((zte-b.predict(Xte))**2))
        if nm=="Lasso": nnz.append((int(np.sum(b[-1].coef_!=0)),X.shape[1]))
fig,axes=plt.subplots(1,2,figsize=(9.4,3.4))
for k,mk in zip(res,["o-","s-","^-"]): axes[0].semilogy(range(1,maxd+1),res[k],mk,label=k)
axes[0].axhline(0.04,ls="--",c="k",lw=1); axes[0].text(1.2,0.045,r"$\sigma^2$",fontsize=9)
axes[0].set_xlabel("polynomial degree"); axes[0].set_ylabel("test MSE"); axes[0].legend()
axes[0].set_title(r"$n=100$, $\sigma=0.2$",fontsize=10)
axes[1].plot(range(1,maxd+1),[a for a,b in nnz],"o-",label="Lasso: non-zero")
axes[1].plot(range(1,maxd+1),[b for a,b in nnz],"k--",label="total coefficients $p$")
axes[1].set_xlabel("polynomial degree"); axes[1].set_ylabel("number of coefficients"); axes[1].legend()
axes[1].set_title("the Lasso selects",fontsize=10)
save(fig,3,"franke_model_selection")

# 3. coefficient paths on the toy problem
X=np.array([[2.,0.],[0.,1.],[0.,0.]]); yv=np.array([4.,2.,3.]); n=3
lams=np.logspace(-3,2,200)
rp=np.array([Ridge(alpha=l,fit_intercept=False).fit(X,yv).coef_ for l in lams])
lp=np.array([Lasso(alpha=l/(2*n),fit_intercept=False,max_iter=100000).fit(X,yv).coef_ for l in lams])
fig,axes=plt.subplots(1,2,figsize=(9.0,3.2),sharey=True)
for k in range(2):
    axes[0].semilogx(lams,rp[:,k],label=rf"$\theta_{k}$")
    axes[1].semilogx(lams,lp[:,k],label=rf"$\theta_{k}$")
axes[0].semilogx(lams,8/(4+lams),"k:",lw=1); axes[0].semilogx(lams,2/(1+lams),"k:",lw=1)
axes[1].axvline(4,ls=":",c="gray"); axes[1].axvline(16,ls=":",c="gray")
axes[1].text(4,1.7,r"$\lambda=4$",fontsize=8,rotation=90); axes[1].text(16,1.7,r"$\lambda=16$",fontsize=8,rotation=90)
axes[0].set_title("Ridge (dotted: analytical)",fontsize=10); axes[1].set_title("Lasso",fontsize=10)
for a in axes: a.set_xlabel(r"$\lambda$"); a.legend()
axes[0].set_ylabel(r"coefficient")
save(fig,3,"ridge_lasso_paths")

# 4. shrinkage factor
sig=np.logspace(-2,1,300)
fig,ax=plt.subplots(figsize=(5.4,3.4))
for l in [0.001,0.01,0.1,1.0]:
    ax.semilogx(sig,sig**2/(sig**2+l),label=rf"$\lambda={l}$")
ax.set_xlabel(r"singular value $\sigma_i$"); ax.set_ylabel(r"shrinkage $\sigma_i^2/(\sigma_i^2+\lambda)$")
ax.legend(); ax.set_ylim(-0.03,1.03)
save(fig,3,"ridge_shrinkage")
