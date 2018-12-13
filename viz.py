import numpy as np
import matplotlib.pyplot as plt

class analytic_gaussian_posterior:
    def __init__(self):
        pass
    
    def MV_normal_probability(self,mu,S,x):
        dim = mu.shape[0]
        inner = ((x-mu).T @ np.linalg.inv(S) @ (x-mu))[0,0]
        num = np.exp(-0.5 * inner)
        denom = np.sqrt(np.linalg.det(S) * (2*np.pi)**dim)
        return num/denom

    def get_likelihood(self,contx,query_points):
        dim = contx.shape[-1]
        I = np.eye(dim)
        n = contx.shape[-2]
        V0_inv = (0.5*I)
        S_inv = I
        x_bar = np.mean(contx[:,:,:],axis=1).T

        mu0 = np.zeros((dim,1))
        VN = np.linalg.inv(V0_inv + n * S_inv)
        muN = VN @ (V0_inv @ mu0 + n * S_inv @ x_bar)
        
        num_output = query_points.shape[-2]
        output = np.zeros((num_output))
        for j in range(num_output):
            output[j] = self.MV_normal_probability(muN,np.linalg.inv(S_inv) + VN,query_points[:,j,:].T)
        return output

def plot_likelihood(mu, context_x, model, x_range=[-3,3], y_range=[-3,3], N=101, num_aux=0):
    xx = np.linspace(x_range[0], x_range[1], N)
    yy = np.linspace(y_range[0], y_range[1], N)
    XX, YY = np.meshgrid(xx,yy)
    query_pts = np.stack([XX.flatten(),YY.flatten()], axis=-1)
    query_pts = np.concatenate([query_pts, np.random.randn( *( XX.flatten().shape[0], num_aux) )], axis=-1)
    query_pts = np.expand_dims( query_pts, axis=0 )
    probs = model.get_likelihood(context_x, query_pts)
    ZZ = probs.reshape([N,N])
    m = plt.contourf(XX,YY,ZZ, cmap=plt.cm.Greens)
    h1 = plt.scatter(context_x[0,:,0],context_x[0,:,1],color='r',marker='x',s=30, label='Context')
    h2 = plt.scatter(mu[0][0],mu[0][1], color='blue' ,marker='o',s=30, label='True Mean')
    
    cb = plt.colorbar(m)
    # cb.set_label("$p(y)$")
    
#     plt.legend(handles=[h1,h2])
    
    plt.xlim(x_range)
    plt.ylim(y_range)
    
    
def plot_conditional_likelihood(model, context_x, context_y, x_range=[-5., 5.], y_range=[-2., 2.], N=101, num_aux=0):
    xx = np.linspace(x_range[0], x_range[1], N)
    yy = np.linspace(y_range[0], y_range[1], N)
    XX, YY = np.meshgrid(xx,yy)
    x_q = XX.flatten().reshape([1,-1,1])
    y_q = YY.flatten().reshape([1,-1,1])
    y_q = np.concatenate([y_q, 0.0*np.random.randn( *(1, XX.flatten().shape[0], num_aux) )], axis=-1)
    
    probs = model.get_likelihood(context_x, context_y, x_q, y_q)
    ZZ = probs.reshape([N,N])
    m = plt.contourf(XX,YY,ZZ, cmap=plt.cm.Greens)
    h1 = plt.scatter(context_x[0,:,0],context_y[0,:,0],color='r',marker='x',s=30, label='Context')

    cb = plt.colorbar(m)
    
    plt.xlim(x_range)
    plt.ylim(y_range)