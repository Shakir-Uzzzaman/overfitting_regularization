import numpy as np
def compute_cost_linear_reg(X,y,w,b,lemda=1):
    m=X.shape[0]
    n=len(w)
    cost=0
    for i in range(m):
        f_wb_i=np.dot(X[i],w)+b
        cost+=(f_wb_i-y[i])**2
    cost=cost/(2*m)
    reg_cost=0
    for j in range(n):
        reg_cost+=w[j]**2
    reg_cost=(lemda/(2*m))*reg_cost
    total_cost=cost+reg_cost
    return total_cost
np.random.seed(1)
X_tmp = np.random.rand(5,6)
y_tmp = np.array([0,1,0,1,0])
w_tmp = np.random.rand(X_tmp.shape[1]).reshape(-1,)-0.5
b_tmp = 0.5
lambda_tmp = 0.7
cost_tmp = compute_cost_linear_reg(X_tmp, y_tmp, w_tmp, b_tmp, lambda_tmp)

print("Regularized cost:", cost_tmp)

def sigmoid(z):
    g= g= 1/(1+np.exp(-z))
    return g

def compute_cost_logistic_reg(X,y,w,b,lemda=1):
    m,n=X.shape
    cost=0
    for i in range(m):
        z_i=np.dot(X[i],w)+b
        f_wb_i=sigmoid(z_i)
        cost+=-y[i]*np.log(f_wb_i) - (1-y[i])*np.log(1-f_wb_i)
    cost=cost/m
    reg_cost=0
    for j in range(n):
        reg_cost+=w[j]**2
    reg_cost=(lemda/(2*m))*reg_cost
    total_cost=reg_cost+cost
    return total_cost
np.random.seed(1)
X_tmp = np.random.rand(5,6)
y_tmp = np.array([0,1,0,1,0])
w_tmp = np.random.rand(X_tmp.shape[1]).reshape(-1,)-0.5
b_tmp = 0.5
lambda_tmp = 0.7
cost_tmp = compute_cost_logistic_reg(X_tmp, y_tmp, w_tmp, b_tmp, lambda_tmp)

print("Regularized cost:", cost_tmp)

def compute_gradient_linear_reg(X,y,w,b,lemda):
    m,n=X.shape
    dj_dw=np.zeros((n,))
    dj_db=0
    for i in range(m):
        f_wb_i=np.dot(X[i],w)+b
        err=f_wb_i-y[i]
        for j in range(n):
            dj_dw[j]+=err*X[i,j]
        dj_db+=err
    dj_dw=dj_dw/m
    dj_db=dj_db/m
    for j in range(n):
            dj_dw[j]+=(lemda/m)*w[j]
    
    return dj_dw,dj_db
np.random.seed(1)
X_tmp = np.random.rand(5,3)
y_tmp = np.array([0,1,0,1,0])
w_tmp = np.random.rand(X_tmp.shape[1])
b_tmp = 0.5
lambda_tmp = 0.7
dj_dw_tmp, dj_db_tmp =  compute_gradient_linear_reg(X_tmp, y_tmp, w_tmp, b_tmp, lambda_tmp)

print(f"dj_db: {dj_db_tmp}", )
print(f"Regularized dj_dw:\n {dj_dw_tmp.tolist()}", )

def compute_gradient_logistic_reg(X,y,w,b,lemda):
    m,n=X.shape
    dj_dw=np.zeros((n,))
    dj_db=0
    for i in range(m):
        z_i=np.dot(X[i],w)+b
        f_wb_i=sigmoid(z_i)
        err=f_wb_i-y[i]
        for j in range(n):
            dj_dw[j]=dj_dw[j]+err*X[i,j]
        dj_db+=err
    dj_dw=dj_dw/m
    dj_db=dj_db/m
    for j in range(n):
        dj_dw[j]=dj_dw[j]+(lemda/m)*w[j]
    return dj_dw,dj_db
np.random.seed(1)
X_tmp = np.random.rand(5,3)
y_tmp = np.array([0,1,0,1,0])
w_tmp = np.random.rand(X_tmp.shape[1])
b_tmp = 0.5
lambda_tmp = 0.7
dj_dw_tmp, dj_db_tmp =  compute_gradient_logistic_reg(X_tmp, y_tmp, w_tmp, b_tmp, lambda_tmp)

print(f"dj_db: {dj_db_tmp}", )
print(f"Regularized dj_dw:\n {dj_dw_tmp.tolist()}", )












        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    