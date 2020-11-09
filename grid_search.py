#!/usr/bin/env python3
# Marcos del Cueto
import math
import random
import statistics
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.mplot3d import Axes3D
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
######################################################################################################
def main():
    # Create {x1,x2,f} dataset every 1.0 from -10 to 10, with a noise of +/- 2
    x1,x2,f=generate_data(-10,10,1.0,0.5)
    # Plot data
    plot_data(x1,x2,f)
    # Prepare X and y for KRR
    X,y = prepare_data_to_KRR(x1,x2,f)
    # Create hyperparams grid
    graph_x,graph_y,graph_z = create_hyperparams_grid(X,y)
    # Plot hyperparams_grid
    plot_hyperparams_grid(graph_x,graph_y,graph_z)
######################################################################################################
def generate_data(xmin,xmax,Delta,noise):
    # Calculate f=sin(x1)+cos(x2)
    x1 = np.arange(xmin,xmax+Delta,Delta)   # generate x1 values from xmin to xmax
    x2 = np.arange(xmin,xmax+Delta,Delta)   # generate x2 values from xmin to xmax
    x1, x2 = np.meshgrid(x1,x2)             # make x1,x2 grid of points
    f = np.sin(x1) + np.cos(x2)             # calculate for all (x1,x2) grid
    # Add random noise to f
    random.seed(2020)                       # set random seed for reproducibility
    for i in range(len(f)):
        for j in range(len(f[0])):
            f[i][j] = f[i][j] + random.uniform(-noise,noise)  # add random noise to f(x1,x2)
    return x1,x2,f
######################################################################################################
def prepare_data_to_KRR(x1,x2,f):
    X = []
    for i in range(len(f)):
        for j in range(len(f)):
            X_term = []
            X_term.append(x1[i][j])
            X_term.append(x2[i][j])
            X.append(X_term)
    y=f.flatten()
    X=np.array(X)
    y=np.array(y)
    return X,y
######################################################################################################
def KRR_function(hyperparams,X,y):
    # Assign hyper-parameters
    alpha_value,gamma_value = hyperparams
    # Initialize lists with final results
    y_pred_total = []
    y_test_total = []
    # Split data into test and train: random state fixed for reproducibility
    kf = KFold(n_splits=10,shuffle=True,random_state=2020)
    # kf-fold cross-validation loop
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # Scale X_train and X_test
        scaler = preprocessing.StandardScaler().fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        # Fit KRR with (X_train_scaled, y_train), and predict X_test_scaled
        KRR = KernelRidge(kernel='rbf',alpha=alpha_value,gamma=gamma_value)
        y_pred = KRR.fit(X_train_scaled, y_train).predict(X_test_scaled)
        # Append y_pred and y_test values of this k-fold step to list with total values
        y_pred_total.append(y_pred)
        y_test_total.append(y_test)
    # Flatten lists with test and predicted values
    y_pred_total = [item for sublist in y_pred_total for item in sublist]
    y_test_total = [item for sublist in y_test_total for item in sublist]
    # Calculate error metric of test and predicted values: rmse
    rmse = np.sqrt(mean_squared_error(y_test_total, y_pred_total))
    r_pearson,_=pearsonr(y_test_total,y_pred_total)
    print('KRR k-fold cross-validation . alpha: %7.6f, gamma: %7.4f, RMSE: %7.4f, r: %7.4f' %(alpha_value,gamma_value,rmse,r_pearson))
    return rmse
######################################################################################################
def create_hyperparams_grid(X,y):
    graph_x = []
    graph_y = []
    graph_z = []
    for alpha_value in np.arange(-5.0,2.0,0.7):
        alpha_value = pow(10,alpha_value)
        graph_x_row = []
        graph_y_row = []
        graph_z_row = []
        for gamma_value in np.arange(0.0,20,2):
            hyperparams = (alpha_value,gamma_value)
            rmse = KRR_function(hyperparams,X,y)
            graph_x_row.append(alpha_value)
            graph_y_row.append(gamma_value)
            graph_z_row.append(rmse)
        graph_x.append(graph_x_row)
        graph_y.append(graph_y_row)
        graph_z.append(graph_z_row)
        print('')
    graph_x=np.array(graph_x)
    graph_y=np.array(graph_y)
    graph_z=np.array(graph_z)
    min_z = np.min(graph_z)
    pos_min_z = np.argwhere(graph_z == np.min(graph_z))[0]
    print('Minimum RMSE: %.4f' %(min_z))
    print('Optimum alpha: %f' %(graph_x[pos_min_z[0],pos_min_z[1]]))
    print('Optimum gamma: %f' %(graph_y[pos_min_z[0],pos_min_z[1]]))
    return graph_x,graph_y,graph_z
######################################################################################################
def plot_data(x1,x2,f):
    fig = plt.figure()
    # Right subplot
    ax = fig.add_subplot(1, 2,2)
    ax.set(adjustable='box', aspect='equal')
    surface=ax.contourf(x1, x2, f, cmap='viridis',zorder=0)
    cbar=plt.colorbar(surface)
    cbar.set_label("$f(x_1,x_2)$",fontsize=16)
    cbar.ax.tick_params(labelsize=14)
    ax.set_xlabel('$x_1$',fontsize=16)
    ax.set_xticks(np.arange(-10,12.5,2.5))
    ax.set_xticklabels(np.arange(-10,12.5,2.5),fontsize=14)
    ax.set_ylabel('$x_2$',fontsize=16)
    ax.set_yticks(np.arange(-10,12.5,2.5))
    ax.set_yticklabels(np.arange(-10,12.5,2.5),fontsize=14)
    # Left subplot
    ax1 = fig.add_subplot(1, 2, 1,projection='3d')
    ax1.plot_surface(x1, x2, f, rstride=1, cstride=1,linewidth=0, antialiased=False,cmap='viridis',zorder=0)
    ax1.set_xlabel('$x_1$',fontsize=16)
    ax1.set_xticks(np.arange(-10,12.5,2.5))
    ax1.set_xticklabels(np.arange(-10,12.5,2.5),fontsize=14)
    ax1.set_ylabel('$x_2$',fontsize=16)
    ax1.set_yticks(np.arange(-10,12.5,2.5))
    ax1.set_yticklabels(np.arange(-10,12.5,2.5),fontsize=14)
    ax1.set_zlabel('$f(x_1,x_2)$',fontsize=16)
    ax1.set_zticks(np.arange(-3,4,1))
    ax1.set_zticklabels(np.arange(-3,4,1),fontsize=14)
    # Separation line
    ax.plot([-0.30, -0.30], [0.0, 1.0], transform=ax.transAxes, clip_on=False,color="black")
    # Plot
    plt.subplots_adjust(wspace = 0.5)
    fig = plt.gcf()
    fig.set_size_inches(21.683, 9.140)
    file_name = 'Figure_data.png'
    plt.savefig(file_name,format='png',dpi=600,bbox_inches='tight')
    plt.close()
######################################################################################################
def plot_hyperparams_grid(graph_x,graph_y,graph_z):
    plt.xscale('log')
    size_list=np.array(graph_z)
    size_list=30/(size_list)**2
    points=plt.scatter(graph_x, graph_y, c=graph_z, cmap='viridis',vmin=0.3,vmax=1.0,marker='o',s=size_list)
    cbar=plt.colorbar(points)
    cbar.set_label("$RMSE$", fontsize=14)
    plt.xlabel(r'$\alpha$',fontsize=14)
    plt.ylabel(r'$\gamma$',fontsize=14)
    plt.yticks(np.arange(0,19,2))
    file_name = 'Figure_hyperparams_grid.png'
    plt.savefig(file_name,format='png',dpi=600)
    plt.close()
######################################################################################################
main()
