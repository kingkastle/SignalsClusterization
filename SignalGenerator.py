'''
Created on 05/11/2015

@author: rafaelcastillo
'''

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import pandas
import random
import numpy as np
from sklearn import preprocessing
from sklearn.cluster import KMeans
import os


def create_cos(number_graphs,length,amp):
    # This function is used to generate cos-like graphs for testing
    # number_graphs: to plot, the minimum number is 2
    # length: number of points included in the x axis
    # amp: Y domain modifications to draw different shapes
    if number_graphs < 2: number_graphs = 2
    x = np.arange(length)
    amp = np.pi*amp
    xx = np.linspace(np.pi*0.3*amp, -np.pi*0.3*amp, length)
    for i in range(number_graphs):
        iterable = (2*np.cos(x) + random.random()*2.5 for x in xx)
        y = np.fromiter(iterable, np.float)
        if i == 0: 
            yfinal =  y
            continue
        yfinal = np.vstack((yfinal,y))
    return x,yfinal

def create_random(number_graphs,length,amp):
    # This function is used to generate cos-kind graphs for testing
    # number_graphs: to plot, the minimum number is 2
    # length: number of points included in the x axis
    # amp: Y domain modifications to draw different shapes
    x = np.arange(length)
    if number_graphs < 2: number_graphs = 2
    for i in range(number_graphs):
        iterable = (random.random()*amp for x in x)
        y = np.fromiter(iterable, np.float)
        if i == 0: 
            yfinal =  y
            continue
        yfinal = np.vstack((yfinal,y))
    return x,yfinal

def create_photo(npdata):
    # This function transform a numpy array to a dataframe to generate the heatmap and 3D visualization in function threeD_plot
    length = npdata[0].size
    df = pandas.DataFrame(np.zeros((101, length)), columns = [str(x + 1) for x in range(length)])
    for index, x in np.ndenumerate(npdata):
        df.iloc[int(x*100),index[1] - 1] += 1
    df = df.sort_index(ascending=False)
    return df


def CreateClusters(ClustersDefinition):
    # This function generates the dataset object including all the signals specified in ClusterGenerator
    # and with the defined number of points.
    # example of ClustersDefinition = {'cos_0':[10,24,1],'cos_1':[80,24,15],'random':[10,24,3]}
    data = pandas.DataFrame()
    for item in ClustersDefinition:
        params = ClustersDefinition[item]
        if 'cos'in item:
            x,y = create_cos(params[0],params[1],params[2])
            data_cluster = pandas.DataFrame(y)
        else:
            x,y = create_random(params[0],params[1],params[2])
            data_cluster = pandas.DataFrame(y)
        data = pandas.concat([data,data_cluster])  
    return data

def ZValues(x,y,df,smooth):
    # this is an auxiliary function for threeD_plot to smooth the shape of the data.
    result = 0
    numberPoints = 0
    xval = np.linspace(x- smooth,x+smooth,smooth*2+1)
    yval = np.linspace(y- smooth,y+smooth,smooth*2+1)
    for x_item in xval:
        if x_item < 0 or x_item >= df.shape[0]: continue
        for y_item in yval:
            if y_item < 0 or y_item >= df.shape[1]: continue
            result += df.iloc[x,y]
            numberPoints += 1
    return result/float(numberPoints)

def plot_df(data,normalized):
    # This represents the dataset generated using in the X-axis the length defined in the cos/random functions and Y-Axis the functions values.
    # Labels are set to 'hour' and 'Consumption' since this project is orientated to clusterize daily Electricity/Gas Consumptions
    if normalized == 'NO': 
        title = 'Raw data'
    else:
        title = 'Data Normalized'
    num_rows = data.shape[0]
    x_values = np.arange(data.shape[1])
    fig, ax = plt.subplots()
    for i in range(num_rows):
        ax.plot(x_values, data.iloc[i])
    ax.set_ylabel('Consumption')
    ax.set_xlabel('hour')
    ax.set_title(title)
    plt.show()
    #plt.savefig(u'/home/rafaelcastillo/workspace/Beeva/GasNatural/signals.png', bbox_inches='tight')#.show()
    
def normalize_data(data):
    # Clustering like KMeans calls for scaled features.
    data = data.astype('float')
    std_scale = preprocessing.MinMaxScaler().fit(data.transpose())
    df_std = std_scale.transform(data.transpose())
    return pandas.DataFrame(np.transpose(df_std))

def threeD_plot(data3D):
    # This is a 3D representation of the dataset and a heatmap. 
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = np.arange(15,data3D.shape[0] - 15)
    y = np.arange(1,data3D.shape[1] - 1)
    X, Y = np.meshgrid(x, y)
    zs = np.array([ZValues(x,y,data3D,3) for x,y in zip(np.ravel(X), np.ravel(Y))])
    Z = zs.reshape(X.shape)
    ax.plot_surface(X, Y, Z,rstride=1, cstride=1, cmap=cm.coolwarm,linewidth=0, antialiased=False)
    ax.set_xlabel('')
    ax.set_ylabel('Hours')
    ax.set_zlabel('Height')
    plt.show()
    
    imgplot = plt.imshow(data3D, cmap="hot",aspect='auto')
    plt.colorbar()
    plt.show()
    
def perform_PCA(data,NumberDimensions):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=NumberDimensions, whiten=True).fit(data.transpose())
    PCAData = pca.transform(data.transpose())
    fig, ax = plt.subplots()
    fig.tight_layout()
    x_values = np.arange(24)
    y_values = PCAData
    ax.plot(x_values, y_values)
    ax.set_title('PCA Explained Variance {0}'.format(pca.explained_variance_ratio_))
    ax.set_ylabel('Consumption')
    ax.set_xlabel('hour')
    plt.show()
    
def perform_DBSCAN(data,DBSCAN_Object):
    from sklearn.cluster import DBSCAN
    ClusterData = DBSCAN_Object.fit(data).labels_
    NumberClusters = np.amax(ClusterData)
    if NumberClusters == -1:
        print "No clusters identified"
        return
    x_values = np.arange(24)
    fig, ax = plt.subplots()
    fig.tight_layout()
    colors = ['red','green','yellow']
    num_rows = data.shape[0]
    for i in range(num_rows):
        if ClusterData[i] == -1: continue
        ax.plot(x_values, data.iloc[i],color = colors[ClusterData[i]],alpha=0.3,lw=1)
    for i,item in enumerate(range(NumberClusters)):
        ax.plot(x_values, data.iloc[np.where(ClusterData == item)].mean(),color = colors[i],alpha=1,lw=4)
    ax.set_title('DBSCAN')
    ax.set_ylabel('Consumption')
    ax.set_xlabel('hour')
    plt.show()
    
def perform_KMEANS(data,KMEANS_Object,plotCluster = True):
    Clusters = KMEANS_Object.fit(data)
    if np.unique(Clusters.labels_).size == 1:
        print "Only 1 cluster to plot, no figures are generated"
        return Clusters
    if plotCluster: draw_clusters(data,Clusters)
    return Clusters

def find_KMEANS_number_clusters(data,maxclusters = 20):
    from sklearn.cluster import KMeans
    results = pandas.DataFrame(columns=['NumberClusters','inertia'])
    n_row = 0
    for i in range(1,maxclusters):
        KMEANS_Object = KMeans(n_clusters=i,n_init=1000,max_iter=500,n_jobs=-1)
        results.loc[n_row] = [i,perform_KMEANS(data,KMEANS_Object,plotCluster = False).inertia_]
        n_row += 1
    results['inertia'].plot()
    plt.ylabel('inertia')
    plt.xlabel('number of clusters')
    plt.show()
    plt.savefig('InertiaAnalysis.png', bbox_inches='tight')
    

def draw_clusters(data,Clusters):
    # Determine the number of clusters, number of values in the x axis and number of signals:
    x_values = np.arange(data.shape[1])
    num_rows = data.shape[0]
    NumberClusters = np.unique(Clusters.labels_).size
    # To differenciate the different clusters, it is used the colormap jet:
    cmap = plt.cm.jet
    cmaplist = [cmap(i) for i in range(cmap.N)]
    steps = cmap.N/NumberClusters
    colors = [cmaplist[x] for x in range(0,cmap.N,steps)]
    # Draw all clusters together
    fig2, ax = plt.subplots()
    for i in range(num_rows):
        ax.plot(x_values, data.iloc[i],color = colors[Clusters.labels_[i]],alpha=0.3,lw=1)
    for i,item in enumerate(range(NumberClusters)):
        ax.plot(x_values, Clusters.cluster_centers_[i],color = colors[i],alpha=1,lw=4)
    ax.set_title('KMEANS',fontsize=9)
    ax.set_ylabel('Consumption',fontsize=9)
    ax.set_xlabel('hour',fontsize=9)
    ax.set_ylim([0,1])
    ax.set_xlim([0,data.shape[1]])
    ax.tick_params(axis='both', which='major', labelsize=9)
    plt.show()
    fig2.savefig('Clusters_Together.png', bbox_inches='tight')
    # Draw all clusters separately
    fig, axarr = plt.subplots(NumberClusters, sharex=True)
    for i,item in enumerate(range(NumberClusters)):
        for j in range(num_rows):
            if Clusters.labels_[j] != i: continue
            axarr[i].plot(x_values, data.iloc[j],color = colors[Clusters.labels_[j]],alpha=0.3,lw=1)
        axarr[i].plot(x_values, data.iloc[np.where(Clusters.labels_ == item)].mean(),color = colors[i],alpha=1,lw=4)
        axarr[i].set_title('Cluster:{0} | Elements Included:{1}'.format(i,np.where(Clusters.labels_ == item)[0].size),fontsize=9)
        axarr[i].set_ylim([0,1])
        axarr[i].set_xlim([0,data.shape[1]])
        axarr[i].tick_params(axis='both', which='major', labelsize=9)
    plt.show()
    fig.savefig('Clusters_Separately.png', bbox_inches='tight')
    
def characterize_clusters(data,Clusters,plotCluster = True):
    # Important: data is the real data in a dataframe where columns are the different hours.
    # This df object includes max,mean,min,std,se values per hour per cluster
    Clusters_results = pandas.DataFrame(columns=['hour','Cluster','Max','Min','Mean','STD','SE'])
    num_rows = 0
    # Get the number of columns in the dataset
    x_values = np.arange(data.shape[1])
    NumberClusters = np.unique(Clusters.labels_).size
    for i in range(NumberClusters):
        for j in x_values:
            df_temp = data[j].iloc[np.where(Clusters.labels_ == i)]
            Clusters_results.loc[num_rows] = [j,i,df_temp.max(),df_temp.min(),df_temp.mean(),df_temp.std(),df_temp.std()/(df_temp.shape[0])**(0.5)]
            num_rows += 1
    if plotCluster:
        # To differenciate the different clusters, it is used the colormap jet:
        cmap = plt.cm.jet
        cmaplist = [cmap(i) for i in range(cmap.N)]
        steps = cmap.N/NumberClusters
        colors = [cmaplist[x] for x in range(0,cmap.N,steps)]
        fig2, ax = plt.subplots()
        for i,item in enumerate(range(NumberClusters)):
            #if i!= 2: continue
            y_values = Clusters_results['Mean'][(Clusters_results['Cluster'] == i)]
            y_values_plus_STD = Clusters_results['Mean'][(Clusters_results['Cluster'] == i)] + Clusters_results['SE'][(Clusters_results['Cluster'] == i)]
            y_values_minus_STD = Clusters_results['Mean'][(Clusters_results['Cluster'] == i)] - Clusters_results['SE'][(Clusters_results['Cluster'] == i)]
            ax.plot(x_values, y_values,color = colors[i],alpha=1,lw=4)
            ax.fill_between(x_values, y_values, y_values_plus_STD, facecolor=colors[i], alpha=0.4)
            ax.fill_between(x_values, y_values_minus_STD, y_values,facecolor=colors[i], alpha=0.4)
        ax.set_title('Cluster: Media +/- Error Estandar ',fontsize=9)
        ax.set_ylabel('Consumption',fontsize=9)
        ax.set_xlabel('hour',fontsize=9)
        ax.set_ylim(bottom=1)
        ax.set_xlim([0,data.shape[1]])
        ax.tick_params(axis='both', which='major', labelsize=9)
        plt.show()
        fig2.savefig('Real_Clusters.png', bbox_inches='tight')
    return Clusters_results
    
    
    
def perform_MinBatckKMEANS(data,MiniKMEANS_Object):
    from sklearn.cluster import KMeans
    Clusters = MiniKMEANS_Object.fit(data)
    ClusterData = Clusters.predict(data)
    NumberClusters = MiniKMEANS_Object.get_params()['n_clusters']
    x_values = np.arange(24)
    fig, ax = plt.subplots()
    fig.suptitle('KMEANS')
    cmap = plt.cm.jet
    cmaplist = [cmap(i) for i in range(cmap.N)]
    steps = cmap.N/NumberClusters
    colors = [cmaplist[x] for x in range(0,cmap.N,steps)]
    num_rows = data.shape[0]
    for i in range(num_rows):
        ax.plot(x_values, data.iloc[i],color = colors[ClusterData[i]],alpha=0.3,lw=1)
    for i,item in enumerate(range(NumberClusters)):
        ax.plot(x_values, data.iloc[np.where(ClusterData == item)].mean(),color = colors[i],alpha=1,lw=4)
    ax.set_title('KMEANS',fontsize=9)
    ax.set_ylabel('Consumption',fontsize=9)
    ax.set_xlabel('hour',fontsize=9)
    plt.show()
    
def main():
    # This is a working example:
    ClustersDefinition = {'cos_0':[10,24,1],'cos_1':[80,24,15],'random':[10,24,3]}
    data = CreateClusters(ClustersDefinition)
    plot_df(data,'NO')
    data_norm = normalize_data(data)
    plot_df(data_norm,'YES')
    find_KMEANS_number_clusters(data_norm,maxclusters = 20)
    KMEANS_Object = KMeans(n_clusters=3,n_init=1000,max_iter=500,n_jobs=-1)
    perform_KMEANS(data_norm,KMEANS_Object,plotCluster = True)
    
    
      
if __name__ == "__main__":
    main()
    
    
    

