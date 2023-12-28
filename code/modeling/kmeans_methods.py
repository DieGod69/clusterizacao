import os
import warnings 
import numpy as np
import pandas as pd
import seaborn as sns
from paretochart import pareto
import matplotlib.pyplot as plt
from sklearn_extra.cluster import KMedoids
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from yellowbrick.cluster import KElbowVisualizer
from sklearn.metrics import davies_bouldin_score
from scipy.spatial.distance import pdist, squareform

sns. set_style('ticks')
sns.set_context('talk')
warnings.filterwarnings('ignore')

class Mod_KMeans():
    
    
    def __init__(self):
        pass
    
    
    def import_df(self,f_name):
        data = pd.read_csv(
            os.getcwd().replace('/' or '//', r'\\').replace('code', 'data').replace('modeling', 'processed')+ f'\\{f_name}'
            )
        return data
    
    
    
    def clustering_kmeans(self, df, n_cluster):
        model = Pipeline([(
            'kmeans', KMeans(n_clusters = n_cluster,
            init='k-means++',
            max_iter=10000,
            random_state=42)
        )])
        
        cluster_kmeans = model.fit_predict(df)
        return cluster_kmeans
    
    
    def pca_(self, df, n_componet):
        pca = PCA(n_components= n_componet)
        df = pd.DataFrame(pca.fit_transform(df))
        return pareto(pca.explained_variance_ratio_)
    
    
    def best_cluster(self, df, model):
        fig, ax = plt.subplots(1,1,figsize=(16,8))
        visualizer = KElbowVisualizer(model, k=(2,12), distance_metric='euclidean', ax=ax)
        visualizer.fit(df)
        return visualizer.show()

    
    
    def count_clusters(self, df, cluster_column):
        cluster_counts = df[cluster_column].value_counts()
        for cluster, count in cluster_counts.items():
            print(f'A quantidade de países no cluster {cluster} é {count}')
            
            
    def metrics_model(self, df, n_cluster):
        model = KMeans(n_clusters=n_cluster, init='k-means++', max_iter=10000, random_state=10)
        label = model.fit_predict(df)
        print(f'Índice de Devies-Bouldin: {davies_bouldin_score(df,label)}')
        
    
    def report_clustering(self, df, cluster_type):
        for cluster in df[cluster_type].unique():
            cluster_data = df[cluster_type] == cluster
            countries = df.loc[cluster_data, 'country']
            print('_'*100)
            print(f'O cluster {cluster} possui {cluster_data.sum()} países, sendo eles: \n\n {countries.tolist()}')
            print('_'*100)
            
            print(f'Estatística descritiva do cluster {cluster}\n')
            print('_'*100)
            print(df[df[cluster_type]==cluster].describe())
            print('_'*100)
            print(f'Boxplot do cluster {cluster}\n')
            plt.figure(figsize=(12,8))
            sns.boxplot(df[df[cluster_type]==cluster]);
            plt.show()
            
    
    def method_pca(self, df, n_component):
        pca = PCA(n_components = n_component)
        df = pd.DataFrame(pca.fit_transform(df))
        return df
        
        
    def best_point_np(self, df, cluster_name, n_cluster):
        for k in range(n_cluster):
            distance_np = df[df[cluster_name]==k].drop([cluster_name], axis=1)
            cluster_np = np.array(distance_np)
            distances = []
            for i, item1 in enumerate(cluster_np):
                avg_distance = 0
                for j, item2 in enumerate(cluster_np):
                    if i != j:
                        dist = np.sqrt(np.sum((item1 - item2)**2))
                        avg_distance += dist
                avg_distance /= len(cluster_np) -1
                distances.append(avg_distance)
            best_item = np.argmin(distances)
            best_point_np = distance_np.iloc[best_item].name.upper()
            print(f' O país com melhor ponto médio de cluster {k} é: {best_point_np}')
            
            
    def medoids(self, df, cluster_type):
        for i in range(cluster_type):
            df_medoid = df[df[cluster_type]==i].drop([cluster_type], axis=1)
            df_array_medoid = np.array(df_medoid)
            
            KMobj = KMedoids(n_clusters=1).fit(df_array_medoid)
            kmedoid = KMobj.cluster_centers_.flatten().tolist()
            kmedoid = list(map(float, kmedoid))
            
            isin_filter = df_medoid.isin(kmedoid)
            row_filter = isin_filter.sum(axis=1) == len(kmedoid)
            loc_line = df_medoid.loc[row_filter]
            
            print(f'O medóide do cluster {i} é {loc_line.index[0].upper()}')