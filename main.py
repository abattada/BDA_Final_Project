import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

file_path_prefix = '/kaggle/input/dataset/'
target_type = ["private", "public"]

for type in target_type:
    
    df = pd.read_csv(file_path_prefix + type + "_data.csv")
    dimension = df.shape[1]
    ids = df['id']
    features = df.drop(columns=['id'])
    
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(features)
    
    kmeans = KMeans(n_clusters=4*dimension -1, random_state=0)
    labels = kmeans.fit_predict(data_scaled)

    # 降維到2D
    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(data_scaled)

    # 做成 DataFrame
    vis_df = pd.DataFrame()
    vis_df['PCA1'] = data_pca[:, 0]
    vis_df['PCA2'] = data_pca[:, 1]
    vis_df['Cluster'] = labels.astype(str)

    # 繪圖
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=vis_df, x='PCA1', y='PCA2', hue='Cluster', palette='Set2')
    plt.title(f'{type.capitalize()} Data - PCA Cluster Visualization')
    plt.legend(title='Cluster')
    plt.show()

    result = pd.DataFrame({'id': ids, 'label': labels})
    result.to_csv( type +'_submission.csv', index=False)
