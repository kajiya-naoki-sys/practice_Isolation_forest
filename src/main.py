import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.tree import plot_tree
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest

iris = load_iris(as_frame=True)
X,y = iris.data, iris.target
df = iris.frame
pca = PCA(n_components=2)
x = df.iloc[:,0:4]
pca.fit(x)
pca.components_.T

X = pca.transform(x)

def iris_plot(s=40,c=y,edgecolor='k',title='iris_dataset', save_path='iris_plot.png'):
    
    fig = plt.figure(figsize=(8,4),dpi=100)

    ax = fig.add_subplot(1,1,1)
    ax.scatter(X[:,0],y=X[:,1],s=s,c=c,edgecolor=edgecolor)
    
    ax.xaxis.set_tick_params(direction="out", labelsize=16, width=3, pad=10)
    ax.yaxis.set_tick_params(direction="out", labelsize=16, width=3, pad=10)
    
    ax.set_xlabel('pca_component1', fontsize=16, labelpad=20, weight='bold')
    ax.set_ylabel('pca_component2', fontsize=16, labelpad=20, weight='bold')
    
    # グラフのタイトル
    ax.set_title(title, fontsize=16, weight='bold')
    
    # グリッド設定
    ax.grid(True)

    plt.savefig(save_path)
    print(f"グラフを保存しました: {save_path}")

model = IsolationForest(
                            n_estimators=100,
                            max_samples='auto', 
                            contamination=0.05,
                            max_features=2, 
                            bootstrap=False,
                            n_jobs=-1,
                            random_state=1
                        )
model.fit(X)

df['anomaly_label'] = model.predict(X)
df['scores'] = model.decision_function(X)

anomaly_df = df[df.anomaly_label==-1]    
iris_plot(s=40, c=df['anomaly_label'], edgecolor="k", title="Iris dataset", save_path="iris_plot.png")