from django.shortcuts import render

# Create your views here.
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.gridspec as gridspec
from collections import Counter
from sklearn import metrics
import numpy as np
import os
from django.conf import settings

def index(request):
    return render(request, 'index.html')

# Original code: https://bit.ly/2TNHBZ5
def plot_data(X, y):
    plt.plot(X[:, 0][y==0], X[:, 1][y==0], 'k.', markersize=2)
    plt.plot(X[:, 0][y==1], X[:, 1][y==1], 'r.', markersize=2)

def plot_centroids(centroids, weights=None, circle_color='w', cross_color='k'):
    if weights is not None:
        centroids = centroids[weights > weights.max() / 10]
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='o', s=30, linewidths=8,
                color=circle_color, zorder=10, alpha=0.9)
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=50, linewidths=50,
                color=cross_color, zorder=11, alpha=1)

def plot_decision_boundaries(clusterer, X, y, resolution=1000, show_centroids=True):
    mins = X.min(axis=0) - 0.1
    maxs = X.max(axis=0) + 0.1
    xx, yy = np.meshgrid(np.linspace(mins[0], maxs[0], resolution),
                         np.linspace(mins[1], maxs[1], resolution))
    Z = clusterer.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]),
                cmap="Pastel2")
    plt.contour(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]),
                linewidths=1, colors='k')
    plot_data(X, y)
    if show_centroids:
        plot_centroids(clusterer.cluster_centers_)

def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)

df = pd.read_csv("creditcard.csv")

def mostrarInfo(request):
    head = df.head(10)
    nc = len(df.columns)
    lcd = len(df)
    tab1 = df["Class"].value_counts()
    list1 = df.isna().any()
    des = df.describe()

    # Representamos gráficamente las características
    features = df.drop("Class", axis=1)

    plt.figure(figsize=(12,32))
    gs = gridspec.GridSpec(8, 4)
    gs.update(hspace=0.8)

    for i, f in enumerate(features):
        ax = plt.subplot(gs[i])
        sns.distplot(df[f][df["Class"] == 1])
        sns.distplot(df[f][df["Class"] == 0])
        ax.set_xlabel('')
        ax.set_title('feature: ' + str(f))

    plt.savefig(os.path.join(settings.BASE_DIR, 'static', 'images', 'decision_boundary.png'))

    # Representación gráfica de dos características
    plt.figure(figsize=(12, 6))
    plt.scatter(df["V10"][df['Class'] == 0], df["V14"][df['Class'] == 0], c="g", marker=".")
    plt.scatter(df["V10"][df['Class'] == 1], df["V14"][df['Class'] == 1], c="r", marker=".")
    plt.xlabel("V10", fontsize=14)
    plt.ylabel("V14", fontsize=14)
    plt.savefig(os.path.join(settings.BASE_DIR, 'static', 'images', 'dos caracteristicas.png'))

    context = {
    'head': head,
    'nc': nc,
    'lcd': lcd,
    'tab1': tab1,
    'list1':list1,
    'des':des,
    }
    return render(request, 'info.html', context)



# _________ preparacion del conjunto de datos _________

def mostrarConjunto(request):
    df = pd.read_csv("creditcard.csv")
    df = df.drop(["Time", "Amount"], axis=1)
    X = df[["V10", "V14"]].copy()
    tab2 = X.head(10)

    # Generamos los clusters para nuestro conjunto de datos sin etiquetar
    from sklearn.cluster import KMeans

    kmeans = KMeans(n_clusters=5, random_state=42)
    clusters = kmeans.fit_predict(X)

    plt.figure(figsize=(12, 6))
    plot_decision_boundaries(kmeans, X.values, df["Class"].values)
    plt.xlabel("V10", fontsize=14)
    plt.ylabel("V14", fontsize=14)
    plt.savefig(os.path.join(settings.BASE_DIR, 'static', 'images', 'plot_decision_boundaries.png'))
    counter = Counter(clusters.tolist())
    bad_counter = Counter(clusters[df['Class'] == 1].tolist())

    for key in sorted(counter.keys()):
        z = ("Label {0} has {1} samples - {2} are malicious samples".format(
            key, counter[key], bad_counter[key]))
    context = {
        'tab2':tab2,
        'z':z
    }
    return render(request, 'conjunto.html', context)

# _______Reduccion de caracteristicas ______
def mostrarCaracteristic(request):
    X = df.drop("Class", axis=1)
    y = df["Class"].copy()
    
    from sklearn.cluster import KMeans

    kmeans = KMeans(n_clusters=5, random_state=42)
    clusters = kmeans.fit_predict(X)

    counter = Counter(clusters.tolist())
    bad_counter = Counter(clusters[y == 1].tolist())

    text1 = []
    for key in sorted(counter.keys()):
        text1.append("Label {0} has {1} samples - {2} are malicious samples".format(
            key, counter[key], bad_counter[key]))
        
    # Utilizamos Random Forest para realizar selección de características
    from sklearn.ensemble import RandomForestClassifier

    clf_rnd = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    text2 = clf_rnd.fit(X, y)

    # Seleccionamos las características más importantes
    feature_importances = {name: score for name, score in zip(list(df), clf_rnd.feature_importances_)}
    feature_importances_sorted = pd.Series(feature_importances).sort_values(ascending=False)
    
    # Reducimos el conjunto de datos a las 7 características más importantes
    X_reduced = X[list(feature_importances_sorted.head(7).index)].copy()
    tab3 = X_reduced.head(10)
    
    from sklearn.cluster import KMeans

    kmeans = KMeans(n_clusters=5, random_state=42)
    clusters = kmeans.fit_predict(X_reduced)

    # Evaluamos los clusters y el contenido que se han formado
    counter = Counter(clusters.tolist())
    bad_counter = Counter(clusters[y == 1].tolist())

    cluster_info = []
    for key in sorted(counter.keys()):
        cluster_info.append("Label {0} has {1} samples - {2} are malicious samples".format(
            key, counter[key], bad_counter[key]))

    ps = purity_score(y, clusters)
    s = metrics.silhouette_score(X_reduced, clusters, sample_size=10000)
    ch = metrics.calinski_harabasz_score(X_reduced, clusters)
    context = {
        'text1':text1,
        'text2':text2,#texto
        'tab3':tab3,  #tabla
        'cluster_info':cluster_info,
        'ps':ps,
        's':s,
        'ch':ch
    }
    return render(request, 'caracteres.html', context)

def mostrarDBSCAN(request):
    df = pd.read_csv("creditcard.csv", nrows=70000)
    df = df.drop(["Time", "Amount"], axis=1)
    from sklearn.cluster import DBSCAN

    X = df[["V10", "V14"]].copy()
    y = df["Class"].copy()

    from sklearn.cluster import DBSCAN
    dbscan = DBSCAN(eps=0.15, min_samples=13)
    texto1 = dbscan.fit(X)

    # funcion
    def plot_dbscan(dbscan, X, size):
        core_mask = np.zeros_like(dbscan.labels_, dtype=bool)
        core_mask[dbscan.core_sample_indices_] = True
        anomalies_mask = dbscan.labels_ == -1
        non_core_mask = ~(core_mask | anomalies_mask)

        cores = dbscan.components_
        anomalies = X[anomalies_mask]
        non_cores = X[non_core_mask]

        plt.scatter(cores[:, 0], cores[:, 1],
                    c=dbscan.labels_[core_mask], marker='o', s=size, cmap="Paired")
        plt.scatter(cores[:, 0], cores[:, 1], marker='*', s=20, c=dbscan.labels_[core_mask])
        plt.scatter(anomalies[:, 0], anomalies[:, 1],
                    c="r", marker=".", s=100)
        plt.scatter(non_cores[:, 0], non_cores[:, 1], c=dbscan.labels_[non_core_mask], marker=".")
        plt.title("eps={:.2f}, min_samples={}".format(dbscan.eps, dbscan.min_samples), fontsize=14)

    plt.figure(figsize=(12, 6))
    plot_dbscan(dbscan, X.values, size=100)
    plt.xlabel("V10", fontsize=14)
    plt.ylabel("V14", fontsize=14)
    plt.savefig(os.path.join(settings.BASE_DIR, 'static', 'images', 'mancha_de_sangre.png'))

    counter = Counter(dbscan.labels_.tolist())
    bad_counter = Counter(dbscan.labels_[y == 1].tolist())

    lista1 = []
    for key in sorted(counter.keys()):
        lista1.append("Label {0} has {1} samples - {2} are malicious samples".format(
            key, counter[key], bad_counter[key]))
        
    context = {
        'texto1':texto1,
        'lista1':lista1,
    }

    return render(request, 'ultimo.html', context)


def carDBSCAN(request):
    df = pd.read_csv("creditcard.csv", nrows=70000)
    X = df.drop("Class", axis=1)
    y = df["Class"].copy()
     # funcion
    def plot_dbscan(dbscan, X, size):
        core_mask = np.zeros_like(dbscan.labels_, dtype=bool)
        core_mask[dbscan.core_sample_indices_] = True
        anomalies_mask = dbscan.labels_ == -1
        non_core_mask = ~(core_mask | anomalies_mask)

        cores = dbscan.components_
        anomalies = X[anomalies_mask]
        non_cores = X[non_core_mask]

        plt.scatter(cores[:, 0], cores[:, 1],
                    c=dbscan.labels_[core_mask], marker='o', s=size, cmap="Paired")
        plt.scatter(cores[:, 0], cores[:, 1], marker='*', s=20, c=dbscan.labels_[core_mask])
        plt.scatter(anomalies[:, 0], anomalies[:, 1],
                    c="r", marker=".", s=100)
        plt.scatter(non_cores[:, 0], non_cores[:, 1], c=dbscan.labels_[non_core_mask], marker=".")
        plt.title("eps={:.2f}, min_samples={}".format(dbscan.eps, dbscan.min_samples), fontsize=14)

    from sklearn.ensemble import RandomForestClassifier

    clf_rnd = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    clf_rnd.fit(X, y)

    # Seleccionamos las características más importantes
    feature_importances = {name: score for name, score in zip(list(df), clf_rnd.feature_importances_)}
    feature_importances_sorted = pd.Series(feature_importances).sort_values(ascending=False)

    # Reducimos el conjunto de datos a las 7 características más importantes
    X_reduced = X[list(feature_importances_sorted.head(7).index)].copy()
    
    tab4 = X_reduced.head(10)
    

    from sklearn.cluster import DBSCAN

    dbscan = DBSCAN(eps=0.70, min_samples=25)
    textos1 = dbscan.fit(X_reduced)

    counter = Counter(dbscan.labels_.tolist())
    bad_counter = Counter(dbscan.labels_[y == 1].tolist())

    lista4 = []
    for key in sorted(counter.keys()):
        lista4.append("Label {0} has {1} samples - {2} are malicious samples".format(
            key, counter[key], bad_counter[key]))
        
    # Obtenemos los clusters del objeto dbscan
    clusters = dbscan.labels_

    # Calculamos el purity score, es importante darse cuenta de que recibe las etiquetas
    pss = purity_score(y, clusters)

    scc = metrics.silhouette_score(X_reduced, clusters, sample_size=10000)

    chs = metrics.calinski_harabasz_score(X_reduced, clusters)

    # Generamos un conjunto de datos
    from sklearn.datasets import make_moons

    X, y = make_moons(n_samples=1000, noise=0.05, random_state=42)

    plt.figure(figsize=(12, 6))
    plt.scatter(X[:,0][y == 0], X[:,1][y == 0], c="g", marker=".")
    plt.scatter(X[:,0][y == 1], X[:,1][y == 1], c="r", marker=".")
    plt.savefig(os.path.join(settings.BASE_DIR, 'static', 'images', 'uzumaki_uno.png'))

    from sklearn.cluster import DBSCAN

    dbscan = DBSCAN(eps=0.1, min_samples=6)
    dbscan.fit(X)

    # Representamos el límite de decisión
    plt.figure(figsize=(12, 6))
    plot_dbscan(dbscan, X, size=100)
    plt.savefig(os.path.join(settings.BASE_DIR, 'static', 'images', 'uzumaki_dos.png'))


    context = {
        'tab4':tab4,
        'textos1': textos1,
        'lista4':lista4,
        'pss':pss,
        'scc':scc, 
        'chs':chs
    }
    return render(request, 'ultimos.html', context)
