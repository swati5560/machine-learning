# from sklearn.datasets import make_s_curve
# from sklearn.manifold import Isomap
# import matplotlib.pyplot as plt
# X, color = make_s_curve(n_samples=1000, random_state=42)
# isomap = Isomap(n_neighbors=10, n_components=2)
# X_isomap = isomap.fit_transform(X)
# fig, ax = plt.subplots(1, 2, figsize=(12, 5))
# ax[0].scatter(X[:, 0], X[:, 2], c=color, cmap=plt.cm.Spectral)
# ax[0].set_title('Original 3D Data')
# ax[1].scatter(X_isomap[:, 0], X_isomap[:, 1], c=color, cmap=plt.cm.Spectral)
# ax[1].set_title('Isomap Reduced 2D Data')
# plt.show()

# from sklearn.datasets import load_digits
# from sklearn.manifold import Isomap
# import matplotlib.pyplot as plt
# digits = load_digits()
# isomap = Isomap(n_neighbors=30, n_components=2)
# digits_isomap = isomap.fit_transform(digits.data)
# fig, ax = plt.subplots(1, 2, figsize=(12, 5))
# ax[0].scatter(digits.data[:, 0], digits.data[:, 1], c=digits.target, cmap=plt.cm.tab10)
# ax[0].set_title('Original 2D Data (First Two Features)')
# ax[1].scatter(digits_isomap[:, 0], digits_isomap[:, 1], c=digits.target, cmap=plt.cm.tab10)
# ax[1].set_title('Isomap Reduced 2D Data')
# plt.show()

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from sklearn.datasets import load_iris
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from sklearn.model_selection import train_test_split
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# from sklearn.ensemble import RandomForestClassifier
# from matplotlib.colors import ListedColormap

# iris = load_iris()
# dataset = pd.DataFrame(columns=iris.feature_names, data=iris.data)
# dataset['target'] = iris.target
# X = dataset.iloc[:, 0:4].values
# y = dataset.iloc[:, 4].values

# sc = StandardScaler()
# X_scaled = sc.fit_transform(X)
# le = LabelEncoder()
# y_encoded = le.fit_transform(y)
# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# fig = plt.figure(figsize=(7,5))
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(X[:,0], X[:,1], X[:,2], c=y, cmap='rainbow', alpha=0.7, edgecolors='b')
# ax.set_xlabel(iris.feature_names[0])
# ax.set_ylabel(iris.feature_names[1])
# ax.set_zlabel(iris.feature_names[2])
# ax.set_title('Original Iris Dataset (3D)')
# plt.show()

# X_train_2D = X_train[:, :2]
# rf_without_lda = RandomForestClassifier(max_depth=2, random_state=0)
# rf_without_lda.fit(X_train_2D, y_train)
# x_min, x_max = X_train_2D[:,0].min() - 1, X_train_2D[:,0].max() + 1
# y_min, y_max = X_train_2D[:,1].min() - 1, X_train_2D[:,1].max() + 1
# xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
# np.arange(y_min, y_max, 0.02))
# Z = rf_without_lda.predict(np.c_[xx.ravel(), yy.ravel()])
# Z = Z.reshape(xx.shape)
# cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
# plt.figure(figsize=(7,5))
# plt.contourf(xx, yy, Z, alpha=0.3, cmap=cmap_light)
# plt.scatter(X_train_2D[:,0], X_train_2D[:,1], c=y_train, cmap='rainbow', edgecolors='b')
# plt.xlabel(iris.feature_names[0])
# plt.ylabel(iris.feature_names[1])
# plt.title('Random Forest Decision Boundary Without LDA')
# plt.show()

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# from sklearn.datasets import load_iris
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# from sklearn.ensemble import RandomForestClassifier
# from matplotlib.colors import ListedColormap

# # Load dataset
# iris = load_iris()
# X = iris.data
# y = iris.target

# # Standardize features
# sc = StandardScaler()
# X_scaled = sc.fit_transform(X)

# # Train-test split
# X_train, X_test, y_train, y_test = train_test_split(
#     X_scaled, y, test_size=0.2, random_state=42
# )

# # ===== LDA TRANSFORMATION (THIS WAS MISSING) =====
# lda = LinearDiscriminantAnalysis(n_components=2)
# X_train_lda = lda.fit_transform(X_train, y_train)
# X_test_lda = lda.transform(X_test)

# # Colormap
# cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])

# # ===== RANDOM FOREST WITH LDA =====
# rf_with_lda = RandomForestClassifier(max_depth=2, random_state=0)
# rf_with_lda.fit(X_train_lda, y_train)

# # Decision boundary grid
# x_min, x_max = X_train_lda[:, 0].min() - 1, X_train_lda[:, 0].max() + 1
# y_min, y_max = X_train_lda[:, 1].min() - 1, X_train_lda[:, 1].max() + 1

# xx, yy = np.meshgrid(
#     np.arange(x_min, x_max, 0.02),
#     np.arange(y_min, y_max, 0.02)
# )

# Z = rf_with_lda.predict(np.c_[xx.ravel(), yy.ravel()])
# Z = Z.reshape(xx.shape)

# # Plot
# plt.figure(figsize=(7,5))
# plt.contourf(xx, yy, Z, alpha=0.3, cmap=cmap_light)
# plt.scatter(
#     X_train_lda[:, 0],
#     X_train_lda[:, 1],
#     c=y_train,
#     cmap='rainbow',
#     edgecolors='b'
# )
# plt.xlabel('LDA Component 1')
# plt.ylabel('LDA Component 2')
# plt.title('Random Forest Decision Boundary With LDA')
# plt.show()


# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import confusion_matrix
# import matplotlib.pyplot as plt
# import seaborn as sns

# data = {
# 'Height': [170, 165, 180, 175, 160, 172, 168, 177, 162, 158],
# 'Weight': [65, 59, 75, 68, 55, 70, 62, 74, 58, 54],
# 'Age': [30, 25, 35, 28, 22, 32, 27, 33, 24, 21],
# 'Gender': [1, 0, 1, 1, 0, 1, 0, 1, 0, 0]  # 1 = Male, 0 = Female
# }
# df = pd.DataFrame(data)
# print(df)

# X = df.drop('Gender', axis=1)
# y = df['Gender']
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# pca = PCA(n_components=2)
# X_pca = pca.fit_transform(X_scaled)
# X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.6, random_state=24)
# model = LogisticRegression()
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)

# cm = confusion_matrix(y_test, y_pred)
# plt.figure(figsize=(5,4))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Female', 'Male'], yticklabels=['Female', 'Male'])
# plt.xlabel('Predicted Label')
# plt.ylabel('True Label')
# plt.title('Confusion Matrix')
# plt.show()

# y_numeric = pd.factorize(y)[0]
# plt.figure(figsize=(20, 55))
# plt.subplot(1, 2, 1)
# plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y_numeric, cmap='coolwarm', edgecolor='k', s=80)
# plt.xlabel('Original Feature 1')
# plt.ylabel('Original Feature 2')
# plt.title('Before PCA: Using First 2 Standardized Features')
# plt.colorbar(label='Target classes')
# plt.subplot(1, 2, 2)
# plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_numeric, cmap='coolwarm', edgecolor='k', s=80)
# plt.xlabel('Principal Component 1')
# plt.ylabel('Principal Component 2')
# plt.title('After PCA: Projected onto 2 Principal Components')
# plt.colorbar(label='Target classes')
# plt.tight_layout()
# plt.show()


# import matplotlib.pyplot as plt
# import numpy as np
# from sklearn.cluster import DBSCAN
# from sklearn import metrics
# from sklearn.datasets import make_blobs
# from sklearn.preprocessing import StandardScaler
# from sklearn import datasets
# X, y_true = make_blobs(n_samples=300, centers=4,
# cluster_std=0.50, random_state=0)

# db = DBSCAN(eps=0.3, min_samples=10).fit(X)
# core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
# core_samples_mask[db.core_sample_indices_] = True
# labels = db.labels_
# n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
# unique_labels = set(labels)
# colors = ['y', 'b', 'g', 'r']
# print(colors)
# for k, col in zip(unique_labels, colors):
#     if k == -1:
#         col = 'k'
#     class_member_mask = (labels == k)
#     xy = X[class_member_mask & core_samples_mask]
#     plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
#     markeredgecolor='k',
#     markersize=6)
#     xy = X[class_member_mask & ~core_samples_mask]
#     plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
#     markeredgecolor='k',
#     markersize=6)
#     plt.title('number of clusters: %d' % n_clusters_)
#     plt.show()

# from sklearn.metrics import  adjusted_rand_score
# sc = metrics.silhouette_score(X, labels)
# print("Silhouette Coefficient:%5.2f" % sc)
# ari = adjusted_rand_score(y_true, labels)
# print("Adjusted Rand Index: %8.2f" % ari)    

#: Importing the necessary libraries
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.datasets import make_blobs

# #Creating Custom Dataset
# X,y = make_blobs(n_samples = 500,n_features = 2,centers = 3,random_state = 23)
# fig = plt.figure(0)
# plt.grid(True)
# plt.scatter(X[:,0],X[:,1])
# plt.show()


# #Initializing Random Centroids
# k = 3
# clusters = {}
# np.random.seed(23)
# for idx in range(k):
#  center = 2*(2*np.random.random((X.shape[1],))-1)
#  points = []
#  cluster = {
#  'center' : center,
#  'points' : []
#  }

#  clusters[idx] = cluster

# clusters

# #Plotting Random Initialized Center with Data Points
# plt.scatter(X[:,0],X[:,1])
# plt.grid(True)
# for i in clusters:
#  center = clusters[i]['center']
#  plt.scatter(center[0],center[1],marker = '*',c = 'red')
# plt.show()

# #Defining Euclidean Distance
# def distance(p1,p2):
#  return np.sqrt(np.sum((p1-p2)**2))

# #Creating Assign and Update Functions
# def assign_clusters(X, clusters):
#  for idx in range(X.shape[0]):
#     dist = []

#     curr_x = X[idx]

#     for i in range(k):
#         dis = distance(curr_x,clusters[i]['center'])
#         dist.append(dis)
#     curr_cluster = np.argmin(dist)
#     clusters[curr_cluster]['points'].append(curr_x)
#     return clusters
# def update_clusters(X, clusters):
#  for i in range(k):
#     points = np.array(clusters[i]['points'])
#     if points.shape[0] > 0:
#         new_center = points.mean(axis =0)
#         clusters[i]['center'] = new_center

#         clusters[i]['points'] = []
#  return clusters

# # Predicting the Cluster for the Data Points
# def pred_cluster(X, clusters):
#  pred = []
#  for i in range(X.shape[0]):
#     dist = []
#     for j in range(k):
#         dist.append(distance(X[i],clusters[j]['center']))
#     pred.append(np.argmin(dist))
#  return pred

#  #Assigning, Updating and Predicting the Cluster Centers
# clusters = assign_clusters(X,clusters)
# clusters = update_clusters(X,clusters)
# pred = pred_cluster(X,clusters)

# #Plotting Data Points with Predicted Cluster Centers
# plt.scatter(X[:,0],X[:,1],c = pred)
# for i in clusters:
#  center = clusters[i]['center']
#  plt.scatter(center[0],center[1],marker = '^',c = 'red')
# plt.show()

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Plot settings
plt.rc('figure', autolayout=True)
plt.rc('image', cmap='magma')

# Define kernel (edge detection)
kernel = tf.constant([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1],
])

# Load image
image = tf.io.read_file('Ganesh.jpg')
image = tf.io.decode_jpeg(image, channels=1)
image = tf.image.resize(image, size=[300, 300])

# Plot original image
img = tf.squeeze(image).numpy()
plt.figure(figsize=(5, 5))
plt.imshow(img, cmap='gray')
plt.axis('off')
plt.title('Original Gray Scale Image')
plt.show()

# Reformat image
image = tf.image.convert_image_dtype(image, dtype=tf.float32)
image = tf.expand_dims(image, axis=0)

# Reformat kernel
kernel = tf.reshape(kernel, [3, 3, 1, 1])
kernel = tf.cast(kernel, dtype=tf.float32)

# Convolution
image_filter = tf.nn.conv2d(
    input=image,
    filters=kernel,
    strides=1,
    padding='SAME'
)

# Activation (ReLU)
image_detect = tf.nn.relu(image_filter)

# Pooling
image_condense = tf.nn.pool(
    input=image_detect,
    window_shape=(2, 2),
    pooling_type='MAX',
    strides=(2, 2),
    padding='SAME'
)

# Plot results
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(tf.squeeze(image_filter))
plt.axis('off')
plt.title('Convolution')

plt.subplot(1, 3, 2)
plt.imshow(tf.squeeze(image_detect))
plt.axis('off')
plt.title('Activation (ReLU)')

plt.subplot(1, 3, 3)
plt.imshow(tf.squeeze(image_condense))
plt.axis('off')
plt.title('Pooling')

plt.show()

