# from sklearn.datasets import make_moons
# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd
# # Create synthetic 2D data
# X, y = make_moons(n_samples=200, noise=0.5, random_state=40)
# # Create a DataFrame for plotting
# df = pd.DataFrame(X, columns=["Feature 1", "Feature 2"])
# df['Target'] = y
# # Visualize the 2D data
# plt.figure(figsize=(5, 10))
# sns.scatterplot(data=df, x="Feature 1", y="Feature 2", hue="Target", palette="Set1")
# plt.title("Classification Data Diagram(make_moons)")
# plt.grid(True)
# plt.show()

# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# # Normalize the features
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
# # Split into train and test
# X_train, X_test, y_train, y_test = train_test_split(
#  X_scaled, y, test_size=0.5, random_state=40, stratify=y
# )
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import accuracy_score
# # Train a k-NN classifier
# knn = KNeighborsClassifier(n_neighbors=5)
# knn.fit(X_train, y_train)
# # Predict and evaluate
# y_pred = knn.predict(X_test)
# print(f"Test Accuracy (k=5): {accuracy_score(y_test, y_pred):.2f}")


# from sklearn.model_selection import cross_val_score
# import numpy as np
# # Range of k values to try
# k_range = range(1, 21)
# cv_scores = []
# # Evaluate each k using 5-fold cross-validation
# for k in k_range:
#  knn = KNeighborsClassifier(n_neighbors=k)
#  scores = cross_val_score(knn, X_scaled, y, cv=5, scoring='accuracy')
#  cv_scores.append(scores.mean())
# # Plot accuracy vs. k
# plt.figure(figsize=(8, 5))
# plt.plot(k_range, cv_scores, marker='o')
# plt.title("k-NN Cross-Validation Accuracy vs k")
# plt.xlabel("Number of Neighbors: k")
# plt.ylabel("Cross-Validated Accuracy")
# plt.grid(True)
# plt.show()
# # Best k
# best_k = k_range[np.argmax(cv_scores)]
# print(f"Best k from cross-validation: {best_k}")


# # Train final model with best k
# best_knn = KNeighborsClassifier(n_neighbors=best_k)
# best_knn.fit(X_train, y_train)
# # Predict on test data
# y_pred = best_knn.predict(X_test)


# from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
# # Confusion Matrix
# cm = confusion_matrix(y_test, y_pred)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Class 0", "Class 1"])
# disp.plot(cmap="Blues")
# plt.title(f"Confusion Matrix (k={best_k})")
# plt.grid(False)
# plt.show()
# # Detailed classification report
# print("Classification Report:")
# print(classification_report(y_test, y_pred, target_names=["Class 0", "Class 1"]))

  



# # # Predict on mesh grid with best k
# # Z = best_knn.predict(np.c_[xx.ravel(), yy.ravel()])
# # Z = Z.reshape(xx.shape)
# # # Plot decision boundary
# # plt.figure(figsize=(8, 6))

# # plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.3)
# # sns.scatterplot(
# #  x=X_scaled[:, 0],
# #    y=X_scaled[:, 1], 
# #      hue=y,
# #        palette="Set1", 
# #        edgecolor='k')
# # plt.title(f"Decision Boundary with Best k = {best_k}")
# # plt.xlabel("Feature 1 (scaled)")
# # plt.ylabel("Feature 2 (scaled)")
# # plt.grid(True)
# # plt.show()
    


#     # Create mesh grid
# h = 0.02
# x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
# y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1

# xx, yy = np.meshgrid(
#     np.arange(x_min, x_max, h),
#     np.arange(y_min, y_max, h)
# )

# # Predict on mesh grid
# Z = best_knn.predict(np.c_[xx.ravel(), yy.ravel()])
# Z = Z.reshape(xx.shape)

# # Plot decision boundary
# plt.figure(figsize=(6, 8))
# plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.3)

# sns.scatterplot(
#     x=X_scaled[:, 0],
#     y=X_scaled[:, 1],
#     hue=y,
#     palette="Set1",
#     edgecolor="r"
# )

# plt.title(f"Boundary Tree with  k = {best_k}")
# plt.xlabel("Feature 1 (scaled)")
# plt.ylabel("Feature 2 (scaled)")
# plt.grid(True)
# plt.show()


# import pandas as pd
# import numpy as np
# data = {
# 'School ID': [101, 102, 103, np.nan, 105, 106, 107, 108],
# 'Name': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'],
# 'Address': ['123 Main St', '456 Oak Ave', '789 Pine Ln', '101 Elm St', np.nan, '222 Maple Rd', '444 Cedar Blvd', '555 Birch Dr'],
# 'City': ['Mumbai', 'Delhi', 'Bengaluru', 'Chennai', 'Kolkata', np.nan, 'Pune', 'Jaipur'],
# 'Subject': ['Math', 'English', 'Science', 'Math', 'History', 'Math', 'Science', 'English'],
# 'Marks': [85, 92, 78, 89, np.nan, 95, 80, 88],
# 'Rank': [2, 1, 4, 3, 8, 1, 5, 3],
# 'Grade': ['B', 'A', 'C', 'B', 'D', 'A', 'C', 'B']
# }
# df = pd.DataFrame(data)
# print("Sample DataFrame:")
# print(df)


# df_cleaned = df.dropna()
# print("\nDataFrame after removing rows with missing values:")
# print(df_cleaned)

# mean_imputation = df['Marks'].fillna(df['Marks'].mean())
# median_imputation = df['Marks'].fillna(df['Marks'].median())
# mode_imputation = df['Marks'].fillna(df['Marks'].mode().iloc[0])
# print("\nImputation using Mean:")
# print(mean_imputation)
# print("\nImputation using Median:")
# print(median_imputation)
# print("\nImputation using Mode:")
# print(mode_imputation)

# forward_fill = df['Marks'].fillna(method='ffill')
# backward_fill = df['Marks'].fillna(method='bfill')
# print("\nForward Fill:")
# print(forward_fill)
# print("\nBackward Fill:")
# print(backward_fill)


# linear_interpolation = df['Marks'].interpolate(method='linear')
# quadratic_interpolation = df['Marks'].interpolate(method='quadratic')
# print("\nLinear Interpolation:")
# print(linear_interpolation)
# print("\nQuadratic Interpolation:")
# print(quadratic_interpolation)

# Python code for Feature Scaling using Robust Scaling
# """ PART 1: Importing Libraries """
# import pandas as pd
# import numpy as np
# from sklearn import preprocessing
# import matplotlib
# import matplotlib.pyplot as plt
# import seaborn as sns
# matplotlib.style.use('ggplot')
# """ PART 2: Making the data distributions """
# x = pd.DataFrame({
# # Distribution with lower outliers
# 'x1': np.concatenate([np.random.normal(15, 2, 2000), np.random.normal(2, 3, 20)]),
# # Distribution with higher outliers
# 'x2': np.concatenate([np.random.normal(20, 3, 2000), np.random.normal(40, 1, 20)]),
# })
# """ PART 3: Scaling the Data """
# scaler = preprocessing.RobustScaler()
# robust_scaled_df = scaler.fit_transform(x)
# robust_scaled_df = pd.DataFrame(robust_scaled_df, columns=['x1', 'x2'])
# """ PART 4: Visualizing the impact of scaling """
# fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(9, 5))
# ax1.set_title('Before Scaling')
# sns.kdeplot(x['x1'], ax=ax1)
# sns.kdeplot(x['x2'], ax=ax1)
# ax2.set_title('After Robust Scaling')
# sns.kdeplot(robust_scaled_df['x1'], ax=ax2)
# sns.kdeplot(robust_scaled_df['x2'], ax=ax2)
# plt.tight_layout()
# plt.show()

# import pandas as pd
# import numpy as np
# from sklearn import preprocessing
# import matplotlib
# import matplotlib.pyplot as plt
# import seaborn as sns

# matplotlib.style.use('fivethirtyeight')
# x = pd.DataFrame({
# 'x1': np.concatenate([np.random.normal(20, 2, 1000), np.random.normal(1, 2, 25)]),
# 'x2': np.concatenate([np.random.normal(30, 2, 1000), np.random.normal(50, 2, 25)]),
# })
# np.random.normal
# scaler = preprocessing.RobustScaler()
# robust_df = scaler.fit_transform(x)
# robust_df = pd.DataFrame(robust_df, columns=['x1', 'x2'])
# scaler = preprocessing.StandardScaler()
# standard_df = scaler.fit_transform(x)
# standard_df = pd.DataFrame(standard_df, columns=['x1', 'x2'])
# scaler = preprocessing.MinMaxScaler()
# minmax_df = scaler.fit_transform(x)
# minmax_df = pd.DataFrame(minmax_df, columns=['x1', 'x2'])
# fig, (ax1, ax2, ax3, ax4) = plt.subplots(ncols=4, figsize=(20, 5))
# ax1.set_title('Before Scaling')
# sns.kdeplot(x['x1'], ax=ax1, color='r')
# sns.kdeplot(x['x2'], ax=ax1, color='b')
# ax2.set_title('After Robust Scaling')
# sns.kdeplot(robust_df['x1'], ax=ax2, color='red')
# sns.kdeplot(robust_df['x2'], ax=ax2, color='blue')
# ax3.set_title('After Standard Scaling')
# sns.kdeplot(standard_df['x1'], ax=ax3, color='black')
# sns.kdeplot(standard_df['x2'], ax=ax3, color='g')
# ax4.set_title('After Min-Max Scaling')
# sns.kdeplot(minmax_df['x1'], ax=ax4, color='black')
# sns.kdeplot(minmax_df['x2'], ax=ax4, color='g')
# plt.show()

# import pandas as pd
# from sklearn.preprocessing import MinMaxScaler, StandardScaler

# df = pd.read_csv(r"D:\sushswat\kle4457\heart.csv")

# X = df.drop('target', axis=1)
# y = df['target']

# features = ['age','trestbps','chol','thalach','oldpeak']

# # Min-Max Normalization
# scaler = MinMaxScaler()
# X_normalized = X.copy()
# X_normalized[features] = scaler.fit_transform(X[features])

# # Z-score Standardization
# scaler_z = StandardScaler()
# X_standardized = X.copy()
# X_standardized[features] = scaler_z.fit_transform(X[features])

# print(X_normalized.head())
# print(X_standardized.head())


# sc = metrics.silhouette_score(X, labels)
# print("Silhouette Coefficient:%0.2f" % sc)
# ari = adjusted_rand_score(y_true, labels)
# print("Adjusted Rand Index: %0.2f" % ari)





