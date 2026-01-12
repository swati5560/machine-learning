# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import warnings as wr
# wr.filterwarnings('ignore')

# df = pd.read_csv('WineQT.csv')
# print(df.head())

# df.shape

# df.info()

# df.describe().T

# df.columns.tolist()
# quality_counts = df['quality'].value_counts()

# plt.figure(figsize=(8, 6))
# plt.bar(quality_counts.index, quality_counts, color='deeppink')
# plt.title('Count Plot of Quality')
# plt.xlabel('Quality')
# plt.ylabel('Count')
# plt.show()

# sns.set_style("darkgrid")

# numerical_columns = df.select_dtypes(include=["int64", "float64"]).columns

# plt.figure(figsize=(14, len(numerical_columns) * 3))
# for idx, feature in enumerate(numerical_columns, 1):
#     plt.subplot(len(numerical_columns), 2, idx)
#     sns.histplot(df[feature], kde=True)
#     plt.title(f"{feature} | Skewness: {round(df[feature].skew(), 2)}")
# import matplotlib.pyplot as plt

# plt.plot([1, 2, 3], [4, 5, 6])
# plt.show(block=True)
# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(10, 8))

# sns.swarmplot(x="quality", y="alcohol", data=df, palette='viridis')

# plt.title('Swarm Plot for Quality and Alcohol')
# plt.xlabel('Quality')
# plt.ylabel('Alcohol')
# plt.show()

# sns.set_palette("Pastel1")

# plt.figure(figsize=(10, 6))

# sns.pairplot(df)

# plt.title('Count Plot of Quality')
# plt.suptitle('Pair Plot for DataFrame')
# plt.show()


# df['quality'] = df['quality'].astype(str)  
# plt.figure(figsize=(10, 8))
# sns.violinplot(x="quality", y="alcohol", data=df, palette={
#                            '3': 'lightcoral', '4': 'lightblue', '5': 'lightgreen', '6': 'gold', '7': 'lightskyblue', '8': 'lightpink'}, alpha=0.7)
# plt.title('Violin Plot for Quality and Alcohol')
# plt.xlabel('Quality')
# plt.ylabel('Alcohol')
# plt.show()


# sns.boxplot(x='quality', y='alcohol',data=df)
# plt.show()

# plt.figure(figsize=(15, 10))
# sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='Pastel2', linewidths=2)
# plt.title('Correlation Heatmap')
# plt.show()


# import speech_recognition as sr

# # Initialize recognizer class (for recognizing the speech)
# r = sr.Recognizer()

# # Reading Microphone as source
# # listening the speech and store in audio_text variable
# with sr.Microphone() as source:
#     print("Talk")
#     audio_text = r.listen(source)
#     print("Time over, thanks")
#     # recoginze_() method will throw a request
#     # error if the API is unreachable,
#     # hence using exception handling
    
#     try:
#         # using google speech recognition
#         print("Text: "+r.recognize_google(audio_text))
#     except:
#          print("Sorry, I did not get that")


import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


np.random.seed(42)
X = np.random.rand(50, 1) * 100  
Y = 3.5 * X + np.random.randn(50, 1) * 20 

model = LinearRegression()
model.fit(X, Y) 

Y_pred = model.predict(X)

plt.figure(figsize=(8,6)) 
plt.scatter(X, Y, color='blue', label='Data Points') 
plt.plot(X, Y_pred, color='red', linewidth=2, label='Regression Line') 
plt.title('Linear Regression on Random Dataset')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)
plt.show()
print("Slope (Coefficient):", model.coef_[0][0])
print("Intercept:", model.intercept_[0])