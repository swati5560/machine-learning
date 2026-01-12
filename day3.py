# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# df1=pd.read_csv('df1.csv',index_col=0)
# df2=pd.read_csv('df2.csv')
# print(df1.head())
# print(df2.info())

# df2.plot.bar()
# df1.plot.scatter(x='A', y='B')
# df1.plot(
#     style=['-', '--', '-.', ':'],
#     title='Line Plot with Different Styles',
#     xlabel='Index',
#     ylabel='Values',
#     grid=True
# )
# df2.iloc[0].plot.pie(
#     autopct='%1.1f%%',
#     figsize=(6, 6),
#     title='Pie Chart'
# )
# plt.ylabel('')
# # Show the plot
# plt.show()

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import warnings as wr
# wr.filterwarnings('ignore')

# df = pd.read_csv("D:\sushswat\kle4457\wineQT.csv")
# print(df.head())

# # sns.set_style("darkgrid")
# # numerical_columns = df.select_dtypes(include=["int64", "float64"]).columns
# # plt.figure(figsize=(14, len(numerical_columns) * 3))
# # for idx, feature in enumerate(numerical_columns, 1):
# #     plt.subplot(len(numerical_columns), 2, idx)
# #     sns.histplot(df[feature], kde=True)
# #     plt.title(f"{feature} | Skewness: {round(df[feature].skew(), 2)}")
# #     plt.tight_layout()
# #     plt.show()

# # quality_counts = df['quality'].value_counts()
# # plt.figure(figsize=(8, 6))
# # plt.bar(quality_counts.index, quality_counts, color='deeppink')
# # plt.title('Count Plot of Quality')
# # plt.xlabel('Quality')
# # plt.ylabel('Count')
# # plt.show()

# plt.figure(figsize=(10, 8))
# sns.swarmplot(x="quality", y="alcohol", data=df, palette='viridis')
# plt.title('Swarm Plot for Quality and Alcohol')
# plt.xlabel('Quality')
# plt.ylabel('Alcohol')
# plt.show()

