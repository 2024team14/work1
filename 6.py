import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Step 1: 匯入資料
data = pd.read_csv("C:\\Users\\emily\\OneDrive\\桌面\\train.csv", delimiter=';')
print("資料集前五筆:\n", data.head())
print("\n資料集欄位資訊:\n", data.info())

# 檢查缺失值
print("\n檢查缺失值:\n", data.isnull().sum())

# Step 2: 資料分析與視覺化
# 將年齡分群
bins = [0, 20, 40, 60, 80, 100]
labels = ['0-20', '20-40', '40-60', '60-80', '80-100']
data['age_group'] = pd.cut(data['age'], bins=bins, labels=labels)

# 將目標變數 'y' 轉換為數值
data['y_numeric'] = data['y'].apply(lambda x: 1 if x == 'yes' else 0)

# 計算每個年齡群體中是否訂閱的總數
age_group_counts = data.groupby(['age_group', 'y'])['y'].count().unstack().fillna(0)

# 繪製年齡區間的水平長條圖
age_group_counts.rename(columns={'yes': 'Yes', 'no': 'No'}, inplace=True)
age_group_counts.plot(kind='barh', stacked=False, figsize=(10, 6), color=['skyblue', 'salmon'])

# 添加標題與標籤
plt.title('Count of y by Age Group', fontsize=14)
plt.xlabel('Count', fontsize=12)
plt.ylabel('Age Group', fontsize=12)
plt.legend(title='y', labels=['No', 'Yes'], loc='upper right')
plt.tight_layout()
plt.show()

# Step 3: 預測模型
# 篩選所需的欄位
selected_features = ['age', 'balance', 'loan']
X = data[selected_features]
y = data['y_numeric']

# 將 loan 欄位轉換為數值型
X['loan'] = X['loan'].apply(lambda x: 1 if x == 'yes' else 0)

# 切分資料集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 建立羅吉斯回歸模型
model = LogisticRegression()
model.fit(X_train, y_train)

# 預測並計算準確度
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nLogistic Regression 模型的測試準確度：{accuracy:.2f}")