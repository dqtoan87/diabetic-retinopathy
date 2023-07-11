import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib

# Tải dữ liệu từ file CSV
data = pd.read_csv('/content/Data10_3.csv')

# Chia dữ liệu thành features (đặc trưng) và target (nhãn)
X = data.drop('Value', axis=1)
y = data['Value']

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tạo mô hình cây quyết định
model = DecisionTreeClassifier()

# Huấn luyện mô hình trên tập huấn luyện
model.fit(X_train, y_train)

# Dự đoán nhãn cho tập kiểm tra
y_pred = model.predict(X_test)

# Đánh giá độ chính xác của mô hình
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Lưu mô hình vào file
joblib.dump(model, '/content/model_DR.pkl')