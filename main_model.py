# main_model.py

import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# قراءة البيانات
df = pd.read_csv('Players Database.csv')
f_df = df.dropna()

# إعادة تسمية الأعمدة
f_df.columns = ['Pos', 'Age', 'Matches_Played', 'Starts', 'Minutes', '90s', 'Goals', 'Assists', 'xG']

# تحويل Pos إلى أرقام
f_df['Pos'] = f_df['Pos'].astype('category').cat.codes

# تجهيز المدخلات والمخرجات
X = f_df[['Pos', 'Age', 'Matches_Played', 'Starts', 'Minutes', '90s', 'Goals', 'Assists']]
y = f_df['xG']

# تقسيم البيانات
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=59)

# تدريب النموذج
model = RandomForestRegressor(random_state=59)
model.fit(X_train, y_train)

# التنبؤ والتقييم
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R² Score:", r2)
