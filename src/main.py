import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.impute import SimpleImputer

data = pd.read_csv('../datasets/california_housing.csv')
print("Primeras filas del dataset:\n", data.head())
print("\nResumen estadístico:\n", data.describe())

data = pd.get_dummies(data, columns=['ocean_proximity'], drop_first=True)

imputer = SimpleImputer(strategy='mean')
data['total_bedrooms'] = imputer.fit_transform(data[['total_bedrooms']])

data_numeric = data.select_dtypes(include=[np.number])

plt.figure(figsize=(12, 8))
sns.heatmap(data_numeric.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Mapa de Correlación entre Variables Numéricas")
plt.show()

data['median_house_value'] = np.log1p(data['median_house_value'])
data['total_bedrooms'] = np.log1p(data['total_bedrooms'])

X = data.drop(columns=['median_house_value'])
y = data['median_house_value']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

selector = SelectKBest(score_func=f_regression, k=5)
X_selected = selector.fit_transform(X_scaled, y)
selected_features = np.array(X.columns)[selector.get_support()]
print("Características seleccionadas:", selected_features)

models = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.1)
}

print("\nEvaluación de Modelos:")
for name, model in models.items():
    scores = cross_val_score(model, X_selected, y, scoring="neg_mean_squared_error", cv=10)
    rmse_scores = np.sqrt(-scores)
    print(f"{name} - RMSE promedio: {rmse_scores.mean():.3f}, Desviación estándar: {rmse_scores.std():.3f}")

ridge_params = {'alpha': [0.1, 1.0, 10.0, 100.0]}
lasso_params = {'alpha': [0.001, 0.01, 0.1, 1.0]}

ridge_grid = GridSearchCV(Ridge(), ridge_params, cv=10, scoring='neg_mean_squared_error')
ridge_grid.fit(X_selected, y)
print("\nMejor parámetro para Ridge:", ridge_grid.best_params_)

lasso_grid = GridSearchCV(Lasso(), lasso_params, cv=10, scoring='neg_mean_squared_error')
lasso_grid.fit(X_selected, y)
print("Mejor parámetro para Lasso:", lasso_grid.best_params_)

X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)
best_model = Lasso(alpha=lasso_grid.best_params_['alpha'])
best_model.fit(X_train, y_train)

y_pred = best_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f"\nEvaluación final en el conjunto de prueba:")
print(f"RMSE: {rmse:.3f}")
print(f"R2 Score: {r2:.3f}")
