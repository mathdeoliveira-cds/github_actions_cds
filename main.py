import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Carregar dados (supondo que temos um arquivo CSV no repositório)
data = pd.read_csv('data.csv')

# Realizar uma análise simples
summary = data.describe()
summary.to_csv('results.csv')

# Criar um gráfico
plt.figure(figsize=(10, 6))
sns.scatterplot(x='feature', y='target', data=data)
plt.title('Relação entre Feature e Target')
plt.savefig('plot.png')

# Modelo de regressão linear simples
X = data[['feature']]
y = data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Adicionar métricas ao arquivo de resultados
with open('results.csv', 'a') as f:
    f.write(f"\nMSE: {mean_squared_error(y_test, y_pred)}")
    f.write(f"\nR2 Score: {r2_score(y_test, y_pred)}")

print("Análise concluída. Resultados salvos em 'results.csv' e 'plot.png'.")