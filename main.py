import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler

# Carregar o dataset
data = load_diabetes()

# Converter para DataFrame para fácil manipulação
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

# Visualizar as primeiras linhas do dataset
print(df.head())

# Verificar correlações entre as variáveis
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()

# Verificar a distribuição da variável alvo
sns.histplot(df['target'])
plt.show()

# Separar as features X e o target y
X = df.drop('target', axis=1)
y = df['target']

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Padronizar os dados
scaler= StandardScaler()
X_train= scaler.fit_transform(X_train)
X_test= scaler.transform(X_test)

# Inicializar o modelo
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Treinar o modelo
model.fit(X_train, y_train)

# Fazer previsões
y_pred = model.predict(X_test)

# Acurácia do modelo
accuracy = accuracy_score(y_test, y_pred)
print(f"Acurácia: {accuracy:.2f}")

# Matriz de Confusão
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.title("Matriz de Confusão")
plt.ylabel("Verdadeiro")
plt.xlabel("Previsto")
plt.show()

# Relatório de Classificação (Precision, Recall, F1-Score)
print(classification_report(y_test, y_pred))


