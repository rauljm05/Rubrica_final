import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.datasets import load_iris

# ---------------------------
# 1. Cargar y procesar datos
# ---------------------------
st.title("Iris Species Classification Project")

st.write("Este dashboard permite predecir la especie de una flor según las dimensiones del sépalo y pétalo.")

data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# División datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalamiento
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Modelo
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

# Métricas
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Mostrar métricas
st.subheader("Model Performance Metrics")
st.write(f"**Accuracy:** {accuracy:.2f}")
st.write(f"**Precision:** {precision:.2f}")
st.write(f"**Recall:** {recall:.2f}")
st.write(f"**F1-Score:** {f1:.2f}")

# ---------------------------
# 2. Predicción del usuario
# ---------------------------
st.subheader("Realiza una predicción")

sepal_length = st.slider("Sepal length (cm)", float(X['sepal length (cm)'].min()), float(X['sepal length (cm)'].max()))
sepal_width = st.slider("Sepal width (cm)", float(X['sepal width (cm)'].min()), float(X['sepal width (cm)'].max()))
petal_length = st.slider("Petal length (cm)", float(X['petal length (cm)'].min()), float(X['petal length (cm)'].max()))
petal_width = st.slider("Petal width (cm)", float(X['petal width (cm)'].min()), float(X['petal width (cm)'].max()))

input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
input_data_scaled = scaler.transform(input_data)
prediction = model.predict(input_data_scaled)

species = data.target_names[prediction][0]
st.write(f"Predicción de especie:{species}")
# ---------------------------
# 3D Scatter Plot
# ---------------------------
st.subheader("Visualización 3D de la muestra ingresada")

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Datos reales del dataset
ax.scatter(
    X['sepal length (cm)'], 
    X['petal length (cm)'], 
    X['petal width (cm)'],
    c=y,
    marker='o',
    alpha=0.6
)

# Punto del usuario
ax.scatter(
    input_data[0][0],  # sepal length
    input_data[0][2],  # petal length
    input_data[0][3],  # petal width
    c='red',
    marker='^',
    s=100,
    label='Nueva muestra'
)

# Etiquetas
ax.set_xlabel('Sepal Length (cm)')
ax.set_ylabel('Petal Length (cm)')
ax.set_zlabel('Petal Width (cm)')
ax.set_title('Distribución 3D y nueva muestra')

ax.legend()

st.pyplot(fig)

# ---------------------------
# 3. Mejoras futuras
# ---------------------------
st.sidebar.title(" Opciones")
st.sidebar.write("Próximamente podrás añadir:")
st.sidebar.write("- Gráfico 3D con la muestra nueva")
st.sidebar.write("- Visualizaciones adicionales")
st.sidebar.write("- Selección de modelo")
# ---------------------------
# 4. Visualizaciones
# ---------------------------
st.subheader("Visualización de datos")

st.write("Distribución de características por especie")
st.bar_chart(X)

st.write("Relación entre características (scatter plot)")
import seaborn as sns
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
sns.scatterplot(data=X, x="sepal length (cm)", y="petal length (cm)", hue=y, palette="deep", ax=ax)
st.pyplot(fig)
# -----------------------------------------------
# 5. Visualizaciones adicionales del dataset
# -----------------------------------------------
st.subheader("📊 Análisis Exploratorio de Datos")

st.write("**Distribución de cada característica (histograma)**")
fig, axs = plt.subplots(2, 2, figsize=(10, 6))
axs = axs.ravel()

for idx, col in enumerate(X.columns):
    axs[idx].hist(X[col], bins=10, alpha=0.7)
    axs[idx].set_title(col)

st.pyplot(fig)

st.write("Matriz de correlación")
corr = X.corr()
fig, ax = plt.subplots()
sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)

st.write("Relación entre características (pairplot)")
import seaborn as sns
fig = sns.pairplot(pd.concat([X, pd.DataFrame(y, columns=['species'])], axis=1), hue='species')
st.pyplot(fig)
