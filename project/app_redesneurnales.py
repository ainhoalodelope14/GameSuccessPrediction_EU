import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf

# Título
st.title("Predicción de ventas en Europa con Red Neuronal")
st.write("Carga de datos, entrenamiento del modelo y predicción con redes neuronales.")

# Cargar datos
uploaded_file = st.file_uploader("Carga el archivo CSV de ventas de videojuegos", type="csv")

if uploaded_file is not None:
    videojuegos = pd.read_csv(uploaded_file)
    videojuegos.dropna(inplace=True)

    # Codificación de variables categóricas
    le_platform = LabelEncoder()
    le_genre = LabelEncoder()
    le_publisher = LabelEncoder()

    videojuegos['Platform'] = le_platform.fit_transform(videojuegos['Platform'])
    videojuegos['Genre'] = le_genre.fit_transform(videojuegos['Genre'])
    videojuegos['Publisher'] = le_publisher.fit_transform(videojuegos['Publisher'])

    X = videojuegos[['Platform', 'Year', 'Genre', 'Publisher', 'NA_Sales', 'JP_Sales', 'Other_Sales']]
    y = videojuegos['EU_Sales']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Modelo de red neuronal
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    with st.spinner("Entrenando el modelo..."):
        history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2, verbose=0)

    # Mostrar gráfico de pérdida
    st.subheader("Gráfico de pérdida durante el entrenamiento")
    fig, ax = plt.subplots()
    ax.plot(history.history['loss'])
    ax.set_title('Pérdida durante el entrenamiento')
    ax.set_xlabel('Épocas')
    ax.set_ylabel('Loss (MSE)')
    ax.grid(True)
    st.pyplot(fig)

    # Evaluación del modelo
    y_pred_nn = model.predict(X_test).flatten()
    mse_nn = mean_squared_error(y_test, y_pred_nn)
    r2_nn = r2_score(y_test, y_pred_nn)

    st.subheader("Resultados del modelo")
    st.write(f"**MSE (Error cuadrático medio):** {mse_nn:.4f}")
    st.write(f"**R² (Coeficiente de determinación):** {r2_nn:.4f}")

    st.subheader("Predicción para nuevo videojuego")

    platform = st.selectbox("Plataforma", le_platform.classes_)
    year = st.number_input("Año", min_value=1980, max_value=2030, value=2025)
    genre = st.selectbox("Género", le_genre.classes_)
    publisher = st.selectbox("Publisher", le_publisher.classes_)
    na_sales = st.number_input("Ventas en Norteamérica (millones)", min_value=0.0, value=15.0)
    jp_sales = st.number_input("Ventas en Japón (millones)", min_value=0.0, value=2.5)
    other_sales = st.number_input("Otras ventas (millones)", min_value=0.0, value=10.3)

    if st.button("Predecir ventas en Europa"):
        nuevo_juego = {
            'Platform': le_platform.transform([platform])[0],
            'Year': year,
            'Genre': le_genre.transform([genre])[0],
            'Publisher': le_publisher.transform([publisher])[0],
            'NA_Sales': na_sales,
            'JP_Sales': jp_sales,
            'Other_Sales': other_sales
        }

        nuevo_df = pd.DataFrame([nuevo_juego])
        nuevo_df_scaled = scaler.transform(nuevo_df)

        prediccion_nn = model.predict(nuevo_df_scaled)
        st.success(f"Predicción de ventas en Europa: **{prediccion_nn[0][0]:.2f} millones**")
else:
    st.info("Por favor, carga un archivo CSV para continuar.")
