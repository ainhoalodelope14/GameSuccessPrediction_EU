# app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

st.title("📊 Predicción de Ventas en Europa de Videojuegos")

# Cargar CSV
st.sidebar.header("🔽 Carga tu archivo CSV")
uploaded_file = st.sidebar.file_uploader("Selecciona el archivo 'Video_Games_Sales.csv'", type=["csv"])

if uploaded_file:
    videojuegos = pd.read_csv(uploaded_file)

    st.subheader("Vista previa del dataset")
    st.write(videojuegos.head())

    st.write("Tamaño original:", videojuegos.shape)

    st.subheader("Valores nulos por columna")
    st.write(videojuegos.isnull().sum())

    videojuegos.dropna(inplace=True)

    st.success(f"Datos limpios. Tamaño actualizado: {videojuegos.shape}")

    # --- GRÁFICOS ---
    st.subheader("🎮 Videojuegos más vendidos a nivel global")
    fig1, ax1 = plt.subplots(figsize=(14, 8))
    sns.barplot(x=videojuegos['Global_Sales'].head(30), y=videojuegos['Name'].head(30), palette="viridis", ax=ax1)
    ax1.set_title("Ventas Globales de los Videojuegos Más Vendidos", fontsize=16)
    ax1.set_xlabel("Ventas Globales (millones)")
    ax1.set_ylabel("Título del Videojuego")
    st.pyplot(fig1)

    st.subheader("🎲 Ventas por Género")
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    sns.barplot(x='Genre', y='Global_Sales', data=videojuegos, palette='viridis', ax=ax2)
    ax2.set_title('Top-selling Genres')
    ax2.set_xlabel('Género')
    ax2.set_ylabel('Ventas Globales (millones)')
    ax2.tick_params(axis='x', rotation=45)
    st.pyplot(fig2)

    # --- PREPROCESAMIENTO ---
    le_platform = LabelEncoder()
    le_genre = LabelEncoder()
    le_publisher = LabelEncoder()

    videojuegos['Platform'] = le_platform.fit_transform(videojuegos['Platform'])
    videojuegos['Genre'] = le_genre.fit_transform(videojuegos['Genre'])
    videojuegos['Publisher'] = le_publisher.fit_transform(videojuegos['Publisher'])

    X = videojuegos[['Platform', 'Year', 'Genre', 'Publisher', 'NA_Sales', 'JP_Sales', 'Other_Sales']]
    y = videojuegos['EU_Sales']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.subheader("📈 Evaluación del Modelo")
    st.write(f"**Error Cuadrático Medio (MSE):** {mse:.2f}")
    st.write(f"**Coeficiente de Determinación (R²):** {r2:.2f}")

    # --- PREDICCIÓN PERSONALIZADA ---
    st.subheader("🧮 Predicción Personalizada de Ventas en Europa")

    plataformas = list(le_platform.classes_)
    generos = list(le_genre.classes_)
    publishers = list(le_publisher.classes_)

    col1, col2 = st.columns(2)
    with col1:
        platform_input = st.selectbox("Plataforma", plataformas)
        genre_input = st.selectbox("Género", generos)
        publisher_input = st.selectbox("Publisher", publishers)

    with col2:
        year_input = st.number_input("Año", min_value=1980, max_value=2025, value=2023)
        na_sales_input = st.number_input("Ventas en Norteamérica (millones)", min_value=0.0, value=1.0)
        jp_sales_input = st.number_input("Ventas en Japón (millones)", min_value=0.0, value=0.5)
        other_sales_input = st.number_input("Ventas en Resto del Mundo (millones)", min_value=0.0, value=0.5)

    if st.button("Predecir ventas en Europa"):
        nuevo_juego = {
            'Platform': le_platform.transform([platform_input])[0],
            'Year': year_input,
            'Genre': le_genre.transform([genre_input])[0],
            'Publisher': le_publisher.transform([publisher_input])[0],
            'NA_Sales': na_sales_input,
            'JP_Sales': jp_sales_input,
            'Other_Sales': other_sales_input
        }

        nuevo_df = pd.DataFrame([nuevo_juego])
        prediccion = model.predict(nuevo_df)
        st.success(f"🎯 Predicción de ventas en Europa: **{prediccion[0]:.2f} millones**")

else:
    st.info("🔄 Por favor, sube el archivo CSV para comenzar.")
