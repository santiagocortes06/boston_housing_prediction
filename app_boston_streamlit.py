import joblib
import pandas as pd
import streamlit as st
from pathlib import Path

st.set_page_config(page_title='Predicción Boston Housing', page_icon='🏠', layout='wide')

MODEL_PATH = Path(__file__).with_name('modelo_boston_gb_ajustado.joblib')
DEFAULT_COLUMNS = [
    'crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age',
    'dis', 'tax', 'ptratio', 'black', 'lstat'
]
DEFAULT_VALUES = {
    'crim': 0.25,
    'zn': 0.0,
    'indus': 9.8,
    'chas': 0,
    'nox': 0.54,
    'rm': 6.2,
    'age': 77.0,
    'dis': 3.3,
    'tax': 330.0,
    'ptratio': 19.0,
    'black': 392.0,
    'lstat': 11.2,
}
HELP_TEXT = {
    'crim': 'Tasa de criminalidad per cápita por ciudad.',
    'zn': 'Proporción de suelo residencial zonificado para lotes grandes.',
    'indus': 'Proporción de acres de negocio no minorista por ciudad.',
    'chas': 'Cercanía al río Charles: 1 = sí, 0 = no.',
    'nox': 'Concentración de óxidos de nitrógeno.',
    'rm': 'Número promedio de habitaciones por vivienda.',
    'age': 'Proporción de viviendas ocupadas por sus dueños construidas antes de 1940.',
    'dis': 'Distancia ponderada a centros de empleo de Boston.',
    'tax': 'Tasa de impuesto predial por 10,000 dólares.',
    'ptratio': 'Relación alumno-profesor por ciudad.',
    'black': 'Indicador socioeconómico transformado del dataset original.',
    'lstat': 'Porcentaje de población de menor estatus.',
}

@st.cache_resource

def load_package(path: Path):
    return joblib.load(path)


def build_input_form(columns):
    values = {}
    col1, col2 = st.columns(2)
    left_cols = columns[: len(columns) // 2]
    right_cols = columns[len(columns) // 2 :]

    with col1:
        for name in left_cols:
            if name == 'chas':
                values[name] = st.selectbox(
                    'chas',
                    options=[0, 1],
                    index=DEFAULT_VALUES[name],
                    help=HELP_TEXT.get(name, '')
                )
            else:
                values[name] = st.number_input(
                    name,
                    value=float(DEFAULT_VALUES.get(name, 0.0)),
                    step=0.1,
                    help=HELP_TEXT.get(name, '')
                )
    with col2:
        for name in right_cols:
            if name == 'chas':
                values[name] = st.selectbox(
                    'chas',
                    options=[0, 1],
                    index=DEFAULT_VALUES[name],
                    help=HELP_TEXT.get(name, '')
                )
            else:
                values[name] = st.number_input(
                    name,
                    value=float(DEFAULT_VALUES.get(name, 0.0)),
                    step=0.1,
                    help=HELP_TEXT.get(name, '')
                )
    return values


def main():
    st.title('Predicción del valor de viviendas en Boston')
    st.write(
        'Esta aplicación usa el **GradientBoostingRegressor Ajustado** entrenado en el proyecto '
        'para estimar el valor de una vivienda a partir de 12 variables de entrada.'
    )

    if not MODEL_PATH.exists():
        st.error(
            'No se encontró el archivo `modelo_boston_gb_ajustado.joblib` en la misma carpeta de la app. '
            'Copia ese archivo junto a este script antes de ejecutar o desplegar la aplicación.'
        )
        st.stop()

    package = load_package(MODEL_PATH)
    model = package['modelo']
    columns = package.get('columnas_entrada', DEFAULT_COLUMNS)
    model_name = package.get('nombre_modelo', 'Modelo cargado')

    st.caption(f'Modelo cargado: {model_name}')

    with st.expander('Ver descripción de variables'):
        for c in columns:
            st.write(f'**{c}**: {HELP_TEXT.get(c, "Sin descripción disponible.")}')

    with st.form('form_prediccion'):
        st.subheader('Ingrese las variables de la vivienda')
        values = build_input_form(columns)
        submitted = st.form_submit_button('Predecir valor')

    if submitted:
        input_df = pd.DataFrame([values])[columns]
        prediction = float(model.predict(input_df)[0])

        st.success('Predicción generada correctamente.')
        c1, c2 = st.columns(2)
        with c1:
            st.metric('Valor estimado (medv)', f'{prediction:.2f}')
        with c2:
            st.metric('Aproximación en dólares', f'USD {prediction * 1000:,.0f}')

        st.subheader('Datos usados en la predicción')
        st.dataframe(input_df, use_container_width=True)

        st.info(
            'Nota: en el dataset Boston Housing, `medv` suele interpretarse en miles de dólares. '
            'Ajuste esa lectura si en su curso el profesor usa otra convención.'
        )


if __name__ == '__main__':
    main()
