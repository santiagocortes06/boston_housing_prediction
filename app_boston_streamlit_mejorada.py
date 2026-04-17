import joblib
import pandas as pd
import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title='Boston Housing | Predicción',
    page_icon='🏠',
    layout='wide',
    initial_sidebar_state='expanded'
)

MODEL_PATH = Path(__file__).with_name('modelo_boston_gb_ajustado.joblib')
DEFAULT_COLUMNS = [
    'crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age',
    'dis', 'tax', 'ptratio', 'black', 'lstat'
]

FIELD_CONFIG = {
    'crim': {
        'label': 'CRIM', 'help': 'Tasa de criminalidad per cápita por ciudad.',
        'default': 0.25, 'min': 0.0, 'max': 100.0, 'step': 0.01
    },
    'zn': {
        'label': 'ZN', 'help': 'Porcentaje de suelo residencial zonificado para lotes grandes.',
        'default': 0.0, 'min': 0.0, 'max': 100.0, 'step': 1.0
    },
    'indus': {
        'label': 'INDUS', 'help': 'Proporción de acres de negocio no minorista por ciudad.',
        'default': 9.8, 'min': 0.0, 'max': 30.0, 'step': 0.1
    },
    'chas': {
        'label': 'CHAS', 'help': 'Cercanía al río Charles: 1 = sí, 0 = no.',
        'default': 0
    },
    'nox': {
        'label': 'NOX', 'help': 'Concentración de óxidos de nitrógeno.',
        'default': 0.54, 'min': 0.3, 'max': 1.0, 'step': 0.01
    },
    'rm': {
        'label': 'RM', 'help': 'Número promedio de habitaciones por vivienda.',
        'default': 6.2, 'min': 3.0, 'max': 9.0, 'step': 0.1
    },
    'age': {
        'label': 'AGE', 'help': 'Porcentaje de viviendas construidas antes de 1940.',
        'default': 77.0, 'min': 0.0, 'max': 100.0, 'step': 1.0
    },
    'dis': {
        'label': 'DIS', 'help': 'Distancia ponderada a centros de empleo de Boston.',
        'default': 3.3, 'min': 0.0, 'max': 15.0, 'step': 0.1
    },
    'tax': {
        'label': 'TAX', 'help': 'Tasa de impuesto predial por 10,000 dólares.',
        'default': 330.0, 'min': 100.0, 'max': 800.0, 'step': 1.0
    },
    'ptratio': {
        'label': 'PTRATIO', 'help': 'Relación alumno-profesor por ciudad.',
        'default': 19.0, 'min': 10.0, 'max': 25.0, 'step': 0.1
    },
    'black': {
        'label': 'BLACK', 'help': 'Indicador socioeconómico transformado del dataset original.',
        'default': 392.0, 'min': 0.0, 'max': 400.0, 'step': 0.1
    },
    'lstat': {
        'label': 'LSTAT', 'help': 'Porcentaje de población de menor estatus.',
        'default': 11.2, 'min': 0.0, 'max': 40.0, 'step': 0.1
    }
}

MODEL_SUMMARY = {
    'Modelo final': 'GradientBoostingRegressor Ajustado',
    'MAE': '1.9955',
    'RMSE': '2.8158',
    'R²': '0.9157',
    'MAPE': '10.0365%'
}

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.1rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        color: #4b5563;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #f8fafc 0%, #eef2ff 100%);
        border: 1px solid #e5e7eb;
        border-radius: 16px;
        padding: 1rem;
    }
    .small-note {
        font-size: 0.9rem;
        color: #6b7280;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_package(path: Path):
    return joblib.load(path)


def init_state():
    if 'defaults_loaded' not in st.session_state:
        for k, cfg in FIELD_CONFIG.items():
            st.session_state[k] = cfg['default']
        st.session_state['defaults_loaded'] = True


def load_example_low():
    sample = {
        'crim': 0.08, 'zn': 18.0, 'indus': 2.3, 'chas': 0, 'nox': 0.45,
        'rm': 6.3, 'age': 42.0, 'dis': 4.9, 'tax': 280.0, 'ptratio': 17.8,
        'black': 392.0, 'lstat': 7.5
    }
    st.session_state.update(sample)


def load_example_high():
    sample = {
        'crim': 0.03, 'zn': 35.0, 'indus': 3.0, 'chas': 1, 'nox': 0.42,
        'rm': 7.6, 'age': 35.0, 'dis': 6.0, 'tax': 260.0, 'ptratio': 16.5,
        'black': 395.0, 'lstat': 4.0
    }
    st.session_state.update(sample)


def reset_defaults():
    for k, cfg in FIELD_CONFIG.items():
        st.session_state[k] = cfg['default']


def render_sidebar():
    st.sidebar.header('Panel de control')
    st.sidebar.caption('Utiliza ejemplos rápidos o restablece los valores por defecto.')
    if st.sidebar.button('Cargar ejemplo estándar'):
        load_example_low()
    if st.sidebar.button('Cargar ejemplo de vivienda alta'):
        load_example_high()
    if st.sidebar.button('Restablecer valores'):
        reset_defaults()

    st.sidebar.divider()
    st.sidebar.subheader('Resumen del modelo')
    for k, v in MODEL_SUMMARY.items():
        st.sidebar.write(f'**{k}:** {v}')

    st.sidebar.divider()
    st.sidebar.info(
        'Recomendación: usa valores dentro de rangos razonables. '
        'Entradas extremadamente alejadas del comportamiento del dataset pueden producir estimaciones menos confiables.'
    )


def build_input_form(columns):
    values = {}
    left, right = st.columns(2)
    half = len(columns) // 2
    left_cols = columns[:half]
    right_cols = columns[half:]

    for col_container, names in [(left, left_cols), (right, right_cols)]:
        with col_container:
            for name in names:
                cfg = FIELD_CONFIG[name]
                if name == 'chas':
                    values[name] = st.selectbox(
                        cfg['label'], options=[0, 1], key=name, help=cfg['help']
                    )
                else:
                    values[name] = st.number_input(
                        cfg['label'],
                        min_value=float(cfg['min']),
                        max_value=float(cfg['max']),
                        value=float(st.session_state[name]),
                        step=float(cfg['step']),
                        key=name,
                        help=cfg['help']
                    )
    return values


def render_header(model_name: str):
    st.markdown("<div class='main-title'>Predicción del valor de viviendas en Boston</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='subtitle'>Aplicación desarrollada en Streamlit para estimar el valor de una vivienda utilizando el mejor modelo del proyecto: "
        f"<b>{model_name}</b>.</div>",
        unsafe_allow_html=True,
    )


def main():
    init_state()
    render_sidebar()

    if not MODEL_PATH.exists():
        st.error(
            'No se encontró el archivo `modelo_boston_gb_ajustado.joblib` en la misma carpeta de la aplicación. '
            'Verifica que el archivo esté en el repositorio junto a `app_boston_streamlit.py`.'
        )
        st.stop()

    try:
        package = load_package(MODEL_PATH)
    except Exception as exc:
        st.error('No fue posible cargar el modelo. Revisa que el archivo `.joblib` no esté corrupto.')
        st.exception(exc)
        st.stop()

    model = package['modelo']
    columns = package.get('columnas_entrada', DEFAULT_COLUMNS)
    model_name = package.get('nombre_modelo', 'Modelo cargado')

    render_header(model_name)

    tab1, tab2, tab3 = st.tabs(['Predicción', 'Variables', 'Acerca del modelo'])

    with tab1:
        with st.form('form_prediccion'):
            st.subheader('Ingrese las características de la vivienda')
            values = build_input_form(columns)
            submitted = st.form_submit_button('Calcular predicción', use_container_width=True)

        if submitted:
            input_df = pd.DataFrame([values])[columns]
            prediction = float(model.predict(input_df)[0])

            c1, c2 = st.columns([1, 1])
            with c1:
                st.metric('Valor estimado (MEDV)', f'{prediction:.2f}')
            with c2:
                st.metric('Aproximación en dólares', f'USD {prediction * 1000:,.0f}')

            st.success('Predicción generada correctamente.')
            st.dataframe(input_df, use_container_width=True)
            st.caption('Nota: en el dataset Boston Housing, MEDV suele interpretarse en miles de dólares.')

    with tab2:
        st.subheader('Descripción de variables')
        rows = []
        for name in columns:
            cfg = FIELD_CONFIG.get(name, {})
            rows.append({
                'Variable': cfg.get('label', name),
                'Descripción': cfg.get('help', 'Sin descripción disponible.'),
                'Rango sugerido': (
                    '0 o 1' if name == 'chas' else f"{cfg.get('min', '-') } a {cfg.get('max', '-') }"
                )
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

    with tab3:
        st.subheader('Resumen técnico')
        st.write(
            'El modelo utilizado corresponde al **GradientBoostingRegressor Ajustado**, '
            'seleccionado como mejor modelo del proyecto tras comparar algoritmos clásicos y ensambles '
            'mediante métricas de regresión como MAE, RMSE, R² y MAPE.'
        )
        st.markdown(
            '- **MAE:** 1.9955  \n'
            '- **RMSE:** 2.8158  \n'
            '- **R²:** 0.9157  \n'
            '- **MAPE:** 10.0365%'
        )
        st.info(
            'Esta aplicación tiene fines académicos. La precisión depende de que los datos de entrada '
            'sean coherentes con el comportamiento del dataset utilizado en el entrenamiento.'
        )


if __name__ == '__main__':
    main()
