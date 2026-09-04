from pathlib import Path
from textwrap import dedent
import base64
import streamlit as st
import requests
import pandas as pd
from PIL import Image, ImageOps

from src.monitoring.data_quality_gates import (
    EXPECTED_COLUMNS,
    validate_batch,
    process_batch_with_gates,
)
from src.monitoring.drift_detection import (
    load_raw_data,
    build_reference_and_batches,
    evaluate_drift_for_batch,
)
from src.monitoring.model_performance import evaluate_labeled_batch
from src.monitoring.retraining_decision import decide_retraining

BASE_DIR = Path(__file__).resolve().parent
API_URL = "http://127.0.0.1:8000"
def solicitar_prediccion(payload: dict):
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=payload,
            timeout=10,
        )

        response.raise_for_status()
        return response.json(), None

    except requests.exceptions.RequestException as error:
        return None, str(error)
CSS_PATH = BASE_DIR / "assets" / "styles" / "styles.css"
HERO_PATH = BASE_DIR / "assets" / "images" / "hero_1994.png"
HEADER_VISUAL_PATH = (
    BASE_DIR / "assets" / "images" / "header_aci94_visual.png"
)

def image_to_base64(image_path: Path) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
    
HEADER_VISUAL_B64 = image_to_base64(HEADER_VISUAL_PATH)

st.set_page_config(
    page_title="ACI94 · Census Income 1994",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

def load_css(path: Path) -> None:
    st.markdown(f"<style>{path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

def image_to_data_uri(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"

load_css(CSS_PATH)
hero_uri = image_to_data_uri(HERO_PATH)

if "screen" not in st.session_state:
    st.session_state.screen = "home"

def render_flow_header(step: int | str) -> None:
    step_text = f"{step} / 3" if isinstance(step, int) else step

    st.html(
        dedent(
            f"""
            <div class="flow-header">

                <div>
                    <div class="flow-brand">
                        <span>ACI</span><strong>94</strong>
                    </div>
                    <div class="flow-sub">
                        CENSUS INCOME · 1994
                    </div>
                </div>

                <div class="flow-visual">
                    <img
                        src="data:image/png;base64,{HEADER_VISUAL_B64}"
                        alt="Visual ACI94"
                    >
                </div>

                <div class="step-number">
                    {step_text}
                </div>

            </div>
            """
        )
    )

if st.session_state.screen == "home":
    st.html(
        dedent(f"""
        <main class="aci-shell">
            <header class="aci-header">
                <div class="brand-wrap">
                    <div class="brand">ACI<span>94</span></div>
                    <div class="brand-sub">CENSUS INCOME · 1994</div>
                </div>
                <div class="header-meta">UNITED STATES · HISTORICAL DATA</div>
            </header>

            <section class="hero-grid">
                <div class="hero-copy">
                    <div class="eyebrow">ADULT CENSUS INCOME · 1994</div>
                    <h1>Explore patrones de ingreso con datos del Censo de EE. UU. de 1994</h1>

                    <p class="lead">
                        <strong>Problema de negocio:</strong> predecir si una persona pertenece
                        a la categoría de ingreso anual <strong>&gt; US$50K</strong>
                        analizando además si el modelo presenta
                        diferencias de comportamiento entre dos subgrupos.
                    </p>

                    <div class="principle">
                        <strong>Un caso histórico para estudiar ML y MLOps, no para representar la realidad actual.
                    </div>

                    <div class="feature-row">
                        <div class="feature-item">
                            <div class="icon-circle blue">▤</div>
                            <div><strong>Datos históricos</strong><span>Census Income 1994</span></div>
                        </div>
                        <div class="feature-item">
                            <div class="icon-circle burgundy">◎</div>
                            <div><strong>Perfil</strong><span>Socioeconómico y laboral</span></div>
                        </div>
                        <div class="feature-item">
                            <div class="icon-circle blue">◇</div>
                            <div><strong>Modelo</strong><span>Clasificación interactiva</span></div>
                        </div>
                        <div class="feature-item">
                            <div class="icon-circle mustard">↗</div>
                            <div><strong>Contexto</strong><span>Interpretación responsable</span></div>
                        </div>
                    </div>
                </div>

                <div class="hero-visual">
                    <img src="{hero_uri}" alt="Collage editorial inspirado en Census Income 1994">
                    <div class="year-stamp">1994</div>
                </div>
            </section>
        </main>
        """)
    )

    _, center, _ = st.columns([1, 1.45, 1])
    with center:
        if st.button("COMENZAR EXPLORACIÓN  →", use_container_width=True, type="primary"):
            st.session_state.screen = "profile"
            st.rerun()

    st.html(
        dedent("""
        <footer class="aci-footer">
            <div>ADULT CENSUS INCOME · UNITED STATES · 1994</div>
            <div>Explore el proyecto · Conozca el modelo</div>
        </footer>
        """)
    )

elif st.session_state.screen == "profile":

    render_flow_header(1)
    st.html(
        dedent("""
        <main class="aci-shell">

            <section class="flow-intro">
                <div class="eyebrow">CONSTRUYA UN PERFIL</div>
                <h1>Sobre el perfil</h1>
                <p>
                    Comencemos con algunas características generales
                    para construir el perfil que desea explorar.
                </p>
            </section>

            <div class="progress-wrap">
                <div class="progress-item active">
                    <span>1</span>
                    <strong>PERFIL</strong>
                </div>

                <div class="progress-line"></div>

                <div class="progress-item">
                    <span>2</span>
                    <strong>ACTIVIDAD</strong>
                </div>

                <div class="progress-line"></div>

                <div class="progress-item">
                    <span>3</span>
                    <strong>CONTEXTO</strong>
                </div>
            </div>
        </main>
        """)
    )

 # =========================================================
    # PASO 1 - PERFIL
    # =========================================================

    edad = st.number_input(
        "Edad",
        min_value=17,
        max_value=90,
        value=35,
        step=1,
        key="edad"
    )

    educacion = st.selectbox(
        "Nivel educativo",
        [
            "Seleccione una opción...",
            "Preescolar",
            "1.º a 4.º grado",
            "5.º a 6.º grado",
            "7.º a 8.º grado",
            "9.º grado",
            "10.º grado",
            "11.º grado",
            "12.º grado",
            "Secundaria completa",
            "Estudios universitarios parciales",
            "Formación técnica vocacional",
            "Formación técnica académica",
            "Bachillerato universitario",
            "Maestría",
            "Título profesional avanzado",
            "Doctorado",
        ],
        key="educacion"
    )

    EDUCATION_MAP = {
        "Preescolar": 1,
        "1.º a 4.º grado": 2,
        "5.º a 6.º grado": 3,
        "7.º a 8.º grado": 4,
        "9.º grado": 5,
        "10.º grado": 6,
        "11.º grado": 7,
        "12.º grado": 8,
        "Secundaria completa": 9,
        "Estudios universitarios parciales": 10,
        "Formación técnica vocacional": 11,
        "Formación técnica académica": 12,
        "Bachillerato universitario": 13,
        "Maestría": 14,
        "Título profesional avanzado": 15,
        "Doctorado": 16,
    }

    estado_civil = st.selectbox(
        "Estado civil",
        [
            "Seleccione una opción...",
            "Soltero/a",
            "Casado/a, cónyuge presente",
            "Casado/a, cónyuge ausente",
            "Casado/a con miembro de Fuerzas Armadas",
            "Separado/a",
            "Divorciado/a",
            "Viudo/a",
        ],
        key="estado_civil"
    )

    MARITAL_STATUS_MAP = {
        "Soltero/a": "Never-married",
        "Casado/a, cónyuge presente": "Married-civ-spouse",
        "Casado/a, cónyuge ausente": "Married-spouse-absent",
        "Casado/a con miembro de Fuerzas Armadas": "Married-AF-spouse",
        "Separado/a": "Separated",
        "Divorciado/a": "Divorced",
        "Viudo/a": "Widowed",
    }

    relacion = st.selectbox(
        "Relación dentro del hogar",
        [
            "Seleccione una opción...",
            "Esposo",
            "Esposa",
            "Hijo/a",
            "Otro familiar",
            "Sin vínculo familiar con el hogar",
            "Persona no casada",
        ],
        key="relacion",
        help=(
            "Se refiere a la relación de la persona con los demás "
            "integrantes del hogar según las categorías utilizadas "
            "por el Censo de 1994."
        )
    )

    RELATIONSHIP_MAP = {
        "Esposo": "Husband",
        "Esposa": "Wife",
        "Hijo/a": "Own-child",
        "Otro familiar": "Other-relative",
        "Sin vínculo familiar con el hogar": "Not-in-family",
        "Persona no casada": "Unmarried",
    }

    perfil_completo = (
        educacion != "Seleccione una opción..."
        and estado_civil != "Seleccione una opción..."
        and relacion != "Seleccione una opción..."
    )

    col_back, spacer, col_next = st.columns([1, 2, 1])

    with col_back:
        if st.button("← VOLVER", use_container_width=True):
            st.session_state.screen = "home"
            st.rerun()

    with col_next:
        if st.button(
            "CONTINUAR →",
            use_container_width=True,
            type="primary",
            disabled=not perfil_completo
        ):
            # Valores exactos que espera el modelo
            st.session_state.age_model = int(edad)
            st.session_state.education_num_model = EDUCATION_MAP[educacion]
            st.session_state.marital_status_model = MARITAL_STATUS_MAP[estado_civil]
            st.session_state.relationship_model = RELATIONSHIP_MAP[relacion]

            st.session_state.screen = "activity"
            st.rerun()

elif st.session_state.screen == "activity":

    render_flow_header(2)
    st.html(
        dedent("""
        <main class="aci-shell">

            <section class="flow-intro">
                <div class="eyebrow">CONSTRUYA UN PERFIL</div>
                <h1>Actividad laboral</h1>
                <p>
                    Ahora describamos el contexto de trabajo
                    del perfil que desea explorar.
                </p>
            </section>

            <div class="progress-wrap">
                <div class="progress-item completed">
                    <span>✓</span>
                    <strong>PERFIL</strong>
                </div>

                <div class="progress-line completed-line"></div>

                <div class="progress-item active">
                    <span>2</span>
                    <strong>ACTIVIDAD</strong>
                </div>

                <div class="progress-line"></div>

                <div class="progress-item">
                    <span>3</span>
                    <strong>CONTEXTO</strong>
                </div>
            </div>
        </main>
        """)
    )

    # =========================================================
    # PASO 2 - ACTIVIDAD LABORAL
    # =========================================================

    tipo_empleo = st.selectbox(
        "Tipo de empleador / actividad",
        [
            "Seleccione una opción...",
            "Sector privado",
            "Gobierno federal",
            "Gobierno estatal",
            "Gobierno local",
            "Trabajo independiente con empresa",
            "Trabajo independiente sin empresa",
            "Trabajo sin remuneración",
            "Sin experiencia laboral",
            "No especificado",
        ],
        key="tipo_empleo"
    )

    WORKCLASS_MAP = {
        "Sector privado": "Private",
        "Gobierno federal": "Federal-gov",
        "Gobierno estatal": "State-gov",
        "Gobierno local": "Local-gov",
        "Trabajo independiente con empresa": "Self-emp-inc",
        "Trabajo independiente sin empresa": "Self-emp-not-inc",
        "Trabajo sin remuneración": "Without-pay",
        "Sin experiencia laboral": "Never-worked",
        "No especificado": "Unknown",
    }

    ocupacion = st.selectbox(
        "Ocupación",
        [
            "Seleccione una opción...",
            "Administración / oficina",
            "Fuerzas Armadas",
            "Oficios / reparación",
            "Dirección / gerencia",
            "Agricultura / pesca",
            "Manipulación / limpieza",
            "Operación de maquinaria",
            "Otros servicios",
            "Servicio doméstico",
            "Profesional especializada",
            "Protección / seguridad",
            "Ventas",
            "Soporte técnico",
            "Transporte",
            "No especificada",
        ],
        key="ocupacion"
    )

    OCCUPATION_MAP = {
        "Administración / oficina": "Adm-clerical",
        "Fuerzas Armadas": "Armed-Forces",
        "Oficios / reparación": "Craft-repair",
        "Dirección / gerencia": "Exec-managerial",
        "Agricultura / pesca": "Farming-fishing",
        "Manipulación / limpieza": "Handlers-cleaners",
        "Operación de maquinaria": "Machine-op-inspct",
        "Otros servicios": "Other-service",
        "Servicio doméstico": "Priv-house-serv",
        "Profesional especializada": "Prof-specialty",
        "Protección / seguridad": "Protective-serv",
        "Ventas": "Sales",
        "Soporte técnico": "Tech-support",
        "Transporte": "Transport-moving",
        "No especificada": "Unknown",
    }

    horas = st.number_input(
        "Horas trabajadas por semana",
        min_value=1,
        max_value=99,
        value=40,
        step=1,
        key="horas_semana"
    )

    actividad_completa = (
        tipo_empleo != "Seleccione una opción..."
        and ocupacion != "Seleccione una opción..."
    )

    col_back, spacer, col_next = st.columns([1, 2, 1])

    with col_back:
        if st.button("← VOLVER", use_container_width=True):
            st.session_state.screen = "profile"
            st.rerun()

    with col_next:
        if st.button(
            "CONTINUAR →",
            use_container_width=True,
            type="primary",
            disabled=not actividad_completa
        ):
            # Valores exactos que espera el modelo
            st.session_state.hours_per_week_model = int(horas)
            st.session_state.workclass_model = WORKCLASS_MAP[tipo_empleo]
            st.session_state.occupation_model = OCCUPATION_MAP[ocupacion]

            st.session_state.screen = "context"
            st.rerun()

elif st.session_state.screen == "context":

    render_flow_header(3)

    st.html(
        dedent(
            """
            <section class="flow-intro">
                <div class="eyebrow">3 / 3 · CONTEXTO ECONÓMICO</div>
                <h1>Un último aspecto del perfil</h1>

                <p>
                    Estas preguntas se refieren a ganancias o pérdidas obtenidas
                    por inversiones, propiedades u otros activos.
                    <strong>No se refieren al salario habitual.</strong>
                </p>
            </section>
            """
        )
    )

    st.markdown("### Ganancias de capital")

    tiene_ganancia = st.radio(
        "Durante el año, ¿obtuvo alguna ganancia de capital?",
        options=["No", "Sí"],
        index=None,
        horizontal=True,
        key="tiene_ganancia_capital",
        help=(
            "Por ejemplo, una ganancia obtenida por la venta "
            "de una inversión, propiedad u otro activo."
        ),
    )

    if tiene_ganancia == "Sí":
        ganancia_capital = st.number_input(
            "Monto aproximado de la ganancia (US$)",
            min_value=1,
            value=100,
            step=100,
            key="ganancia_capital",
        )
    else:
        ganancia_capital = 0

    st.markdown("---")

    st.markdown("### Pérdidas de capital")

    tiene_perdida = st.radio(
        "Durante el año, ¿tuvo alguna pérdida de capital?",
        options=["No", "Sí"],
        index=None,
        horizontal=True,
        key="tiene_perdida_capital",
        help=(
            "Por ejemplo, una pérdida generada al vender "
            "una inversión, propiedad u otro activo por menos "
            "de su valor de adquisición."
        ),
    )

    if tiene_perdida == "Sí":
        perdida_capital = st.number_input(
            "Monto aproximado de la pérdida (US$)",
            min_value=1,
            value=100,
            step=100,
            key="perdida_capital",
        )
    else:
        perdida_capital = 0

    respuestas_completas = (
        tiene_ganancia is not None
        and tiene_perdida is not None
    )

    col_back, col_next = st.columns([1, 1])

    with col_back:
        if st.button(
            "← VOLVER",
            use_container_width=True,
            key="back_context",
        ):
            st.session_state.screen = "activity"
            st.rerun()

    with col_next:
        if st.button(
            "ANALIZAR PERFIL →",
            use_container_width=True,
            type="primary",
            disabled=not respuestas_completas,
            key="analyze_profile",
        ):
            # Valores exactos que posteriormente recibirá la API
            st.session_state.capital_gain_model = int(ganancia_capital)
            st.session_state.capital_loss_model = int(perdida_capital)

            st.session_state.screen = "result"
            st.rerun()

elif st.session_state.screen == "result":

    required_keys = [
        "age_model",
        "education_num_model",
        "hours_per_week_model",
        "capital_gain_model",
        "capital_loss_model",
        "workclass_model",
        "marital_status_model",
        "occupation_model",
        "relationship_model",
    ]

    if not all(key in st.session_state for key in required_keys):
        st.warning(
            "Faltan datos del perfil. Por favor complete nuevamente el recorrido."
        )
        st.session_state.screen = "profile"
        st.rerun()

    payload = {
        "age": st.session_state.age_model,
        "education-num": st.session_state.education_num_model,
        "hours-per-week": st.session_state.hours_per_week_model,
        "capital-gain": st.session_state.capital_gain_model,
        "capital-loss": st.session_state.capital_loss_model,
        "workclass": st.session_state.workclass_model,
        "marital-status": st.session_state.marital_status_model,
        "occupation": st.session_state.occupation_model,
        "relationship": st.session_state.relationship_model,
    }

    resultado, error = solicitar_prediccion(payload)

    if error:
        st.error(
            "No fue posible obtener la predicción del modelo. "
            "Verifique que la API esté disponible."
        )
        st.stop()

    clasificacion = resultado["prediction"]
    probabilidad = resultado["probability"]

    clasificacion_mostrada = (
        "> US$50K"
        if clasificacion == ">50K"
        else "≤ US$50K"
    )

    probabilidad_mostrada = f"{probabilidad * 100:.1f}%"

    render_flow_header("RESULTADO")

    st.html(
        dedent(f"""
        <main class="aci-shell">

            <section class="result-intro">

                <div class="eyebrow">
                    EXPLORACIÓN COMPLETADA
                </div>

                <h1>Según los patrones de 1994</h1>

                <p>
                    El perfil que construyó sería clasificado por
                    nuestro modelo dentro de la siguiente categoría:
                </p>

            </section>


            <section class="result-card">

                <div class="result-label">
                    CLASIFICACIÓN ESTIMADA
                </div>

                <div class="result-value">
                    {clasificacion_mostrada}
                </div>

                <div class="result-divider"></div>

                <div class="probability-label">
                    PROBABILIDAD ESTIMADA
                </div>

                <div class="probability-value">
                    {probabilidad_mostrada}
                </div>

                <p class="result-note">
                    Probabilidad estimada por el modelo de pertenecer
                    a la categoría de ingreso &gt; US$50K.
                </p>

            </section>


            <section class="result-context">

                <strong>¿Qué significa este resultado?</strong>

                <p>
                    El modelo identifica patrones estadísticos históricos.
                    Esta estimación no representa una evaluación personal,
                    una recomendación salarial ni una relación causal.
                </p>

            </section>

        </main>
        """)
    )

    col_restart, spacer, col_more = st.columns([1, .35, 1])

    with col_restart:
        if st.button(
            "EXPLORAR OTRO PERFIL",
            use_container_width=True
        ):
            st.session_state.screen = "profile"
            st.rerun()

    with col_more:
        if st.button(
            "CONOCER ACI94  →",
            use_container_width=True,
            type="primary",
        ):
            st.session_state.screen = "demo_01"
            st.rerun()

elif st.session_state.screen == "demo_01":

    st.html(
        """
        <style>
        .demo-page {
            max-width: 1280px;
            margin: 0 auto;
            padding: 0.5rem 1rem 2rem 1rem;
        }

        .demo-title {
            text-align: center;
            color: #102F46;
            font-size: clamp(2rem, 4vw, 3.6rem);
            font-weight: 800;
            line-height: 1.05;
            margin-bottom: 0.4rem;
        }

        .demo-subtitle {
            text-align: center;
            color: #40515B;
            font-size: 1.05rem;
            margin-bottom: 2rem;
        }

        .pipeline-demo {
            display: grid;
            grid-template-columns: repeat(7, 1fr);
            gap: 0.7rem;
            align-items: stretch;
        }

        .pipeline-card {
            background: rgba(250, 244, 226, 0.78);
            border: 1px solid rgba(16, 47, 70, 0.22);
            border-radius: 10px;
            padding: 1rem 0.7rem;
            text-align: center;
            min-height: 175px;
        }

        .pipeline-number {
            width: 32px;
            height: 32px;
            margin: 0 auto 0.7rem auto;
            border-radius: 50%;
            background: #102F46;
            color: #F7EED6;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 700;
        }

        .pipeline-label {
            color: #102F46;
            font-weight: 800;
            font-size: 0.9rem;
            min-height: 40px;
        }

        .pipeline-value {
            color: #A84F2D;
            font-size: 1.45rem;
            font-weight: 800;
            margin-top: 0.7rem;
        }

        .pipeline-detail {
            color: #4E5C56;
            font-size: 0.78rem;
            margin-top: 0.35rem;
            line-height: 1.25;
        }

        .evidence-title {
            margin-top: 1.5rem;
            margin-bottom: 0.8rem;
            text-align: center;
            color: #102F46;
            font-size: 1.2rem;
            font-weight: 800;
            letter-spacing: 0.05em;
        }

        .evidence-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 0.8rem;
        }

        .evidence-card {
            background: rgba(250, 244, 226, 0.82);
            border: 1px solid rgba(16, 47, 70, 0.22);
            border-radius: 10px;
            padding: 1rem;
            text-align: center;
        }

        .evidence-main {
            color: #102F46;
            font-size: 1.2rem;
            font-weight: 800;
        }

        .evidence-accent {
            color: #647343;
            font-size: 1.55rem;
            font-weight: 800;
            margin: 0.2rem 0;
        }

        .evidence-small {
            color: #4E5C56;
            font-size: 0.82rem;
        }

        .demo-closing {
            margin: 1rem auto 0 auto;
            max-width: 850px;
            text-align: center;
            color: #102F46;
            font-weight: 650;
            font-size: 0.95rem;
            padding: 0.7rem 1rem;
            border-top: 1px solid rgba(16, 47, 70, 0.22);
        }

        @media (max-width: 900px) {
            .pipeline-demo {
                grid-template-columns: 1fr;
            }

            .pipeline-card {
                min-height: auto;
                text-align: left;
                padding: 0.9rem 1rem;
            }

            .pipeline-number {
                margin: 0 0 0.5rem 0;
            }

            .pipeline-label {
                min-height: auto;
            }

            .evidence-grid {
                grid-template-columns: 1fr 1fr;
            }
        }

        @media (max-width: 520px) {
            .evidence-grid {
                grid-template-columns: 1fr;
            }

            .demo-page {
                padding-left: 0.2rem;
                padding-right: 0.2rem;
            }
        }
        </style>

        <div class="demo-page">

            <div class="demo-title">
                ¿Cómo llegamos hasta esta predicción?
            </div>

            <div class="demo-subtitle">
                Un recorrido trazable desde los datos originales hasta el modelo en producción.
            </div>

            <div class="pipeline-demo">

                <div class="pipeline-card">
                    <div class="pipeline-number">1</div>
                    <div class="pipeline-label">DATOS RAW</div>
                    <div class="pipeline-value">48,842</div>
                    <div class="pipeline-detail">
                        registros originales<br>UCI Adult Census Income
                    </div>
                </div>

                <div class="pipeline-card">
                    <div class="pipeline-number">2</div>
                    <div class="pipeline-label">VALIDACIÓN</div>
                    <div class="pipeline-value">7 / 7 PASS</div>
                    <div class="pipeline-detail">
                        Quality Gates<br>sobre datos limpios
                    </div>
                </div>

                <div class="pipeline-card">
                    <div class="pipeline-number">3</div>
                    <div class="pipeline-label">DATOS CLEAN</div>
                    <div class="pipeline-value">48,790</div>
                    <div class="pipeline-detail">
                        registros validados<br>y listos para modelar
                    </div>
                </div>

                <div class="pipeline-card">
                    <div class="pipeline-number">4</div>
                    <div class="pipeline-label">PREPARACIÓN</div>
                    <div class="pipeline-value">Pipeline</div>
                    <div class="pipeline-detail">
                        imputación · codificación<br>transformaciones
                    </div>
                </div>

                <div class="pipeline-card">
                    <div class="pipeline-number">5</div>
                    <div class="pipeline-label">ENTRENAMIENTO</div>
                    <div class="pipeline-value">5-Fold CV</div>
                    <div class="pipeline-detail">
                        Random Forest<br>validación estratificada
                    </div>
                </div>

                <div class="pipeline-card">
                    <div class="pipeline-number">6</div>
                    <div class="pipeline-label">MLFLOW</div>
                    <div class="pipeline-value">Tracking</div>
                    <div class="pipeline-detail">
                        experimentos, métricas<br>y trazabilidad
                    </div>
                </div>

                <div class="pipeline-card">
                    <div class="pipeline-number">7</div>
                    <div class="pipeline-label">PRODUCTION</div>
                    <div class="pipeline-value">Production v1</div>
                    <div class="pipeline-detail">
                        modelo registrado<br>y versionado
                    </div>
                </div>

            </div>

            <div class="evidence-title">
                EVIDENCIA PRINCIPAL
            </div>

            <div class="evidence-grid">

                <div class="evidence-card">
                    <div class="evidence-main">CALIDAD</div>
                    <div class="evidence-accent">7 / 7 PASS</div>
                    <div class="evidence-small">Quality Gates superados</div>
                </div>

                <div class="evidence-card">
                    <div class="evidence-main">CRITERIO PRINCIPAL</div>
                    <div class="evidence-accent">G-Mean 0.8374</div>
                    <div class="evidence-small">desempeño sobre test</div>
                </div>

                <div class="evidence-card">
                    <div class="evidence-main">MODELO FINAL</div>
                    <div class="evidence-accent">Random Forest</div>
                    <div class="evidence-small">v2_without_sensitive</div>
                </div>
               
                <div class="evidence-card">
                    <div class="evidence-main">TRAZABILIDAD</div>
                    <div class="evidence-accent">Production v1</div>
                    <div class="evidence-small">registrado mediante MLflow</div>
                </div>

            </div>

            <div style="
                margin-top: 18px;
                padding: 18px 22px;
                border: 1px solid #c9c2b3;
                border-radius: 8px;
                background: rgba(255,255,255,0.20);
            ">

                <div style="
                    text-align:center;
                    font-weight:700;
                    color:#082f49;
                    font-size:15px;
                    margin-bottom:4px;
                ">
                    ANÁLISIS POR SUBGRUPOS · TEST RESERVADO
                </div>

                <div style="
                    text-align:center;
                    font-size:12px;
                    margin-bottom:14px;
                    color:#555;
                ">
                    El modelo fue auditado comparando su comportamiento entre Female y Male.
                </div>

                <div style="
                    display:grid;
                    grid-template-columns: 1.4fr 1fr 1fr 1fr;
                    gap:8px;
                    text-align:center;
                    align-items:center;
                    font-size:12px;
                ">

                    <div><strong>MÉTRICA</strong></div>
                    <div><strong>FEMALE</strong></div>
                    <div><strong>MALE</strong></div>
                    <div><strong>BRECHA</strong></div>

                    <div style="text-align:left;"><strong>Recall</strong></div>
                    <div>0.7572</div>
                    <div>0.8874</div>
                    <div>0.1302</div>

                    <div style="text-align:left;"><strong>Especificidad</strong></div>
                    <div>0.9453</div>
                    <div>0.7207</div>
                    <div>0.2246</div>

                    <div style="text-align:left;"><strong>G-Mean</strong></div>
                    <div>0.8460</div>
                    <div>0.7997</div>
                    <div>0.0463</div>

                </div>

                <div style="
                    margin-top:14px;
                    padding-top:12px;
                    border-top:1px solid #d7d0c2;
                    font-size:12px;
                    line-height:1.45;
                ">
                    <strong>Hallazgo:</strong>
                    el modelo presenta diferencias de comportamiento entre ambos subgrupos:
                    mayor recall para Male y mayor especificidad para Female.
                    Estas diferencias requieren seguimiento y no demuestran por sí solas discriminación.
                </div>

            </div>

            <div class="demo-closing">
                La predicción que usted acaba de ver proviene de datos validados y de un modelo
                evaluado, registrado y versionado. Su comportamiento también fue analizado
                entre los subgrupos Female y Male.
            </div>

        </div>
        """,
    )

    col_new, col_next = st.columns([1, 1])

    with col_new:
        if st.button(
            "← NUEVA PREDICCIÓN",
            use_container_width=True
        ):
            st.session_state.screen = "profile"
            st.rerun()

    with col_next:
        if st.button(
            "¿CÓMO FUNCIONA LA PREDICCIÓN? →",
            use_container_width=True,
            type="primary"
        ):
            st.session_state.screen = "demo_02"
            st.rerun()

    nav_back, nav_next = st.columns(2)

    with nav_back:
        if st.button(
            "← VOLVER AL RESULTADO",
            use_container_width=True,
            key="demo01_back",
        ):
            st.session_state.screen = "result"
            st.rerun()

            st.session_state.screen = "demo_02"
            st.rerun()

elif st.session_state.screen == "demo_02":

    st.html(
        """
        <style>
        .serving-page {
            max-width: 1180px;
            margin: 0 auto;
            padding: 0.5rem 1rem 1rem 1rem;
        }

        .serving-title {
            text-align: center;
            color: #102F46;
            font-size: clamp(2rem, 4vw, 3.6rem);
            font-weight: 800;
            line-height: 1.05;
            margin-bottom: 0.4rem;
        }

        .serving-subtitle {
            text-align: center;
            color: #40515B;
            font-size: 1.05rem;
            margin-bottom: 2rem;
        }

        .serving-flow {
            display: grid;
            grid-template-columns: 1fr auto 1fr auto 1fr auto 1fr;
            gap: 0.55rem;
            align-items: center;
        }

        .serving-card {
            background: rgba(250, 244, 226, 0.80);
            border: 1px solid rgba(16, 47, 70, 0.22);
            border-radius: 10px;
            padding: 1.2rem 0.8rem;
            text-align: center;
            min-height: 175px;
        }

        .serving-step {
            color: #A84F2D;
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0.06em;
            margin-bottom: 0.55rem;
        }

        .serving-name {
            color: #102F46;
            font-size: 1.15rem;
            font-weight: 800;
            margin-bottom: 0.6rem;
        }

        .serving-main {
            color: #647343;
            font-size: 1.15rem;
            font-weight: 800;
            margin-bottom: 0.35rem;
        }

        .serving-detail {
            color: #4E5C56;
            font-size: 0.82rem;
            line-height: 1.3;
        }

        .serving-arrow {
            color: #A84F2D;
            font-size: 1.8rem;
            font-weight: 800;
            text-align: center;
        }

        .docker-box {
            max-width: 720px;
            margin: 1.5rem auto 0 auto;
            padding: 0.9rem 1rem;
            border: 1px dashed rgba(100, 115, 67, 0.75);
            border-radius: 10px;
            text-align: center;
            background: rgba(100, 115, 67, 0.07);
        }

        .docker-title {
            color: #102F46;
            font-weight: 800;
            font-size: 0.95rem;
            margin-bottom: 0.25rem;
        }

        .docker-content {
            color: #647343;
            font-weight: 800;
            font-size: 1.05rem;
        }

        .docker-detail {
            color: #4E5C56;
            font-size: 0.8rem;
            margin-top: 0.25rem;
        }

        .serving-evidence-title {
            margin-top: 1.5rem;
            margin-bottom: 0.8rem;
            text-align: center;
            color: #102F46;
            font-size: 1.2rem;
            font-weight: 800;
            letter-spacing: 0.05em;
        }

        .serving-evidence {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 0.8rem;
        }

        .serving-evidence-card {
            background: rgba(250, 244, 226, 0.82);
            border: 1px solid rgba(16, 47, 70, 0.22);
            border-radius: 10px;
            padding: 0.9rem;
            text-align: center;
        }

        .serving-evidence-label {
            color: #102F46;
            font-size: 0.82rem;
            font-weight: 800;
        }

        .serving-evidence-value {
            color: #647343;
            font-size: 1.1rem;
            font-weight: 800;
            margin-top: 0.3rem;
        }

        .serving-closing {
            margin: 1rem auto 0 auto;
            max-width: 850px;
            text-align: center;
            color: #102F46;
            font-weight: 600;
            font-size: 0.95rem;
            padding: 0.7rem 1rem;
            border-top: 1px solid rgba(16, 47, 70, 0.22);
        }

        @media (max-width: 900px) {
            .serving-flow {
                grid-template-columns: 1fr;
            }

            .serving-arrow {
                transform: rotate(90deg);
                font-size: 1.4rem;
            }

            .serving-card {
                min-height: auto;
            }

            .serving-evidence {
                grid-template-columns: 1fr 1fr;
            }
        }

        @media (max-width: 520px) {
            .serving-evidence {
                grid-template-columns: 1fr;
            }

            .serving-page {
                padding-left: 0.2rem;
                padding-right: 0.2rem;
            }
        }
        </style>

        <div class="serving-page">

            <div class="serving-title">
                ¿Cómo funciona la predicción?
            </div>

            <div class="serving-subtitle">
                De los datos ingresados por el usuario a una respuesta del modelo en producción.
            </div>

            <div class="serving-flow">

                <div class="serving-card">
                    <div class="serving-step">01 · ENTRADA</div>
                    <div class="serving-name">INTERFAZ</div>
                    <div class="serving-main">Streamlit</div>
                    <div class="serving-detail">
                        El usuario ingresa las variables requeridas para generar la predicción.
                    </div>
                </div>

                <div class="serving-arrow">→</div>

                <div class="serving-card">
                    <div class="serving-step">02 · SOLICITUD</div>
                    <div class="serving-name">FASTAPI</div>
                    <div class="serving-main">POST /predict</div>
                    <div class="serving-detail">
                        Valida el esquema de entrada y envía los datos al servicio del modelo.
                    </div>
                </div>

                <div class="serving-arrow">→</div>

                <div class="serving-card">
                    <div class="serving-step">03 · INFERENCIA</div>
                    <div class="serving-name">PRODUCTION v1</div>
                    <div class="serving-main">Random Forest</div>
                    <div class="serving-detail">
                        El pipeline transforma los datos y el modelo genera la predicción.
                    </div>
                </div>

                <div class="serving-arrow">→</div>

                <div class="serving-card">
                    <div class="serving-step">04 · RESPUESTA</div>
                    <div class="serving-name">RESULTADO</div>
                    <div class="serving-main">Predicción + probabilidad</div>
                    <div class="serving-detail">
                        FastAPI devuelve la respuesta y Streamlit la presenta al usuario.
                    </div>
                </div>

            </div>

            <div class="docker-box">
                <div class="docker-title">ENTORNO REPRODUCIBLE</div>
                <div class="docker-content">Docker · FastAPI + Production v1</div>
                <div class="docker-detail">
                    El contenedor encapsula el servicio de inferencia y sus dependencias.
                </div>
            </div>

            <div class="serving-evidence-title">
                EVIDENCIA PRINCIPAL
            </div>

            <div class="serving-evidence">

                <div class="serving-evidence-card">
                    <div class="serving-evidence-label">MODELO VERSIONADO</div>
                    <div class="serving-evidence-value">Production v1 ✓</div>
                </div>

                <div class="serving-evidence-card">
                    <div class="serving-evidence-label">CONTENEDOR</div>
                    <div class="serving-evidence-value">Docker ✓</div>
                </div>

                <div class="serving-evidence-card">
                    <div class="serving-evidence-label">API FUNCIONAL</div>
                    <div class="serving-evidence-value">FastAPI ✓</div>
                </div>

                <div class="serving-evidence-card">
                    <div class="serving-evidence-label">ENTRADA VALIDADA</div>
                    <div class="serving-evidence-value">Schema API ✓</div>
                </div>

            </div>

            <div class="serving-closing">
                Una misma versión del modelo, servida mediante API y ejecutada
                en un entorno reproducible.
            </div>

        </div>
        """
    )

    col_new, col_next = st.columns([1, 1])

    with col_new:
        if st.button(
            "← NUEVA PREDICCIÓN",
            use_container_width=True,
            key="demo02_new"
        ):
            st.session_state.screen = "profile"
            st.rerun()

    with col_next:
        if st.button(
            "¿QUÉ PASA DESPUÉS DE LA PREDICCIÓN? →",
            use_container_width=True,
            type="primary",
            key="demo02_next"
        ):
            st.session_state.screen = "demo_03"
            st.rerun()

    nav_back, nav_next = st.columns(2)

    with nav_back:
        if st.button(
            "← ANTERIOR",
            use_container_width=True,
            key="demo02_back",
        ):
            st.session_state.screen = "demo_01"
            st.rerun()

            st.session_state.screen = "demo_03"
            st.rerun()

elif st.session_state.screen == "demo_03":

    # ============================================================
    # DEMO 03 · MONITORING
    # ============================================================

    st.markdown(
        "<h1 style='text-align:center; color:#102F46;'>"
        "¿Qué pasa después de la predicción?"
        "</h1>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<p style='text-align:center; color:#40515B; font-size:1.05rem;'>"
        "El modelo ya está en producción. Ahora debemos comprobar "
        "si los datos cambian y si el modelo continúa respondiendo adecuadamente."
        "</p>",
        unsafe_allow_html=True,
    )

    st.info(
        "Esta pantalla responde tres preguntas: "
        "¿cambiaron los datos?, ¿el modelo sigue rindiendo? "
        "y ¿qué recomienda hacer el sistema?"
    )

    tab_scenarios, tab_batch = st.tabs(
        [
            "PROBAR ESCENARIOS",
            "EVALUAR DATOS NUEVOS",
        ]
    )

    # ============================================================
    # TAB 1 · ESCENARIOS DE MONITOREO
    # ============================================================

    with tab_scenarios:

        st.markdown("### Simule qué podría ocurrir en producción")

        scenario = st.radio(
            "Seleccione un escenario:",
            [
                "Normal",
                "Cambio moderado",
                "Drift fuerte",
            ],
            horizontal=True,
            key="monitoring_scenario",
        )

        raw_df = load_raw_data()

        monitoring_batches = build_reference_and_batches(
            raw_df
        )

        reference = monitoring_batches["reference"]

        scenario_config = {
            "Normal": {
                "batch": monitoring_batches["lote_1_normal"],
                "current_gmean": 0.83,
            },
            "Cambio moderado": {
                "batch": monitoring_batches["lote_2_moderado"],
                "current_gmean": 0.81,
            },
            "Drift fuerte": {
                "batch": monitoring_batches["lote_3_fuerte"],
                "current_gmean": 0.72,
            },
        }

        selected = scenario_config[scenario]

        drift_result = evaluate_drift_for_batch(
            reference,
            selected["batch"],
        )

        current_gmean = selected["current_gmean"]
        baseline_gmean = 0.8374

        retraining_result = decide_retraining(
            max_psi=drift_result["max_psi"],
            baseline_metric=baseline_gmean,
            current_metric=current_gmean,
        )

        # --------------------------------------------------------
        # TRADUCCIÓN DE RESULTADOS
        # --------------------------------------------------------

        drift_status = drift_result["status"]

        if drift_status == "OK":
            drift_human = "ESTABLE"

        elif drift_status == "WARNING":
            drift_human = "CAMBIO MODERADO"

        else:
            drift_human = "CAMBIO FUERTE"

        performance_drop = (
            baseline_gmean - current_gmean
        )

        if performance_drop >= 0.05:
            performance_human = "DISMINUYÓ"

        else:
            performance_human = "ESTABLE"

        # --------------------------------------------------------
        # TRES PREGUNTAS PRINCIPALES
        # --------------------------------------------------------

        st.write("")

        col1, col2, col3 = st.columns(3)

        with col1:

            st.markdown(
                "#### ¿LOS DATOS CAMBIARON?"
            )

            st.markdown(
                f"### {drift_human}"
            )

            st.metric(
                "PSI máximo",
                f"{drift_result['max_psi']:.4f}",
            )

            st.caption(
                "PSI compara los datos nuevos con los datos "
                "de referencia utilizados por el sistema."
            )

        with col2:

            st.markdown(
                "#### ¿EL MODELO SIGUE RINDIENDO?"
            )

            st.markdown(
                f"### {performance_human}"
            )

            st.metric(
                "G-Mean actual",
                f"{current_gmean:.4f}",
                delta=(
                    f"{current_gmean - baseline_gmean:+.4f}"
                ),
            )

            st.caption(
                f"Referencia Production v1: "
                f"{baseline_gmean:.4f}"
            )

        with col3:

            st.markdown(
                "#### ¿QUÉ RECOMIENDA EL SISTEMA?"
            )

            st.markdown(
                f"### {retraining_result.decision}"
            )

            st.caption(
                "La recomendación combina la señal de drift "
                "con el comportamiento del modelo."
            )

        # --------------------------------------------------------
        # EVIDENCIA TÉCNICA
        # --------------------------------------------------------

        st.divider()

        st.markdown(
            "### ¿Qué variables están cambiando?"
        )

        drift_table = pd.DataFrame(
            [
                {
                    "Variable": variable,
                    "PSI": round(psi, 4),
                }
                for variable, psi
                in drift_result[
                    "psi_per_column"
                ].items()
            ]
        )

        st.dataframe(
            drift_table,
            use_container_width=True,
            hide_index=True,
        )

        with st.expander(
            "Ver explicación técnica de la decisión"
        ):

            st.write(
                retraining_result.reason
            )

            st.write(
                "PSI permite detectar cambios en la distribución "
                "de los datos. G-Mean permite observar el equilibrio "
                "del desempeño del clasificador entre ambas clases."
            )

        st.caption(
            "En estos escenarios, el PSI se calcula con el módulo real "
            "de detección de drift. Los valores de G-Mean representan "
            "escenarios controlados para demostrar la lógica de decisión."
        )

    # ============================================================
    # TAB 2 · EVALUAR DATOS NUEVOS
    # ============================================================

    with tab_batch:

        st.markdown(
            "### Evaluar datos no vistos previamente"
        )

        st.write(
            "Esta opción permite analizar un nuevo conjunto "
            "de datos como si acabara de llegar a producción."
        )

        uploaded_batch = st.file_uploader(
            "Seleccione un archivo CSV",
            type=["csv"],
            key="monitoring_batch_upload",
        )

        if uploaded_batch is not None:

            try:

                new_batch = pd.read_csv(
                    uploaded_batch
                )

                st.success(
                    f"Batch recibido: "
                    f"{len(new_batch):,} registros"
                )

                st.dataframe(
                    new_batch.head(10),
                    use_container_width=True,
                )

                # ------------------------------------------------
                # DATA QUALITY
                # ------------------------------------------------

                st.markdown(
                    "### 1 · ¿Los datos tienen calidad suficiente?"
                )

                features_only = new_batch.drop(
                    columns=["income"],
                    errors="ignore",
                )

                incidents = validate_batch(
                    features_only
                )

                block_count = sum(
                    incident["severity"] == "BLOCK"
                    for incident in incidents
                )

                warn_count = sum(
                    incident["severity"] == "WARN"
                    for incident in incidents
                )

                if block_count > 0:
                    quality_status = "BLOCKED"

                elif warn_count > 0:
                    quality_status = (
                        "ACCEPTED_WITH_WARNINGS"
                    )

                else:
                    quality_status = "ACCEPTED"

                q1, q2, q3 = st.columns(3)

                q1.metric(
                    "Estado",
                    quality_status,
                )

                q2.metric(
                    "Bloqueos",
                    block_count,
                )

                q3.metric(
                    "Advertencias",
                    warn_count,
                )

                if incidents:

                    st.dataframe(
                        pd.DataFrame(incidents),
                        use_container_width=True,
                        hide_index=True,
                    )

                else:

                    st.success(
                        "El batch supera los controles "
                        "de calidad."
                    )

                # ------------------------------------------------
                # DRIFT
                # ------------------------------------------------

                if quality_status != "BLOCKED":

                    st.markdown(
                        "### 2 · ¿Los datos están cambiando?"
                    )

                    raw_reference = load_raw_data()

                    prepared_batches = (
                        build_reference_and_batches(
                            raw_reference
                        )
                    )

                    reference_batch = (
                        prepared_batches["reference"]
                    )

                    try:

                        drift_result_new = (
                            evaluate_drift_for_batch(
                                reference_batch,
                                features_only,
                            )
                        )

                        new_status = (
                            drift_result_new["status"]
                        )

                        if new_status == "OK":
                            new_status_human = "ESTABLE"

                        elif new_status == "WARNING":
                            new_status_human = (
                                "CAMBIO MODERADO"
                            )

                        else:
                            new_status_human = (
                                "CAMBIO FUERTE"
                            )

                        d1, d2 = st.columns(2)

                        d1.metric(
                            "Resultado",
                            new_status_human,
                        )

                        d2.metric(
                            "PSI máximo",
                            (
                                f"{drift_result_new['max_psi']:.4f}"
                            ),
                        )

                        drift_new_table = pd.DataFrame(
                            [
                                {
                                    "Variable": variable,
                                    "PSI": round(psi, 4),
                                }
                                for variable, psi
                                in drift_result_new[
                                    "psi_per_column"
                                ].items()
                            ]
                        )

                        st.dataframe(
                            drift_new_table,
                            use_container_width=True,
                            hide_index=True,
                        )

                        # ----------------------------------------
                        # PERFORMANCE
                        # ----------------------------------------

                        st.markdown(
                            "### 3 · ¿El modelo sigue rindiendo?"
                        )

                        if "income" not in new_batch.columns:

                            st.info(
                                "Todavía no hay etiquetas reales "
                                "(ground truth). Podemos medir calidad "
                                "y drift inmediatamente, pero el desempeño "
                                "del modelo se evaluará cuando conozcamos "
                                "los resultados reales."
                            )

                        else:

                            st.info(
                                "El batch contiene ground truth. "
                                "Para evaluar desempeño se deben comparar "
                                "estas etiquetas reales con las predicciones "
                                "generadas por Production v1."
                            )

                    except Exception as drift_error:

                        st.warning(
                            "El batch superó los controles "
                            "de calidad, pero no fue posible "
                            "calcular drift."
                        )

                        st.code(
                            str(drift_error)
                        )

                else:

                    st.error(
                        "El batch fue bloqueado por los controles "
                        "de calidad. No continúa hacia inferencia "
                        "ni monitoreo hasta corregir los problemas "
                        "críticos."
                    )

            except Exception as batch_error:

                st.error(
                    "No fue posible procesar el archivo."
                )

                st.code(
                    str(batch_error)
                )

    # ============================================================
    # NAVEGACIÓN
    # ============================================================

    st.divider()

    nav_back, nav_next = st.columns(2)

    with nav_back:

        if st.button(
            "← ANTERIOR",
            use_container_width=True,
            key="demo03_back",
        ):
            st.session_state.screen = "demo_02"
            st.rerun()

    with nav_next:

        if st.button(
            "CONOZCA AL EQUIPO →",
            use_container_width=True,
            type="primary",
            key="demo03_team",
        ):
            st.session_state.screen = "demo_04"
            st.rerun()

elif st.session_state.screen == "demo_04":

    # ============================================================
    # DEMO 04 · EQUIPO
    # ============================================================

    st.markdown(
        "<h1 style='text-align:center; color:#102F46; margin-bottom:0.3rem;'>"
        "¿Quién hizo posible este proyecto?"
        "</h1>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<p style='text-align:center; color:#40515B; font-size:1rem;'>"
        "Tres personas, un proceso compartido y una misma "
        "responsabilidad sobre el resultado."
        "</p>",
        unsafe_allow_html=True,
    )

    st.write("")

    # ------------------------------------------------------------
    # PREPARAR FOTOS CON EL MISMO FORMATO
    # ------------------------------------------------------------

    foto_naomy_path = (
        BASE_DIR / "assets" / "images" / "foto_nao.png"
    )

    foto_dalay_path = (
        BASE_DIR / "assets" / "images" / "foto_day.png"
    )

    foto_vladimir_path = (
        BASE_DIR / "assets" / "images" / "foto_vlad.png"
    )

    def preparar_foto_equipo(image_path):

        image = Image.open(image_path).convert("RGB")

        return ImageOps.fit(
            image,
            (600, 600),
            method=Image.Resampling.LANCZOS,
            centering=(0.5, 0.38),
        )

    foto_naomy = preparar_foto_equipo(
        foto_naomy_path
    )

    foto_dalay = preparar_foto_equipo(
        foto_dalay_path
    )

    foto_vladimir = preparar_foto_equipo(
        foto_vladimir_path
    )

    # ------------------------------------------------------------
    # EQUIPO
    # ------------------------------------------------------------

    col1, col2, col3 = st.columns(
        3,
        gap="large",
    )

    with col1:

        st.image(
            foto_naomy,
            use_container_width=True,
        )

        st.markdown(
            "<div style='text-align:center; "
            "color:#102F46; font-weight:800; "
            "font-size:1.05rem;'>"
            "Naomy Alvarado Zúñiga"
            "</div>",
            unsafe_allow_html=True,
        )

        st.markdown(
            "<div style='text-align:center; "
            "color:#647343; font-size:0.75rem; "
            "font-weight:700; margin-top:0.25rem;'>"
            "</div>",
            unsafe_allow_html=True,
        )

    with col2:

        st.image(
            foto_dalay,
            use_container_width=True,
        )

        st.markdown(
            "<div style='text-align:center; "
            "color:#102F46; font-weight:800; "
            "font-size:1.05rem;'>"
            "Dalay Sánchez Brenes"
            "</div>",
            unsafe_allow_html=True,
        )

        st.markdown(
            "<div style='text-align:center; "
            "color:#647343; font-size:0.75rem; "
            "font-weight:700; margin-top:0.25rem;'>"
            "</div>",
            unsafe_allow_html=True,
        )

    with col3:

        st.image(
            foto_vladimir,
            use_container_width=True,
        )

        st.markdown(
            "<div style='text-align:center; "
            "color:#102F46; font-weight:800; "
            "font-size:1.05rem;'>"
            "Vladímir Marín Durán"
            "</div>",
            unsafe_allow_html=True,
        )

        st.markdown(
            "<div style='text-align:center; "
            "color:#647343; font-size:0.75rem; "
            "font-weight:700; margin-top:0.25rem;'>"
            "</div>",
            unsafe_allow_html=True,
        )

    # ------------------------------------------------------------
    # CICLO DE TRABAJO
    # ------------------------------------------------------------

    st.html(
        """
        <style>

        .aci-work-title {
            text-align: center;
            color: #102F46;
            font-size: 1.15rem;
            font-weight: 800;
            letter-spacing: 0.06em;
            margin-top: 1.8rem;
            margin-bottom: 1rem;
        }

        .aci-work-cycle {
            width: 270px;
            height: 270px;
            margin: 0 auto;
            position: relative;
            border: 2px dashed rgba(100,115,67,0.65);
            border-radius: 50%;
            background: rgba(100,115,67,0.04);
        }

        .aci-cycle-center {
            position: absolute;
            width: 112px;
            height: 112px;
            top: 77px;
            left: 77px;
            border-radius: 50%;
            background: #102F46;
            color: #F7EED6;
            display: flex;
            align-items: center;
            justify-content: center;
            text-align: center;
            font-size: 0.78rem;
            font-weight: 800;
            line-height: 1.15;
        }

        .aci-cycle-item {
            position: absolute;
            color: #102F46;
            background: #F7EED6;
            border: 1px solid rgba(16,47,70,0.25);
            border-radius: 20px;
            padding: 0.36rem 0.7rem;
            font-size: 0.72rem;
            font-weight: 800;
            white-space: nowrap;
        }

        .aci-build {
            top: 5px;
            left: 96px;
        }

        .aci-review {
            top: 116px;
            right: -31px;
        }

        .aci-validate {
            bottom: 5px;
            left: 91px;
        }

        .aci-integrate {
            top: 116px;
            left: -31px;
        }

        .aci-arrow {
            position: absolute;
            color: #A84F2D;
            font-size: 1.35rem;
            font-weight: 800;
        }

        .aci-arrow-1 {
            top: 52px;
            right: 35px;
            transform: rotate(45deg);
        }

        .aci-arrow-2 {
            bottom: 47px;
            right: 38px;
            transform: rotate(135deg);
        }

        .aci-arrow-3 {
            bottom: 47px;
            left: 38px;
            transform: rotate(225deg);
        }

        .aci-arrow-4 {
            top: 52px;
            left: 35px;
            transform: rotate(315deg);
        }

        .aci-team-closing {
            max-width: 850px;
            margin: 1.5rem auto 0 auto;
            text-align: center;
            border-top: 1px solid rgba(16,47,70,0.22);
            padding-top: 1rem;
            color: #102F46;
            font-size: 1.05rem;
            font-weight: 700;
        }

        .aci-team-signature {
            text-align: center;
            color: #A84F2D;
            font-size: 0.75rem;
            font-weight: 800;
            letter-spacing: 0.14em;
            margin-top: 0.6rem;
        }

        </style>

        <div class="aci-work-title">
            NUESTRA FORMA DE TRABAJO
        </div>

        <div class="aci-work-cycle">

            <div class="aci-cycle-item aci-build">
                CONSTRUIR
            </div>

            <div class="aci-cycle-item aci-review">
                REVISAR
            </div>

            <div class="aci-cycle-item aci-validate">
                VALIDAR
            </div>

            <div class="aci-cycle-item aci-integrate">
                INTEGRAR
            </div>

            <div class="aci-arrow aci-arrow-1">→</div>
            <div class="aci-arrow aci-arrow-2">→</div>
            <div class="aci-arrow aci-arrow-3">→</div>
            <div class="aci-arrow aci-arrow-4">→</div>

            <div class="aci-cycle-center">
                TRABAJO<br>
                COMPARTIDO
            </div>

        </div>

        <div class="aci-team-closing">
            No solo construimos un modelo.
            Construimos un proceso trazable,
            reproducible y defendible.
        </div>

        <div class="aci-team-signature">
            ACI94 · DATA · EVIDENCE · IMPACT
        </div>
        """
    )

    # ============================================================
    # NAVEGACIÓN
    # ============================================================

    st.write("")

    if st.button(
        "← ANTERIOR",
        use_container_width=True,
        key="demo04_back",
    ):
        st.session_state.screen = "demo_03"
        st.rerun()