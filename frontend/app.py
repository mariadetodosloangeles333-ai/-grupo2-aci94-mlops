from pathlib import Path
from textwrap import dedent
import base64
import streamlit as st
import requests

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
                        Explore cómo un modelo de Machine Learning clasifica distintos perfiles
                        a partir de patrones socioeconómicos y laborales presentes en datos históricos.
                    </p>

                    <div class="principle">
                        <strong>Información para comprender, no para evaluar personas.</strong>
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

            <div class="demo-closing">
                La predicción que usted acaba de ver proviene de datos validados y de un modelo
                evaluado, registrado y versionado.
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

elif st.session_state.screen == "demo_03":
    st.title("¿Qué pasa después de la predicción?")

    col_new, col_team = st.columns([1, 1])

    with col_new:
        if st.button(
            "← NUEVA PREDICCIÓN",
             use_container_width=True,
             key="demo03_new"
        ):
             st.session_state.screen = "profile"
             st.rerun()

    with col_team:
        if st.button(
             "CONOZCA AL EQUIPO →",
             use_container_width=True,
             type="primary",
             key="demo03_team"
        ):
             st.session_state.screen = "demo_04"
             st.rerun()

elif st.session_state.screen == "demo_04":

    st.html(
        """
        <style>
        .team-page {
            max-width: 1100px;
            margin: 0 auto;
            padding: 0.4rem 1rem 1rem 1rem;
        }

        .team-title {
            text-align: center;
            color: #102F46;
            font-size: clamp(2rem, 4vw, 3.5rem);
            font-weight: 800;
            line-height: 1.05;
            margin-bottom: 0.4rem;
        }

        .team-subtitle {
            text-align: center;
            color: #40515B;
            font-size: 1rem;
            margin-bottom: 1.6rem;
        }

        .team-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 1.2rem;
            max-width: 850px;
            margin: 0 auto;
        }

        .team-card {
            text-align: center;
        }

        .photo-placeholder {
            width: 145px;
            height: 145px;
            margin: 0 auto 0.8rem auto;
            border-radius: 50%;
            border: 2px solid #A84F2D;
            background: rgba(250, 244, 226, 0.85);
            display: flex;
            align-items: center;
            justify-content: center;
            color: #647343;
            font-size: 2rem;
            font-weight: 800;
        }

        .team-name {
            color: #102F46;
            font-size: 1.05rem;
            font-weight: 800;
            line-height: 1.2;
        }

        .team-project {
            color: #647343;
            font-size: 0.78rem;
            font-weight: 700;
            margin-top: 0.25rem;
        }

        .work-title {
            text-align: center;
            color: #102F46;
            font-size: 1.1rem;
            font-weight: 800;
            letter-spacing: 0.06em;
            margin-top: 1.6rem;
            margin-bottom: 0.8rem;
        }

        .work-cycle {
            width: 245px;
            height: 245px;
            margin: 0 auto;
            position: relative;
            border: 2px dashed rgba(100, 115, 67, 0.65);
            border-radius: 50%;
            background: rgba(100, 115, 67, 0.04);
        }

        .cycle-center {
            position: absolute;
            width: 105px;
            height: 105px;
            top: 68px;
            left: 68px;
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

        .cycle-item {
            position: absolute;
            color: #102F46;
            background: #F7EED6;
            border: 1px solid rgba(16, 47, 70, 0.22);
            border-radius: 20px;
            padding: 0.32rem 0.6rem;
            font-size: 0.72rem;
            font-weight: 800;
            white-space: nowrap;
        }

        .cycle-build {
            top: 8px;
            left: 88px;
        }

        .cycle-review {
            top: 103px;
            right: -25px;
        }

        .cycle-validate {
            bottom: 8px;
            left: 82px;
        }

        .cycle-integrate {
            top: 103px;
            left: -25px;
        }

        .cycle-arrow {
            position: absolute;
            color: #A84F2D;
            font-size: 1.25rem;
            font-weight: 800;
        }

        .arrow-1 {
            top: 48px;
            right: 32px;
            transform: rotate(45deg);
        }

        .arrow-2 {
            bottom: 42px;
            right: 35px;
            transform: rotate(135deg);
        }

        .arrow-3 {
            bottom: 42px;
            left: 35px;
            transform: rotate(225deg);
        }

        .arrow-4 {
            top: 48px;
            left: 32px;
            transform: rotate(315deg);
        }

        .team-closing {
            max-width: 850px;
            margin: 1.4rem auto 0 auto;
            text-align: center;
            border-top: 1px solid rgba(16, 47, 70, 0.22);
            padding-top: 0.9rem;
            color: #102F46;
            font-size: 1.05rem;
            font-weight: 700;
        }

        .team-signature {
            text-align: center;
            color: #A84F2D;
            font-size: 0.75rem;
            font-weight: 800;
            letter-spacing: 0.14em;
            margin-top: 0.5rem;
        }

        @media (max-width: 700px) {
            .team-grid {
                grid-template-columns: 1fr;
                gap: 1.4rem;
            }

            .photo-placeholder {
                width: 120px;
                height: 120px;
            }

            .work-cycle {
                transform: scale(0.88);
                margin-top: -0.5rem;
                margin-bottom: -0.5rem;
            }

            .team-page {
                padding-left: 0.2rem;
                padding-right: 0.2rem;
            }
        }
        </style>

        <div class="team-page">

            <div class="team-title">
                ¿Quién hizo posible este proyecto?
            </div>

            <div class="team-subtitle">
                Tres personas, un proceso compartido y una misma responsabilidad sobre el resultado.
            </div>

            <div class="team-grid">

                <div class="team-card">
                    <div class="photo-placeholder">NA</div>
                    <div class="team-name">Naomy Alvarado Zúñiga</div>
                    <div class="team-project">EQUIPO ACI94</div>
                </div>

                <div class="team-card">
                    <div class="photo-placeholder">DS</div>
                    <div class="team-name">Dalay Sánchez Brenes</div>
                    <div class="team-project">EQUIPO ACI94</div>
                </div>

                <div class="team-card">
                    <div class="photo-placeholder">VM</div>
                    <div class="team-name">Vladímir Marín Durán</div>
                    <div class="team-project">EQUIPO ACI94</div>
                </div>

            </div>

            <div class="work-title">
                NUESTRA FORMA DE TRABAJO
            </div>

            <div class="work-cycle">

                <div class="cycle-item cycle-build">CONSTRUIR</div>
                <div class="cycle-item cycle-review">REVISAR</div>
                <div class="cycle-item cycle-validate">VALIDAR</div>
                <div class="cycle-item cycle-integrate">INTEGRAR</div>

                <div class="cycle-arrow arrow-1">→</div>
                <div class="cycle-arrow arrow-2">→</div>
                <div class="cycle-arrow arrow-3">→</div>
                <div class="cycle-arrow arrow-4">→</div>

                <div class="cycle-center">
                    TRABAJO<br>COMPARTIDO
                </div>

            </div>

            <div class="team-closing">
                No solo construimos un modelo. Construimos un proceso
                trazable, reproducible y defendible.
            </div>

            <div class="team-signature">
                ACI94 · DATA · EVIDENCE · IMPACT
            </div>

        </div>
        """
    )

    if st.button(
        "← NUEVA PREDICCIÓN",
        use_container_width=True,
        key="demo04_new"
    ):
        st.session_state.screen = "profile"
        st.rerun()