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
    st.title("¿Cómo llegamos hasta esta predicción?")
    