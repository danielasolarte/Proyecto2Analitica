import dash
from dash import dcc, html, Input, Output, State
import plotly.express as px
import pandas as pd
from dash import dash_table
import numpy as np

import joblib
import keras

import joblib
import numpy as np
from tensorflow.keras.models import load_model

# Cargar modelo pregunta 2
modelo_p2 = load_model("modelo_p2.keras")
scaler_p2 = joblib.load("scaler_p2.pkl")
encoder_p2 = joblib.load("encoder_p2.pkl")

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

# ======================================================
# CARGAR DATOS Y MODELOS
# ======================================================

df = pd.read_csv("df_modelo_preg1.csv")

df['ES_PRIVILEGIADO']   = ((df['COLE_BILINGUE_NUM'] == 1) & (df['COLE_AREA_UBICACION_NUM'] == 0)).astype(int)
df['JORNADA_EXTENDIDA'] = ((df['COLE_JORNADA_COMPLETA'] == 1) | (df['COLE_JORNADA_UNICA'] == 1)).astype(int)
df['COLEGIO_NO_MIXTO']  = ((df['COLE_GENERO_FEMENINO'] == 1) | (df['COLE_GENERO_MASCULINO'] == 1)).astype(int)
df['COLEGIO_TOP']       = ((df['ES_PRIVILEGIADO'] == 1) & (df['JORNADA_EXTENDIDA'] == 1)).astype(int)

model_base_p1 = keras.models.load_model("modelo1base.keras")
model_opt_p1  = keras.models.load_model("modelo2opt.keras")
scaler_p1     = joblib.load("scaler_preg1.pkl")

CARACTERISTICAS_P1 = [c for c in [
    'COLE_AREA_UBICACION_NUM',
    'COLE_BILINGUE_NUM',
    'COLE_GENERO_FEMENINO',
    'COLE_GENERO_MASCULINO',
    'COLE_GENERO_MIXTO',
    'COLE_JORNADA_COMPLETA',
    'COLE_JORNADA_MAÑANA',
    'COLE_JORNADA_NOCHE',
    'COLE_JORNADA_SABATINA',
    'COLE_JORNADA_TARDE',
    'COLE_JORNADA_UNICA',
    'ES_PRIVILEGIADO',
    'JORNADA_EXTENDIDA',
    'COLEGIO_NO_MIXTO',
    'COLEGIO_TOP',
] if c in df.columns]

stats_p1 = df["PUNT_MATEMATICAS"].describe().reset_index()
stats_p1.columns = ["Estadística", "Valor"]
stats_p1["Valor"] = stats_p1["Valor"].round(2)

# ======================================================
# LAYOUT
# ======================================================

app.layout = html.Div([

    html.H2(
        "Analítica de Resultados Saber 11 - Cundinamarca",
        style={'textAlign': 'center'}
    ),

    dcc.Tabs(value="tab-proyecto",style={"fontSize": "18px"}, children=[

        # ======================================================
        # PESTAÑA: NUESTRO PROYECTO 
        # ======================================================
        dcc.Tab(label="Nuestro Proyecto", value="tab-proyecto", children=[

            html.Br(),

            # =========================================
            # BLOQUE SUPERIOR: TEXTO + MAPA (CARD)
            # =========================================
            html.Div([

                # TEXTO (IZQUIERDA)
                html.Div([
                    html.H3("Introducción", style={"fontSize": "26px", "marginBottom": "10px"}),
                    html.P(
                        "Este proyecto presenta el desarrollo de un tablero de analítica interactivo orientado al análisis " \
                        "del desempeño académico en las pruebas Saber 11 en el departamento de Cundinamarca. A partir de datos " \
                        "oficiales del ICFES, se construyeron modelos predictivos basados en redes neuronales con el objetivo de " \
                        "identificar patrones y factores asociados al rendimiento de los estudiantes.",
                        style={"fontSize": "18px", "lineHeight": "1.6"}
                    ),

                    html.Br(),

                    html.H4("Integrantes del Equipo", style={"marginBottom": "10px"}),

                    html.Ul([
                        html.Li("Daniela Solarte"),
                        html.Li("Diego Galván"),
                        html.Li("Danna Isabella Gómez"),
                    ], style={"fontSize": "18px"})
                ], style={"width": "50%", "paddingRight": "20px"}),

                # MAPA (DERECHA)
                html.Div([
                    dcc.Graph(
                        id="mapa-cundinamarca",
                        figure=px.choropleth(
                            locations=["COL"],
                            locationmode="ISO-3",
                            color=[1],
                            color_continuous_scale=[[0, "#2E86AB"], [1, "#2E86AB"]],
                            scope="south america"
                        ).update_layout(
                            geo=dict(
                                showland=True,
                                landcolor="rgb(243,243,243)",
                                showframe=False,
                                showcoastlines=True,
                            ),
                            coloraxis_showscale=False,
                            margin={"r": 0, "t": 0, "l": 0, "b": 0},
                            height=350
                        )
                    )
                ], style={"width": "50%"})

            ], style={
                "display": "flex",
                "alignItems": "center",
                "backgroundColor": "white",
                "padding": "20px",
                "borderRadius": "10px",
                "boxShadow": "0px 2px 8px rgba(0,0,0,0.1)",
                "marginBottom": "30px"
            }),

            # =========================================
            # USUARIO FINAL + OBJETIVO
            # =========================================
            
            html.Div([

                # Usuario final
                html.Div([
                html.H4("Usuario final", style={"marginBottom": "10px"}),

                html.P([
                    html.Strong("DNP"),
                    ":Departamento Nacional de Planeación. Entidad encargada de orientar la planeación nacional "
                    "y la inversión pública, con interés en identificar brechas educativas y focalizar intervenciones."
                ], style={"fontSize": "18px", "lineHeight": "1.6"}),

                html.Br(),

                html.H4("Relevancia", style={"marginBottom": "10px"}),

                html.Ul([
                    html.Li("Qué características del colegio predicen mejor el desempeño"),
                    html.Li("Qué perfil socioeconómico se asocia a bajo rendimiento"),
                    html.Li("Dónde focalizar intervenciones según brechas"),
                ], style={"fontSize": "18px"})
            ],
            style={"width": "50%", "paddingRight": "10px"}),

                ], style={
                    "backgroundColor": "white",
                    "padding": "20px",
                    "borderRadius": "10px",
                    "boxShadow": "0px 2px 8px rgba(0,0,0,0.1)",
                    "marginBottom": "20px"
                }),   

            # =========================================
            # PREGUNTAS DE NEGOCIO
            # =========================================
            html.Div([
                html.H4("Preguntas de negocio", style={"marginBottom": "10px"}),
                html.Ul([
                    html.Li([
                        html.Strong("Pregunta 1 (Daniela): "),
                        "¿Cuál será el nivel de desempeño en matemáticas de un estudiante de Cundinamarca "
                        "según el tipo de colegio al que pertenece? ",
                        html.Em("(Clasificación binaria: Alto vs No-Alto)")
                    ]),
                    html.Li([
                        html.Strong("Pregunta 2 (Isabella): "),
                        "¿Podemos predecir el puntaje global ICFES esperado de cada estudiante de grado 11 "
                        "al iniciar el año escolar, en función de su perfil familiar, socioeconómico y del colegio? ",
                        html.Em("(Regresión)")
                    ]),
                    html.Li([
                        html.Strong("Pregunta 3 (Diego): "),
                        "¿Qué estudiantes tienen alta probabilidad de bajo desempeño a pesar de pertenecer "
                        "a contextos socioeconómicos favorables? ",
                        html.Em("(Clasificación binaria)")
                    ]),
                ], style={"fontSize": "18px"})
            ], style={
                "backgroundColor": "white",
                "padding": "20px",
                "borderRadius": "10px",
                "boxShadow": "0px 2px 8px rgba(0,0,0,0.1)",
                "marginBottom": "20px"
            }),

        ]),

        # ======================================================
        # PESTAÑA: PREGUNTA 1
        # ======================================================
        dcc.Tab(label="Pregunta 1", value="tab-proyecto-1", children=[

            html.Br(),
            html.H3("¿Cuál será el nivel de desempeño en matemáticas de un estudiante de Cundinamarca según el tipo de colegio al que pertenece?", style={
                "textAlign": "center",
                "marginBottom": "20px",
                "fontSize": "24px"
            }),

            html.P(
                "Esta pregunta busca predecir si un estudiante de Cundinamarca alcanzará un " \
                "desempeño alto en matemáticas en las pruebas Saber 11, a partir de características " \
                "relacionadas principalmente con el tipo de colegio al que pertenece y su contexto educativo. " \
                "En particular, el problema se formuló como una clasificación binaria, donde se clasifica a los estudiantes en dos categorías: " \
                "Alto (puntaje mayor o igual a 55) y No-Alto (puntaje inferior a 55)."
            ),

            #SECCIÓN 1: EXPLORACIÓN DE DATOS
            html.Hr(),
            html.H4("Exploración de datos",style={"fontSize": "24px", "marginBottom": "5px"}),

            html.P(
                "En esta sección puede explorar cómo varía el desempeño en matemáticas de los estudiantes según características del colegio. "
                "Utilice los filtros para seleccionar el tipo de jornada y el tipo de colegio, y observe cómo cambian la proporción de estudiantes "
                "con desempeño Alto y No-Alto, así como la distribución de sus puntajes. "
                "Esto le permitirá identificar patrones y comparar diferentes contextos educativos.",
                style={"fontSize": "16px", "lineHeight": "1.6"}
            ),

            html.Div([
                html.Div([
                    html.Label("Tipo de jornada:"),
                    dcc.Dropdown(
                        id="p1-jornada",
                        options=[
                            {"label": "Todas",    "value": "Todas"},
                            {"label": "Mañana",   "value": "COLE_JORNADA_MAÑANA"},
                            {"label": "Completa", "value": "COLE_JORNADA_COMPLETA"},
                            {"label": "Única",    "value": "COLE_JORNADA_UNICA"},
                            {"label": "Tarde",    "value": "COLE_JORNADA_TARDE"},
                            {"label": "Noche",    "value": "COLE_JORNADA_NOCHE"},
                            {"label": "Sabatina", "value": "COLE_JORNADA_SABATINA"},
                        ],
                        value="Todas",
                        clearable=False
                    )
                ], style={"width": "30%", "marginRight": "20px"}),

                html.Div([
                    html.Label("Tipo de colegio:"),
                    dcc.Dropdown(
                        id="p1-tipo",
                        options=[
                            {"label": "Todos",       "value": "Todos"},
                            {"label": "Bilingüe",    "value": "bilingue"},
                            {"label": "No bilingüe", "value": "no_bilingue"},
                            {"label": "Urbano",      "value": "urbano"},
                            {"label": "Rural",       "value": "rural"},
                        ],
                        value="Todos",
                        clearable=False
                    )
                ], style={"width": "30%"}),

            ], style={"display": "flex", "marginBottom": "20px"}),

            html.Div([
                html.Div([dcc.Graph(id="p1-bar-rendimiento")], style={"flex": "1"}),
                html.Div([dcc.Graph(id="p1-violin-mat")],      style={"flex": "1"}),
            ], style={"display": "flex", "gap": "20px"}),

            html.Br(),

            # SECCIÓN 2: PREDICCIÓN
            html.Hr(),

            html.H4(
                "Predicción del desempeño en matemáticas",
                style={"fontSize": "24px", "marginBottom": "5px"}
            ),

            html.P(
                "Seleccione las características del colegio y haga clic en predecir para estimar la probabilidad de que los estudiantes "
                "alcancen un desempeño alto en matemáticas.",
                style={"fontSize": "16px", "color": "#555", "marginBottom": "20px"}
            ),

            html.Div([

                # ── COLUMNA IZQUIERDA: INPUTS ──
                html.Div([

                    html.H5("Características del colegio", style={
                        "color": "#2E86AB", "borderBottom": "2px solid #2E86AB",
                        "paddingBottom": "8px", "marginBottom": "20px", "fontSize": "18px"
                    }),

                    # Área
                    html.Div([
                        html.Label("Área de ubicación", style={"fontWeight": "600", "fontSize": "14px", "marginBottom": "6px", "display": "block"}),
                        dcc.RadioItems(
                            id="p1-area",
                            options=[{"label": "  Urbano", "value": 0}, {"label": "  Rural", "value": 1}],
                            value=0,
                            inline=True,
                            inputStyle={"marginRight": "4px"},
                            labelStyle={"marginRight": "20px", "fontSize": "14px"}
                        ),
                    ], style={"marginBottom": "18px", "backgroundColor": "#f8f9fa", "padding": "12px", "borderRadius": "8px"}),

                    # Bilingüe
                    html.Div([
                        html.Label("¿Es bilingüe?", style={"fontWeight": "600", "fontSize": "14px", "marginBottom": "6px", "display": "block"}),
                        dcc.RadioItems(
                            id="p1-bilingue",
                            options=[{"label": "  Sí", "value": 1}, {"label": "  No", "value": 0}],
                            value=0,
                            inline=True,
                            inputStyle={"marginRight": "4px"},
                            labelStyle={"marginRight": "20px", "fontSize": "14px"}
                        ),
                    ], style={"marginBottom": "18px", "backgroundColor": "#f8f9fa", "padding": "12px", "borderRadius": "8px"}),

                    # Género
                    html.Div([
                        html.Label("Género del colegio", style={"fontWeight": "600", "fontSize": "14px", "marginBottom": "6px", "display": "block"}),
                        dcc.RadioItems(
                            id="p1-genero",
                            options=[
                                {"label": "  Mixto",     "value": "mixto"},
                                {"label": "  Femenino",  "value": "femenino"},
                                {"label": "  Masculino", "value": "masculino"},
                            ],
                            value="mixto",
                            inline=True,
                            inputStyle={"marginRight": "4px"},
                            labelStyle={"marginRight": "20px", "fontSize": "14px"}
                        ),
                    ], style={"marginBottom": "18px", "backgroundColor": "#f8f9fa", "padding": "12px", "borderRadius": "8px"}),

                    # Jornada
                    html.Div([
                        html.Label("Jornada escolar", style={"fontWeight": "600", "fontSize": "14px", "marginBottom": "8px", "display": "block"}),
                        dcc.Dropdown(
                            id="p1-jornada-pred",
                            options=[
                                {"label": "Mañana",   "value": "mañana"},
                                {"label": "Completa", "value": "completa"},
                                {"label": "Única",    "value": "unica"},
                                {"label": "Tarde",    "value": "tarde"},
                                {"label": "Noche",    "value": "noche"},
                                {"label": "Sabatina", "value": "sabatina"},
                            ],
                            value="mañana",
                            clearable=False,
                            style={"fontSize": "14px"}
                        ),
                    ], style={"marginBottom": "18px"}),

                    # Modelo
                    html.Div([
                        html.Label("Modelo a usar", style={"fontWeight": "600", "fontSize": "14px", "marginBottom": "6px", "display": "block"}),
                        dcc.RadioItems(
                            id="p1-modelo-selector",
                            options=[
                                {"label": "  MLP Base",       "value": "base"},
                                {"label": "  MLP Optimizada", "value": "opt"},
                            ],
                            value="opt",
                            inline=True,
                            inputStyle={"marginRight": "4px"},
                            labelStyle={"marginRight": "20px", "fontSize": "14px"}
                        ),
                    ], style={"marginBottom": "24px", "backgroundColor": "#f8f9fa", "padding": "12px", "borderRadius": "8px"}),

                    html.Button(
                        "PREDECIR",
                        id="p1-btn-predecir",
                        n_clicks=0,
                        style={
                            "backgroundColor": "#2E86AB",
                            "color": "white",
                            "border": "none",
                            "padding": "14px 30px",
                            "borderRadius": "10px",
                            "fontSize": "15px",
                            "fontWeight": "bold",
                            "cursor": "pointer",
                            "width": "100%",
                            "letterSpacing": "1px",
                            "boxShadow": "0 4px 12px rgba(46,134,171,0.35)",
                            "display": "block",
                            "lineHeight": "normal",
                            "height": "48px",
                            "textAlign": "center",
                            "fontFamily": "inherit"
                        }
                    ),

                ], style={
                    "width": "42%",
                    "backgroundColor": "white",
                    "padding": "28px",
                    "borderRadius": "14px",
                    "boxShadow": "0 4px 18px rgba(0,0,0,0.08)",
                    "boxSizing": "border-box"
                }),

                # ── COLUMNA DERECHA: RESULTADO ──
                html.Div([

                    html.H5("Resultado de la predicción", style={
                        "color": "#2E86AB", "borderBottom": "2px solid #2E86AB",
                        "paddingBottom": "8px", "marginBottom": "20px", "fontSize": "18px"
                    }),

                    html.Div(id="p1-resultado", style={
                        "fontSize": "18px",
                        "fontWeight": "bold",
                        "textAlign": "center",
                        "padding": "18px",
                        "borderRadius": "10px",
                        "backgroundColor": "#fef3f2",
                        "color": "#7a1c1c",
                        "boxShadow": "0 2px 8px rgba(0,0,0,0.06)",
                        "marginBottom": "16px",
                        "minHeight": "60px",
                        "display": "flex",
                        "alignItems": "center",
                        "justifyContent": "center"
                    }),

                    dcc.Graph(
                        id="p1-gauge",
                        config={"displayModeBar": False},
                        style={"marginBottom": "16px"}
                    ),

                    html.Div(id="p1-interpretacion", style={
                        "backgroundColor": "#f0f7fb",
                        "padding": "18px",
                        "borderRadius": "10px",
                        "border": "1px solid #c8e6f5",
                        "fontSize": "14px",
                        "lineHeight": "1.8",
                        "color": "#333"
                    }),

                ], style={
                    "width": "54%",
                    "backgroundColor": "white",
                    "padding": "28px",
                    "borderRadius": "14px",
                    "boxShadow": "0 4px 18px rgba(0,0,0,0.08)",
                    "boxSizing": "border-box"
                }),

            ], style={
                "display": "flex",
                "gap": "24px",
                "alignItems": "flex-start",
                "marginTop": "10px",
                "marginBottom": "30px"
            }),

            # SECCIÓN 3: COMPARACIÓN DE MODELOS
            html.Hr(),

            html.H4("¿Qué tan bien funciona el modelo?", style={"fontSize": "22px", "marginBottom": "6px"}),

            html.P(
                "Se entrenaron y compararon dos versiones del modelo predictivo. "
                "A continuación se presenta un resumen de qué tan bien cada modelo clasifica a los estudiantes "
                "en Alto y No-Alto desempeño en matemáticas.",
                style={"fontSize": "15px", "color": "#555", "marginBottom": "20px"}
            ),

            # ── TARJETAS DE RESUMEN ──
            html.Div([

                html.Div([
                    html.Div("MLP Base", style={"fontSize": "13px", "color": "#888", "marginBottom": "4px"}),
                    html.Div("Modelo seleccionado ✓", style={"fontSize": "16px", "fontWeight": "bold", "color": "#2E86AB", "marginBottom": "12px"}),
                    html.Div([
                        html.Span("Precisión global", style={"fontSize": "13px", "color": "#555"}),
                        html.Span("66.1%", style={"fontSize": "20px", "fontWeight": "bold", "color": "#2E86AB", "float": "right"}),
                    ], style={"marginBottom": "8px", "overflow": "hidden"}),
                    html.Div([
                        html.Span("Equilibrio entre clases (F1)", style={"fontSize": "13px", "color": "#555"}),
                        html.Span("50.3%", style={"fontSize": "20px", "fontWeight": "bold", "color": "#2E86AB", "float": "right"}),
                    ], style={"marginBottom": "8px", "overflow": "hidden"}),
                    
                    html.Div([
                        html.Span("Detección de Alto desempeño (F1 Alto)", style={"fontSize": "13px", "color": "#555"}),
                        html.Span("22.2%", style={"fontSize": "20px", "fontWeight": "bold", "color": "#2E86AB", "float": "right"}),  # MLP Base
                        # usar "22.1%" para MLP Optimizada
                    ], style={"marginBottom": "8px", "overflow": "hidden"}),
                    
                    html.Div([
                        html.Span("Capacidad discriminativa (AUC)", style={"fontSize": "13px", "color": "#555"}),
                        html.Span("63.9%", style={"fontSize": "20px", "fontWeight": "bold", "color": "#2E86AB", "float": "right"}),
                    ], style={"overflow": "hidden"}),
                ], style={
                    "flex": "1", "backgroundColor": "white", "padding": "22px",
                    "borderRadius": "12px", "boxShadow": "0 2px 10px rgba(0,0,0,0.07)",
                    "borderTop": "4px solid #2E86AB"
                }),

                html.Div([
                    html.Div("MLP Optimizada", style={"fontSize": "13px", "color": "#888", "marginBottom": "4px"}),
                    html.Div("Modelo alternativo", style={"fontSize": "16px", "fontWeight": "bold", "color": "#4ABFBF", "marginBottom": "12px"}),
                    html.Div([
                        html.Span("Precisión global", style={"fontSize": "13px", "color": "#555"}),
                        html.Span("66.1%", style={"fontSize": "20px", "fontWeight": "bold", "color": "#4ABFBF", "float": "right"}),
                    ], style={"marginBottom": "8px", "overflow": "hidden"}),
                    html.Div([
                        html.Span("Equilibrio entre clases (F1)", style={"fontSize": "13px", "color": "#555"}),
                        html.Span("50.2%", style={"fontSize": "20px", "fontWeight": "bold", "color": "#4ABFBF", "float": "right"}),
                    ], style={"marginBottom": "8px", "overflow": "hidden"}),
                    
                    html.Div([
                        html.Span("Detección de Alto desempeño (F1 Alto)", style={"fontSize": "13px", "color": "#555"}),
                        html.Span("22.1%", style={"fontSize": "20px", "fontWeight": "bold", "color": "#4ABFBF", "float": "right"}),  # MLP Base
                        # usar "22.1%" para MLP Optimizada
                    ], style={"marginBottom": "8px", "overflow": "hidden"}),
                    
                    html.Div([
                        html.Span("Capacidad discriminativa (AUC)", style={"fontSize": "13px", "color": "#555"}),
                        html.Span("63.9%", style={"fontSize": "20px", "fontWeight": "bold", "color": "#4ABFBF", "float": "right"}),
                    ], style={"overflow": "hidden"}),
                ], style={
                    "flex": "1", "backgroundColor": "white", "padding": "22px",
                    "borderRadius": "12px", "boxShadow": "0 2px 10px rgba(0,0,0,0.07)",
                    "borderTop": "4px solid #4ABFBF"
                }),

                # TARJETA INTERPRETACIÓN
                html.Div([
                    html.Div("¿Qué significa cada métrica?", style={"fontSize": "15px", "fontWeight": "bold", "color": "#2E86AB", "marginBottom": "12px"}),
                    html.Div([
                        html.Span("Precisión global: ", style={"fontWeight": "600", "fontSize": "13px"}),
                        html.Span("% de estudiantes clasificados correctamente.", style={"fontSize": "13px", "color": "#555"}),
                    ], style={"marginBottom": "10px"}),
                    html.Div([
                        html.Span("Equilibrio entre clases: ", style={"fontWeight": "600", "fontSize": "13px"}),
                        html.Span("qué tan bien detecta tanto estudiantes de Alto como de No-Alto desempeño. Cercano a 1 es ideal.", style={"fontSize": "13px", "color": "#555"}),
                    ], style={"marginBottom": "10px"}),
                    html.Div([
                        html.Span("Capacidad discriminativa: ", style={"fontWeight": "600", "fontSize": "13px"}),
                        html.Span("qué tan bien separa los dos grupos. 0.5 equivale a adivinar al azar; 1.0 es perfecto.", style={"fontSize": "13px", "color": "#555"}),
                    ]),
                ], style={
                    "flex": "1.2", "backgroundColor": "#f0f7fb", "padding": "22px",
                    "borderRadius": "12px", "boxShadow": "0 2px 10px rgba(0,0,0,0.07)",
                    "border": "1px solid #c8e6f5"
                }),

            ], style={"display": "flex", "gap": "20px", "marginBottom": "28px"}),

            # ── GRÁFICA ──
            html.Div([
                dcc.Graph(id="p1-comparacion-barras", config={"displayModeBar": False})
            ], style={
                "backgroundColor": "white", "borderRadius": "12px",
                "boxShadow": "0 2px 10px rgba(0,0,0,0.07)", "padding": "10px",
                "marginBottom": "24px"
            }),

            # ── CONCLUSIÓN INTERPRETATIVA ──
            html.Div([
                html.H5("¿Por qué se eligió el MLP Base?", style={"color": "#2E86AB", "marginBottom": "12px", "fontSize": "16px"}),

                html.P(
                    "Ambos modelos tienen un desempeño muy similar, pero se seleccionó el MLP Base porque logra "
                    "una mejor detección de estudiantes con Alto desempeño (F1 Alto = 22.2% vs 22.1%) y alcanza "
                    "convergencia estable sin señales de sobreajuste, lo que lo hace más confiable y fácil de interpretar.",
                    style={"fontSize": "14px", "color": "#444", "lineHeight": "1.7", "marginBottom": "14px"}
                ),

                html.Div([
                    html.Div([
                        html.Span(" ", style={"fontSize": "16px"}),
                        html.Strong("Limitación importante: "),
                        html.Span(
                            "El modelo identifica correctamente la gran mayoría de estudiantes con No-Alto desempeño "
                            "(~34.400 casos), pero le cuesta detectar a los estudiantes con Alto desempeño (~2.500 de 17.000 casos). "
                            "Esto ocurre porque las variables del colegio capturan tendencias grupales, no trayectorias individuales. "
                            "El modelo es útil para focalizar colegios con mayor riesgo, pero debe complementarse con información "
                            "del estudiante para decisiones individuales.",
                            style={"fontSize": "14px", "color": "#555"}
                        ),
                    ])
                ], style={
                    "backgroundColor": "#fff8e1",
                    "border": "1px solid #ffe082",
                    "borderRadius": "10px",
                    "padding": "16px",
                    "lineHeight": "1.7"
                }),

            ], style={
                "backgroundColor": "white", "padding": "24px",
                "borderRadius": "12px", "boxShadow": "0 2px 10px rgba(0,0,0,0.07)",
                "marginBottom": "20px"
            }),

        ]),
        #####Fin Pregunta 1#####
        dcc.Tab(label="Predicción P2 - MLP", value="tab-pred2", children=[

    html.Br(),
    html.H3("Predicción de Puntaje Global ICFES"),
    html.P("Ingresa el perfil del estudiante para estimar su puntaje global ICFES."),

    html.Div([

        html.Div([
            html.Label("Género del estudiante:"),
            dcc.Dropdown(id="p2-genero",
                options=[{"label": "Femenino", "value": "F"},
                         {"label": "Masculino", "value": "M"}],
                value="F", clearable=False),
            html.Br(),
            html.Label("Área del colegio:"),
            dcc.Dropdown(id="p2-area",
                options=[{"label": "Urbano", "value": "Urbano"},
                         {"label": "Rural", "value": "Rural"}],
                value="Urbano", clearable=False),
            html.Br(),
            html.Label("¿Colegio bilingüe?"),
            dcc.Dropdown(id="p2-bilingue",
                options=[{"label": "Sí", "value": "S"},
                         {"label": "No", "value": "N"}],
                value="N", clearable=False),
            html.Br(),
            html.Label("Género del colegio:"),
            dcc.Dropdown(id="p2-cole-genero",
                options=[{"label": "Mixto", "value": "MIXTO"},
                         {"label": "Femenino", "value": "FEMENINO"},
                         {"label": "Masculino", "value": "MASCULINO"}],
                value="MIXTO", clearable=False),
            html.Br(),
            html.Label("Estrato socioeconómico:"),
            dcc.Slider(id="p2-estrato", min=1, max=6, step=1, value=2,
                marks={i: str(i) for i in range(1, 7)}),
            html.Br(),
            html.Label("¿Tiene internet en casa?"),
            dcc.Dropdown(id="p2-internet",
                options=[{"label": "Sí", "value": "Si"},
                         {"label": "No", "value": "No"}],
                value="Si", clearable=False),
            html.Br(),
            html.Label("¿Tiene computador en casa?"),
            dcc.Dropdown(id="p2-computador",
                options=[{"label": "Sí", "value": "Si"},
                         {"label": "No", "value": "No"}],
                value="Si", clearable=False),
            html.Br(),
            html.Label("¿Tiene automóvil?"),
            dcc.Dropdown(id="p2-automovil",
                options=[{"label": "Sí", "value": "Si"},
                         {"label": "No", "value": "No"}],
                value="No", clearable=False),
            html.Br(),
            html.Label("Número de personas en el hogar:"),
            dcc.Slider(id="p2-personas", min=1, max=10, step=1, value=4,
                marks={i: str(i) for i in range(1, 11)}),
            html.Br(),
            html.Button("Predecir Puntaje", id="p2-btn", n_clicks=0,
                style={"backgroundColor": "#2E8B57", "color": "white",
                       "padding": "10px 20px", "border": "none",
                       "borderRadius": "5px", "cursor": "pointer",
                       "fontSize": "16px"})
        ], style={"width": "45%", "display": "inline-block",
                  "verticalAlign": "top", "paddingRight": "40px"}),

        html.Div([
            html.H4("Resultado:"),
            html.Div(id="p2-resultado",
                style={"fontSize": "48px", "fontWeight": "bold",
                       "color": "#2E8B57", "textAlign": "center",
                       "padding": "40px", "backgroundColor": "white",
                       "borderRadius": "10px",
                       "boxShadow": "0 4px 12px rgba(0,0,0,0.1)"}),
            html.Br(),
            html.Div(id="p2-interpretacion",
                style={"textAlign": "center", "fontSize": "16px",
                       "color": "#555"})
        ], style={"width": "45%", "display": "inline-block",
                  "verticalAlign": "top"})

    ], style={"marginTop": "20px"})
]),


        ###############Aquí irán Pregunta 2 y Pregunta 3..#############

    ])
])

# ======================================================
# CALLBACKS PREGUNTA 1
# ======================================================

@app.callback(
    Output("p1-bar-rendimiento", "figure"),
    Output("p1-violin-mat",      "figure"),
    Input("p1-jornada",          "value"),
    Input("p1-tipo",             "value"),
)
def actualizar_exploracion_p1(jornada_sel, tipo_sel):

    dff = df.copy()

    if jornada_sel != "Todas":
        dff = dff[dff[jornada_sel] == 1]

    if tipo_sel == "bilingue":
        dff = dff[dff["COLE_BILINGUE_NUM"] == 1]
    elif tipo_sel == "no_bilingue":
        dff = dff[dff["COLE_BILINGUE_NUM"] == 0]
    elif tipo_sel == "urbano":
        dff = dff[dff["COLE_AREA_UBICACION_NUM"] == 0]
    elif tipo_sel == "rural":
        dff = dff[dff["COLE_AREA_UBICACION_NUM"] == 1]

    if dff.empty:
        fig_vacia = px.bar(title="Sin datos para los filtros seleccionados")
        return fig_vacia, fig_vacia

    conteo = dff["ALTO_RENDIMIENTO"].value_counts().reset_index()
    conteo.columns = ["Clase", "Cantidad"]
    conteo["Clase"] = conteo["Clase"].map({1: "Alto", 0: "No-Alto"})
    conteo["Porcentaje"] = (conteo["Cantidad"] / conteo["Cantidad"].sum() * 100).round(1)

    fig_bar = px.bar(
        conteo,
        x="Clase",
        y="Porcentaje",
        color="Clase",
        text="Porcentaje",
        color_discrete_map={"Alto": "#2E86AB", "No-Alto": "#F5A623"},
        title="Proporción de estudiantes Alto vs No-Alto en matemáticas"
    )
    fig_bar.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig_bar.update_layout(
        template="simple_white",
        yaxis_title="Porcentaje (%)",
        xaxis_title="Nivel de desempeño",
        showlegend=False,
        yaxis_range=[0, 100]
    )

    dff_violin = dff.copy()
    dff_violin["Nivel"] = dff_violin["ALTO_RENDIMIENTO"].map({1: "Alto", 0: "No-Alto"})

    fig_violin = px.violin(
        dff_violin,
        x="Nivel",
        y="PUNT_MATEMATICAS",
        color="Nivel",
        box=True,
        points=False,
        color_discrete_map={"Alto": "#2E86AB", "No-Alto": "#F5A623"},
        title="Distribución del puntaje en matemáticas por nivel de desempeño"
    )
    fig_violin.update_layout(
        template="simple_white",
        xaxis_title="Nivel de desempeño",
        yaxis_title="Puntaje Matemáticas",
        showlegend=False
    )

    return fig_bar, fig_violin


@app.callback(
    Output("p1-resultado",      "children"),
    Output("p1-resultado",      "style"),
    Output("p1-gauge",          "figure"),
    Output("p1-interpretacion", "children"),
    Input("p1-btn-predecir",    "n_clicks"),
    State("p1-area",            "value"),
    State("p1-bilingue",        "value"),
    State("p1-genero",          "value"),
    State("p1-jornada-pred",    "value"),
    State("p1-modelo-selector", "value"),
    prevent_initial_call=True
)
def predecir_p1(n_clicks, area, bilingue, genero, jornada, modelo_sel):

    genero_femenino  = 1 if genero == "femenino"  else 0
    genero_masculino = 1 if genero == "masculino" else 0
    genero_mixto     = 1 if genero == "mixto"     else 0

    jornada_completa = 1 if jornada == "completa" else 0
    jornada_manana   = 1 if jornada == "mañana"   else 0
    jornada_noche    = 1 if jornada == "noche"    else 0
    jornada_sabatina = 1 if jornada == "sabatina" else 0
    jornada_tarde    = 1 if jornada == "tarde"    else 0
    jornada_unica    = 1 if jornada == "unica"    else 0

    es_privilegiado   = 1 if (bilingue == 1 and area == 0) else 0
    jornada_extendida = 1 if (jornada_completa == 1 or jornada_unica == 1) else 0
    colegio_no_mixto  = 1 if (genero_femenino == 1 or genero_masculino == 1) else 0
    colegio_top       = 1 if (es_privilegiado == 1 and jornada_extendida == 1) else 0

    fila = np.array([[
        area, bilingue,
        genero_femenino, genero_masculino, genero_mixto,
        jornada_completa, jornada_manana, jornada_noche,
        jornada_sabatina, jornada_tarde, jornada_unica,
        es_privilegiado, jornada_extendida, colegio_no_mixto, colegio_top,
    ]], dtype=np.float32)

    fila        = fila[:, :len(CARACTERISTICAS_P1)]
    fila_scaled = scaler_p1.transform(fila)
    modelo      = model_opt_p1 if modelo_sel == "opt" else model_base_p1
    prob        = float(modelo.predict(fila_scaled, verbose=0).ravel()[0])
    clase       = "Alto" if prob >= 0.5 else "No-Alto"

    color_fondo = "#d4edda" if clase == "Alto" else "#fdecea"
    color_texto = "#155724" if clase == "Alto" else "#721c24"

    estilo_resultado = {
        "fontSize": "18px",
        "fontWeight": "bold",
        "textAlign": "center",
        "padding": "18px",
        "borderRadius": "10px",
        "backgroundColor": color_fondo,
        "color": color_texto,
        "boxShadow": "0 2px 8px rgba(0,0,0,0.06)",
        "marginBottom": "16px",
        "minHeight": "60px",
        "display": "flex",
        "alignItems": "center",
        "justifyContent": "center"
    }

    texto_resultado = f"Predicción: {clase}  |  Probabilidad Alto: {prob*100:.1f}%"

    fig_gauge = {
        "data": [{
            "type": "indicator",
            "mode": "gauge+number",
            "value": round(prob * 100, 1),
            "number": {"suffix": "%", "font": {"size": 28}},
            "title": {"text": "Probabilidad de Alto desempeño", "font": {"size": 14}},
            "gauge": {
                "axis": {"range": [0, 100]},
                "bar":  {"color": "#2E86AB"},
                "steps": [
                    {"range": [0,  50], "color": "#fdecea"},
                    {"range": [50, 100], "color": "#d4edda"},
                ],
                "threshold": {
                    "line":  {"color": "black", "width": 3},
                    "thickness": 0.75,
                    "value": 50
                }
            }
        }],
        "layout": {"height": 280, "margin": {"t": 40, "b": 10, "l": 20, "r": 20}}
    }

    nombre_modelo = "MLP Optimizada" if modelo_sel == "opt" else "MLP Base"
    tipo_colegio  = "bilingüe" if bilingue == 1 else "no bilingüe"
    area_label    = "urbano"   if area == 0    else "rural"
    jornada_label = {
        "mañana": "mañana", "completa": "completa", "unica": "única",
        "tarde": "tarde",   "noche": "noche",       "sabatina": "sabatina"
    }.get(jornada, jornada)

    interpretacion = html.Div([
        html.P([html.Strong("Modelo usado: "), nombre_modelo]),
        html.P([
            html.Strong("Perfil del colegio: "),
            f"{area_label.capitalize()}, {tipo_colegio}, jornada {jornada_label}, género {genero}."
        ]),
        html.P(
            f"El modelo estima una probabilidad de {prob*100:.1f}% de que los estudiantes "
            f"de este colegio alcancen un desempeño Alto en matemáticas."
        ),
        html.P(
            "Los colegios bilingües en jornada completa o única presentan consistentemente "
            "mayor probabilidad de Alto desempeño según el análisis exploratorio."
            if clase == "Alto" else
            "Los colegios con jornada nocturna o sabatina y sin bilingüismo tienden a concentrar "
            "los puntajes más bajos. Se recomienda focalizar intervenciones en estas instituciones."
        ),
    ])

    return texto_resultado, estilo_resultado, fig_gauge, interpretacion


@app.callback(
    Output("p1-comparacion-barras", "figure"),
    Input("p1-comparacion-barras",  "id")
)
def grafica_comparacion_p1(_):

    datos = pd.DataFrame([
        {"Modelo": "MLP Base",       "Precisión\nglobal": 0.6612, "F1\nNo-Alto": 0.7834, "F1\nAlto": 0.2222, "Precision\nMacro": 0.6423, "Recall\nMacro": 0.5446, "Capacidad\ndiscriminativa": 0.6389},
        {"Modelo": "MLP Optimizada", "Precisión\nglobal": 0.6614, "F1\nNo-Alto": 0.7836, "F1\nAlto": 0.2213, "Precision\nMacro": 0.6434, "Recall\nMacro": 0.5445, "Capacidad\ndiscriminativa": 0.6389},
    ])

    fig = px.bar(
        datos.melt(id_vars="Modelo", var_name="Métrica", value_name="Valor"),
        x="Métrica",
        y="Valor",
        color="Modelo",
        barmode="group",
        color_discrete_map={"MLP Base": "#2E86AB", "MLP Optimizada": "#4ABFBF"},
        title="Comparación visual de métricas por modelo",
        text_auto=".4f"
    )
    fig.update_traces(textposition="outside", textfont_size=11)
    fig.update_layout(
        template="simple_white",
        yaxis=dict(range=[0, 1.0], tickformat=".2f", title="Valor"),
        xaxis_title="",
        legend_title="Modelo",
        font=dict(size=12),
        title_font_size=15,
        margin={"t": 60, "b": 20},
        bargap=0.25,
        bargroupgap=0.05
    )


    return fig

@app.callback(
    Output("p2-resultado", "children"),
    Output("p2-interpretacion", "children"),
    Input("p2-btn", "n_clicks"),
    State("p2-genero", "value"),
    State("p2-area", "value"),
    State("p2-bilingue", "value"),
    State("p2-cole-genero", "value"),
    State("p2-estrato", "value"),
    State("p2-internet", "value"),
    State("p2-computador", "value"),
    State("p2-automovil", "value"),
    State("p2-personas", "value"),
    prevent_initial_call=True
)
def predecir_puntaje(n_clicks, genero, area, bilingue, cole_genero,
                     estrato, internet, computador, automovil, personas):

    from tensorflow.keras.models import load_model
    modelo_p2 = load_model("modelo_p2.keras")
    scaler_p2 = joblib.load("scaler_p2.pkl")
    encoder_p2 = joblib.load("encoder_p2.pkl")

    # Orden exacto del encoder
    cols_encoder = ['COLE_AREA_UBICACION', 'COLE_BILINGUE', 'COLE_GENERO',
                    'COLE_MCPIO_UBICACION', 'ESTU_GENERO', 'FAMI_CUARTOSHOGAR',
                    'FAMI_EDUCACIONMADRE', 'FAMI_EDUCACIONPADRE', 'FAMI_ESTRATOVIVIENDA',
                    'FAMI_PERSONASHOGAR', 'FAMI_TIENEAUTOMOVIL', 'FAMI_TIENECOMPUTADOR',
                    'FAMI_TIENEINTERNET', 'FAMI_TIENELAVADORA']

    # Orden exacto del scaler
    cols_scaler = ['PERIODO', 'COLE_AREA_UBICACION', 'COLE_BILINGUE', 'COLE_GENERO',
                   'COLE_MCPIO_UBICACION', 'ESTU_GENERO', 'FAMI_CUARTOSHOGAR',
                   'FAMI_EDUCACIONMADRE', 'FAMI_EDUCACIONPADRE', 'FAMI_ESTRATOVIVIENDA',
                   'FAMI_PERSONASHOGAR', 'FAMI_TIENEAUTOMOVIL', 'FAMI_TIENECOMPUTADOR',
                   'FAMI_TIENEINTERNET', 'FAMI_TIENELAVADORA']

    # Crear dataframe con orden del encoder
    input_encoder = pd.DataFrame([[
        area, bilingue, cole_genero, "Zipaquirá", genero,
        "3 a 5", "Secundaria (Bachillerato)", "Secundaria (Bachillerato)",
        "Estrato " + str(estrato), "5 a 6",
        automovil, computador, internet, "Si"
    ]], columns=cols_encoder)

    # Codificar
    input_encoded = encoder_p2.transform(input_encoder)
    input_encoded_df = pd.DataFrame(input_encoded, columns=cols_encoder)

    # Agregar PERIODO y reordenar para el scaler
    input_encoded_df['PERIODO'] = 2019
    input_scaler = input_encoded_df[cols_scaler]

    # Escalar y predecir
    input_scaled = scaler_p2.transform(input_scaler)
    puntaje = modelo_p2.predict(input_scaled)[0][0]
    puntaje = round(float(puntaje), 1)

    if puntaje < 200:
        interpretacion = "⚠️ Riesgo alto — se recomienda intervención temprana"
    elif puntaje < 280:
        interpretacion = "📊 Desempeño medio esperado — monitoreo recomendado"
    else:
        interpretacion = "✅ Buen desempeño esperado"

    return f"{puntaje} pts", interpretacion



if __name__ == '__main__':
    app.run(debug=True)