import dash
from dash import dcc, html, Input, Output, State
import plotly.express as px
import pandas as pd
from dash import dash_table
import numpy as np
import joblib
import keras

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
                    "display": "flex",
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
        dcc.Tab(label="Pregunta 1", value="tab-1", children=[

            html.Br(),
            html.H3("Pregunta 1 — Daniela Solarte"),

            html.Strong(
                "¿Cuál será el nivel de desempeño en matemáticas de un estudiante de Cundinamarca "
                "según el tipo de colegio al que pertenece?"
            ),

            html.Br(), html.Br(),

            html.P(
                "Se desarrolló un modelo de clasificación binaria basado en una red neuronal multicapa (MLP) "
                "que predice si un estudiante obtendrá un desempeño Alto o No-Alto en matemáticas, "
                "a partir de las características institucionales de su colegio."
            ),

            # SECCIÓN 1: EXPLORACIÓN
            html.Hr(),
            html.H4("Exploración de datos"),

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
            html.H4("Predicción con red neuronal"),

            html.P(
                "Seleccione las características del colegio para predecir si sus estudiantes "
                "tendrán un desempeño Alto o No-Alto en matemáticas."
            ),

            html.Div([

                html.Div([

                    html.Label("Área de ubicación:"),
                    dcc.RadioItems(
                        id="p1-area",
                        options=[
                            {"label": "Urbano", "value": 0},
                            {"label": "Rural",  "value": 1},
                        ],
                        value=0,
                        inline=True
                    ),

                    html.Br(),

                    html.Label("¿Es bilingüe?"),
                    dcc.RadioItems(
                        id="p1-bilingue",
                        options=[
                            {"label": "Sí", "value": 1},
                            {"label": "No", "value": 0},
                        ],
                        value=0,
                        inline=True
                    ),

                    html.Br(),

                    html.Label("Género del colegio:"),
                    dcc.RadioItems(
                        id="p1-genero",
                        options=[
                            {"label": "Mixto",     "value": "mixto"},
                            {"label": "Femenino",  "value": "femenino"},
                            {"label": "Masculino", "value": "masculino"},
                        ],
                        value="mixto",
                        inline=True
                    ),

                    html.Br(),

                    html.Label("Jornada escolar:"),
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
                        clearable=False
                    ),

                    html.Br(),

                    html.Label("Modelo a usar:"),
                    dcc.RadioItems(
                        id="p1-modelo-selector",
                        options=[
                            {"label": "MLP Base",       "value": "base"},
                            {"label": "MLP Optimizada", "value": "opt"},
                        ],
                        value="opt",
                        inline=True
                    ),

                    html.Br(),

                    html.Button(
                        "Predecir",
                        id="p1-btn-predecir",
                        n_clicks=0,
                        style={
                            "backgroundColor": "#2E86AB",
                            "color": "white",
                            "border": "none",
                            "padding": "10px 25px",
                            "borderRadius": "8px",
                            "fontSize": "18px",
                            "cursor": "pointer"
                        }
                    ),

                ], style={
                    "flex": "1",
                    "backgroundColor": "#f9f9f9",
                    "padding": "25px",
                    "borderRadius": "12px",
                    "boxShadow": "0 4px 12px rgba(0,0,0,0.08)"
                }),

                html.Div([

                    html.H4("Resultado de la predicción", style={"marginBottom": "15px"}),

                    html.Div(id="p1-resultado", style={
                        "fontSize": "22px",
                        "fontWeight": "bold",
                        "textAlign": "center",
                        "padding": "30px",
                        "borderRadius": "12px",
                        "backgroundColor": "white",
                        "boxShadow": "0 4px 12px rgba(0,0,0,0.08)",
                        "marginBottom": "20px",
                        "minHeight": "80px"
                    }),

                    dcc.Graph(id="p1-gauge"),

                    html.Div(id="p1-interpretacion", style={
                        "backgroundColor": "white",
                        "padding": "20px",
                        "borderRadius": "10px",
                        "boxShadow": "0 4px 12px rgba(0,0,0,0.05)",
                        "marginTop": "20px",
                        "lineHeight": "1.7"
                    }),

                ], style={"flex": "1", "paddingLeft": "30px"}),

            ], style={"display": "flex", "gap": "30px", "alignItems": "flex-start"}),

            html.Br(),

            # SECCIÓN 3: COMPARACIÓN DE MODELOS
            html.Hr(),
            html.H4("Comparación de modelos"),

            html.P("Se entrenaron dos arquitecturas MLP y se compararon usando F1-Score y AUC-ROC."),

            html.Div([
                html.Div([
                    dash_table.DataTable(
                        id="p1-tabla-modelos",
                        columns=[
                            {"name": "Modelo",      "id": "Modelo"},
                            {"name": "Accuracy",    "id": "Accuracy"},
                            {"name": "F1 Macro",    "id": "F1 Macro"},
                            {"name": "AUC-ROC",     "id": "AUC-ROC"},
                            {"name": "Arquitectura","id": "Arquitectura"},
                        ],
                        data=[
                            {
                                "Modelo":       "MLP Base",
                                "Accuracy":     "—",
                                "F1 Macro":     "—",
                                "AUC-ROC":      "—",
                                "Arquitectura": f"{len(CARACTERISTICAS_P1)}→64→32→1"
                            },
                            {
                                "Modelo":       "MLP Optimizada",
                                "Accuracy":     "—",
                                "F1 Macro":     "—",
                                "AUC-ROC":      "—",
                                "Arquitectura": f"{len(CARACTERISTICAS_P1)}→64→32→1 + BN + Dropout"
                            },
                        ],
                        style_cell={"textAlign": "center", "padding": "10px"},
                        style_header={
                            "backgroundColor": "#2E86AB",
                            "color": "white",
                            "fontWeight": "bold"
                        },
                        style_data_conditional=[{
                            "if": {"row_index": "odd"},
                            "backgroundColor": "#f2f9fc"
                        }]
                    )
                ], style={"flex": "1"}),

                html.Div([dcc.Graph(id="p1-comparacion-barras")], style={"flex": "1"}),

            ], style={"display": "flex", "gap": "30px", "alignItems": "flex-start"}),

        ]),

        # Aquí irán Pregunta 2 y Pregunta 3...

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
        "fontSize": "22px",
        "fontWeight": "bold",
        "textAlign": "center",
        "padding": "30px",
        "borderRadius": "12px",
        "backgroundColor": color_fondo,
        "color": color_texto,
        "boxShadow": "0 4px 12px rgba(0,0,0,0.08)",
        "marginBottom": "20px",
        "minHeight": "80px"
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

    # Reemplaza estos valores con los reales de tu MLflow
    datos = pd.DataFrame([
        {"Modelo": "MLP Base",       "F1 Macro": 0.00, "AUC-ROC": 0.00, "Accuracy": 0.00},
        {"Modelo": "MLP Optimizada", "F1 Macro": 0.00, "AUC-ROC": 0.00, "Accuracy": 0.00},
    ])

    fig = px.bar(
        datos.melt(id_vars="Modelo", var_name="Métrica", value_name="Valor"),
        x="Métrica",
        y="Valor",
        color="Modelo",
        barmode="group",
        color_discrete_map={"MLP Base": "#4ABFBF", "MLP Optimizada": "#2E86AB"},
        title="Comparación de métricas: MLP Base vs MLP Optimizada",
        text_auto=".3f"
    )
    fig.update_layout(
        template="simple_white",
        yaxis_range=[0, 1],
        yaxis_title="Valor",
        legend_title="Modelo"
    )

    return fig


if __name__ == '__main__':
    app.run(debug=True)