"""
==============================================================================
 DASHBOARD · Accidentes de Trabajo en el Perú (MTPE - Sistema SAT)
 Trabajo Final · Minería de Datos · UNMSM-FISI · 2026-I

 4 paneles:
   1. EDA + Clustering   (K-means, codo, silueta, DBSCAN)
   2. Predictivo         (5 modelos, matriz de confusión, SHAP)
   3. Pronóstico         (series temporales, MAPE y RMSE)
   4. CRUD de consultas  (crear, listar, editar, eliminar)

 Ejecutar:  streamlit run app.py
==============================================================================
"""
import os
import sqlite3
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import normalizacion as nz

# ─────────────────────────────────────────────────────────────── CONFIGURACIÓN
st.set_page_config(page_title="Accidentes de Trabajo · MTPE",
                   page_icon="🏭", layout="wide")

GRANATE, DORADO, AZUL, VERDE, GRIS = "#7a1128", "#d4a72c", "#3b6ea5", "#2e7d5b", "#6b6b6b"
PALETA = [GRANATE, DORADO, AZUL, VERDE, GRIS, "#b5651d"]
TEMPLATE = "plotly_white"

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data set")
DB = os.path.join(BASE, "consultas.db")

st.markdown(f"""
<style>
  .main .block-container {{ padding-top: 2rem; }}
  h1, h2, h3 {{ color: {GRANATE}; }}
  [data-testid="stMetricValue"] {{ color: {GRANATE}; font-weight: 700; }}
</style>
""", unsafe_allow_html=True)


# ───────────────────────────────────────────────────────────────── CARGA DE DATOS
@st.cache_data
def cargar_datos_crudos():
    """Parquet tal cual. Solo lo necesita el puente hacia los .pkl, que
    fueron entrenados con las cadenas originales sin normalizar."""
    return pd.read_parquet(os.path.join(DATA, "datos.parquet"))


@st.cache_data
def cargar_datos():
    """Datos normalizados: vocabulario unificado 2012-2022 / 2023-2024,
    acentos reparados, centinelas de dato faltante colapsados y banderas
    de calidad (TARGET_FIABLE, PERIODO). Es la única fuente para la UI."""
    return nz.normalizar_datos(cargar_datos_crudos())


@st.cache_data
def cargar_pronostico(_df):
    p = os.path.join(DATA, "pronostico_panel3.csv")
    m = os.path.join(DATA, "metricas_panel3.csv")
    if not os.path.exists(p):
        return None, None
    pron = nz.normalizar_pronostico(pd.read_csv(p, parse_dates=["PERIODO"]), _df)
    met = pd.read_csv(m, index_col=0) if os.path.exists(m) else None
    return pron, met


def acortar(texto, n=30):
    """Etiqueta corta para ejes, sin cortar a medio carácter."""
    texto = str(texto)
    return texto if len(texto) <= n else texto[:n - 1].rstrip(" ,.") + "…"


# Las 2 particiones temporales entrenadas en el notebook 03
PARTICIONES = {
    "2018-2021 / test 2022  (principal)": "modelo_p0.pkl",
    "2018-2020 / test 2021  (experimento)": "modelo_p1.pkl",
}


@st.cache_resource
def cargar_paquete(nombre_pkl):
    """Carga un .pkl de partición (trae los 5 modelos, scaler, métricas y tasas)."""
    import joblib
    ruta = os.path.join(BASE, "models", nombre_pkl)
    return joblib.load(ruta) if os.path.exists(ruta) else None


@st.cache_data
def cargar_puente(nombre_pkl):
    """Traductor canónico → texto crudo para el .pkl indicado.

    Sustituye a la antigua cascada de `if 'ADM' in v` escrita a mano, que
    tenía reglas equivocadas: mandaba 'APRISIONAMIENTO O ATRAPAMIENTO' a
    'OTRAS FORMAS' (categoría con 36% de secuelas frente al 7% real) y
    'AGRICULTOR' a 'PEON' cuando en el modelo AGRICULTOR es la categoría
    base. Ahora el mapa se deriva del propio vocabulario de la partición,
    así que no puede desincronizarse de los datos.
    """
    return nz.puente_modelo(cargar_datos_crudos(), cargar_paquete(nombre_pkl))


def estacion_de(mes):
    """Misma regla con la que se construyó la columna ESTACION del dataset."""
    if mes in (12, 1, 2, 3): return "Verano"
    if mes in (4, 5):        return "Otoño"
    if mes in (6, 7, 8):     return "Invierno"
    return "Primavera"


def predecir(paquete, puente, algoritmo, entrada):
    """Reconstruye las features igual que en el entrenamiento y predice.

    `entrada` viene en vocabulario canónico; `puente` lo devuelve al texto
    crudo con el que se generaron las columnas one-hot del modelo.
    """
    mes = int(entrada["mes"])
    t = paquete["tasas"]
    g = paquete["tasa_global"]

    def crudo(columna, valor):
        # Los selectores solo ofrecen valores del vocabulario del modelo,
        # así que un KeyError aquí es un fallo real, no un caso de usuario.
        return puente[columna][valor]

    fila = {
        "MES_N": mes,
        "TRIMESTRE": (mes - 1) // 3 + 1,
        "ES_FIN_DE_ANIO": int(mes in (11, 12)),
        "REGION": crudo("REGION", entrada["region"]),
        "ACTIVIDAD_ECONOMICA": crudo("ACTIVIDAD_ECONOMICA", entrada["sector"]),
        "SEXO": crudo("SEXO", entrada["sexo"]),
        "CATEGORIA_OCUPACIONAL": crudo("CATEGORIA_OCUPACIONAL", entrada["ocupacion"]),
        "FORMA_DEL_ACCIDENTE_G": crudo("FORMA_DEL_ACCIDENTE_G", entrada["forma"]),
        "AGENTE_CAUSANTE_G": crudo("AGENTE_CAUSANTE_G", entrada["agente"]),
        "ESTACION": crudo("ESTACION", estacion_de(mes)),
    }
    fila["TASA_SECTOR"] = t["ACTIVIDAD_ECONOMICA"].get(fila["ACTIVIDAD_ECONOMICA"], g)
    fila["TASA_REGION"] = t["REGION"].get(fila["REGION"], g)
    fila["TASA_FORMA"] = t["FORMA_DEL_ACCIDENTE_G"].get(fila["FORMA_DEL_ACCIDENTE_G"], g)

    CAT = ["REGION", "ACTIVIDAD_ECONOMICA", "SEXO", "CATEGORIA_OCUPACIONAL",
           "FORMA_DEL_ACCIDENTE_G", "AGENTE_CAUSANTE_G", "ESTACION"]

    X = pd.get_dummies(pd.DataFrame([fila]), columns=CAT, drop_first=False, dtype=int)
    X = X.reindex(columns=paquete["columnas"], fill_value=0)
    X[paquete["num_cols"]] = paquete["scaler"].transform(X[paquete["num_cols"]])
    return float(paquete["modelos"][algoritmo].predict_proba(X)[0, 1])


# ─────────────────────────────────────────────────────── BASE DE DATOS (PANEL 4)
def init_db():
    con = sqlite3.connect(DB)
    con.execute("""CREATE TABLE IF NOT EXISTS consultas (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        region TEXT, sector TEXT, sexo TEXT, ocupacion TEXT,
        forma TEXT, agente TEXT, mes INTEGER,
        prediccion TEXT, probabilidad REAL,
        timestamp TEXT)""")
    con.commit()
    con.close()


def crud_crear(d):
    con = sqlite3.connect(DB)
    con.execute("""INSERT INTO consultas
        (region,sector,sexo,ocupacion,forma,agente,mes,prediccion,probabilidad,timestamp)
        VALUES (?,?,?,?,?,?,?,?,?,?)""",
        (d["region"], d["sector"], d["sexo"], d["ocupacion"], d["forma"], d["agente"],
         d["mes"], d["prediccion"], d["probabilidad"],
         datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    con.commit()
    con.close()


# Columna del CRUD → columna del dataset, para normalizar lo que ya está
# guardado. Las consultas creadas antes de unificar el vocabulario tienen
# el texto crudo ('CONSTRUCCIÎ'), y se mostraban así junto a las nuevas.
CAMPOS_CRUD = {
    "region": "REGION", "sector": "ACTIVIDAD_ECONOMICA", "sexo": "SEXO",
    "ocupacion": "CATEGORIA_OCUPACIONAL", "forma": "FORMA_DEL_ACCIDENTE_G",
    "agente": "AGENTE_CAUSANTE_G",
}


def crud_listar():
    con = sqlite3.connect(DB)
    consultas = pd.read_sql("SELECT * FROM consultas ORDER BY id DESC", con)
    con.close()
    for campo, columna in CAMPOS_CRUD.items():
        consultas[campo] = consultas[campo].map(
            lambda v, c=columna: nz.canonico(v, c) if pd.notna(v) else v)
    return consultas


def crud_actualizar(cid, campos):
    """Actualiza una consulta. La predicción se reescribe junto con las
    entradas: dejarla como estaba producía filas donde la región decía una
    cosa y la probabilidad correspondía a otra."""
    con = sqlite3.connect(DB)
    con.execute("""UPDATE consultas SET region=?,sector=?,sexo=?,ocupacion=?,
                   forma=?,agente=?,mes=?,prediccion=?,probabilidad=?,timestamp=?
                   WHERE id=?""",
                (campos["region"], campos["sector"], campos["sexo"], campos["ocupacion"],
                 campos["forma"], campos["agente"], campos["mes"],
                 campos["prediccion"], campos["probabilidad"],
                 datetime.now().strftime("%Y-%m-%d %H:%M:%S"), cid))
    con.commit()
    con.close()


def crud_eliminar(cid):
    con = sqlite3.connect(DB)
    con.execute("DELETE FROM consultas WHERE id=?", (cid,))
    con.commit()
    con.close()


init_db()
df = cargar_datos()
COBERTURA = nz.cobertura_temporal(df)
ANIO_INI, ANIO_FIN = nz.ANIOS_TARGET_FIABLE
df_fiable = df[df["TARGET_FIABLE"]]


def n_categorias(columna, datos=None):
    """Categorías reales: el centinela de dato faltante no es una categoría."""
    valores = (df if datos is None else datos)[columna].astype(str)
    return int(valores[valores != nz.NO_DETERMINADO].nunique())


# ─────────────────────────────────────────────────────────────────────── HEADER
st.title("🏭 Accidentes de Trabajo en el Perú")
st.caption(f"MTPE · Sistema SAT · {COBERTURA['inicio']} a {COBERTURA['fin']}"
           "  |  Minería de Datos · UNMSM-FISI · 2026-I")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Accidentes analizados", f"{len(df):,}")
# La tasa se calcula SOLO sobre los años en que el target es comparable.
# Sobre los 13 años daba 9.0%, un promedio de tres codificaciones distintas.
c2.metric(f"Con secuela permanente ({ANIO_INI}–{ANIO_FIN})",
          f"{df_fiable['PERMANENTE'].mean()*100:.1f}%",
          help="Solo 2018-2022: fuera de esos años la columna de origen "
               "cambió de catálogo y la tasa cae a <2%, lo que no es un "
               "fenómeno real sino un fallo de codificación.")
c3.metric("Regiones", n_categorias("REGION"))
c4.metric("Sectores económicos", n_categorias("ACTIVIDAD_ECONOMICA"))

# Avisos de integridad: el dataset no cubre lo que su rango sugiere.
if len(COBERTURA["meses_faltantes"]):
    faltan = ", ".join(str(m) for m in COBERTURA["meses_faltantes"])
    st.warning(f"⚠️ **Cobertura incompleta.** Faltan {len(COBERTURA['meses_faltantes'])} "
               f"meses en el dataset: {faltan}. Los años "
               f"{', '.join(map(str, COBERTURA['anios_parciales']))} son parciales, "
               "así que sus totales no son comparables con los años completos.")

_sin_mapear = nz.valores_no_mapeados(cargar_datos_crudos())
if _sin_mapear:
    st.error("⚠️ Hay valores sin regla de normalización (se muestran tal cual). "
             f"Añádelos a `normalizacion.py`: {_sin_mapear}")

st.markdown("---")

TABS = st.tabs(["📊 Panel 1 · EDA + Clustering",
                "🤖 Panel 2 · Predictivo",
                "📈 Panel 3 · Pronóstico",
                "📋 Panel 4 · CRUD"])

# ══════════════════════════════════════════════════════ PANEL 1 · EDA + CLUSTERING
with TABS[0]:
    st.header("Panel 1 · Análisis Exploratorio y Clustering")

    # Panel 1 usa la ventana en que el target es fiable (2018-2022).
    dfp = df_fiable
    st.caption(f"Ventana {ANIO_INI}–{ANIO_FIN}: los únicos años en que la columna "
               "PERMANENTE es comparable. Ojo: 2022 solo tiene enero–mayo en el "
               "dataset, así que aporta menos volumen que un año completo.")

    # Excluye el centinela: 'No determinado' no es un sector ni una forma,
    # y encabezaba los rankings como si lo fuera.
    dfp_sector = dfp[dfp["ACTIVIDAD_ECONOMICA"].astype(str) != nz.NO_DETERMINADO]

    # ---------- EDA ----------
    st.subheader("Análisis exploratorio")
    e1, e2 = st.columns(2)

    with e1:
        top = dfp_sector["ACTIVIDAD_ECONOMICA"].value_counts().head(8)
        fig = px.bar(x=top.values, y=[acortar(t) for t in top.index], orientation="h",
                     color=top.values, color_continuous_scale="Reds",
                     title="Sectores con más accidentes")
        fig.update_layout(template=TEMPLATE, height=340, showlegend=False,
                          coloraxis_showscale=False, xaxis_title="accidentes", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    with e2:
        tasa = (dfp_sector.groupby("ACTIVIDAD_ECONOMICA", observed=True)["PERMANENTE"]
                          .agg(["mean", "size"]))
        tasa = tasa[tasa["size"] >= 500].sort_values("mean", ascending=False).head(8)
        fig = px.bar(x=tasa["mean"] * 100, y=[acortar(t) for t in tasa.index], orientation="h",
                     color=tasa["mean"] * 100, color_continuous_scale="Reds",
                     title="% de secuelas PERMANENTES por sector")
        fig.add_vline(x=dfp["PERMANENTE"].mean() * 100, line_dash="dash", line_color=GRIS)
        fig.update_layout(template=TEMPLATE, height=340, showlegend=False,
                          coloraxis_showscale=False, xaxis_title="% permanentes", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    # ---------- Perfil numérico por unidad región×sector (base del EDA numérico y del clustering) ----------
    @st.cache_data
    def perfil_numerico(dfp):
        # Sin el centinela: una unidad "No determinado × Lima" no es un
        # perfil de siniestralidad interpretable.
        dfp = dfp[(dfp["ACTIVIDAD_ECONOMICA"].astype(str) != nz.NO_DETERMINADO)
                  & (dfp["REGION"].astype(str) != nz.NO_DETERMINADO)]
        g = dfp.groupby(["REGION", "ACTIVIDAD_ECONOMICA"], observed=True)
        p = pd.DataFrame({
            "n_accidentes": g.size(),
            "tasa_permanente": g["PERMANENTE"].mean() * 100,
            "prop_masculino": g["SEXO"].apply(lambda s: (s == "Masculino").mean() * 100),
            "concentracion_forma": g["FORMA_DEL_ACCIDENTE_G"].apply(
                lambda s: s.value_counts(normalize=True).iloc[0] * 100),
        }).reset_index()
        p = p[p["n_accidentes"] >= 50].reset_index(drop=True)
        p["unidad"] = (p["ACTIVIDAD_ECONOMICA"].astype(str).map(lambda t: acortar(t, 24))
                       + " · " + p["REGION"].astype(str))
        return p

    perfil = perfil_numerico(dfp)
    VARS = ["tasa_permanente", "prop_masculino", "concentracion_forma", "n_accidentes"]
    NOMBRES = {"tasa_permanente": "% permanente", "prop_masculino": "% masculino",
               "concentracion_forma": "concentración forma", "n_accidentes": "nº accidentes"}

    # ---------- Estadísticas descriptivas ----------
    st.markdown("---")
    st.subheader("📋 Estadísticas descriptivas")
    st.caption("Sobre las unidades región×sector con ≥50 accidentes "
               f"({len(perfil)} unidades). Estas variables describen el *perfil de siniestralidad*.")
    desc = perfil[VARS].describe().T
    desc["mediana"] = perfil[VARS].median()
    desc["IQR"] = perfil[VARS].quantile(.75) - perfil[VARS].quantile(.25)
    desc = desc[["mean", "50%", "std", "min", "max", "IQR"]]
    desc.columns = ["media", "mediana", "desv.est.", "mín", "máx", "IQR"]
    desc.index = [NOMBRES[v] for v in desc.index]
    st.dataframe(desc.round(2), use_container_width=True)

    # ---------- Histogramas + Boxplots ----------
    st.markdown("---")
    st.subheader("📊 Distribuciones (histogramas y boxplots)")
    var_sel = st.selectbox("Variable a explorar", VARS, format_func=lambda v: NOMBRES[v])

    h1, h2 = st.columns(2)
    with h1:
        fig = px.histogram(perfil, x=var_sel, nbins=25, color_discrete_sequence=[GRANATE],
                           title=f"Histograma · {NOMBRES[var_sel]}")
        fig.add_vline(x=perfil[var_sel].mean(), line_dash="dash", line_color=AZUL,
                      annotation_text="media")
        fig.update_layout(template=TEMPLATE, height=330, bargap=0.05,
                          xaxis_title=NOMBRES[var_sel], yaxis_title="frecuencia")
        st.plotly_chart(fig, use_container_width=True)
    with h2:
        fig = px.box(perfil, y=var_sel, points="outliers", color_discrete_sequence=[DORADO],
                     title=f"Boxplot · {NOMBRES[var_sel]} (outliers = regla 1.5·IQR)")
        fig.update_layout(template=TEMPLATE, height=330, yaxis_title=NOMBRES[var_sel])
        st.plotly_chart(fig, use_container_width=True)

    # ---------- Outliers 1.5·IQR ----------
    q1, q3 = perfil[var_sel].quantile(.25), perfil[var_sel].quantile(.75)
    iqr = q3 - q1
    li, ls = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outliers = perfil[(perfil[var_sel] < li) | (perfil[var_sel] > ls)]
    o1, o2 = st.columns([1, 2])
    o1.metric(f"Outliers en '{NOMBRES[var_sel]}'", len(outliers),
              help=f"Fuera del rango [{li:.1f}, {ls:.1f}] según 1.5·IQR")
    with o2:
        if len(outliers):
            st.caption("Unidades atípicas (no se eliminan: son perfiles reales de interés):")
            st.dataframe(outliers[["unidad", "n_accidentes", var_sel]]
                         .sort_values(var_sel, ascending=False).head(6).round(1),
                         use_container_width=True, hide_index=True)
        else:
            st.caption("No hay outliers para esta variable según la regla 1.5·IQR.")

    # ---------- Mapa de correlación ----------
    st.markdown("---")
    st.subheader("🔗 Mapa de correlación")
    corr = perfil[VARS].corr()
    corr.index = [NOMBRES[v] for v in corr.index]
    corr.columns = [NOMBRES[v] for v in corr.columns]
    fig = px.imshow(corr, text_auto=".2f", aspect="auto", zmin=-1, zmax=1,
                    color_continuous_scale="RdBu_r",
                    title="Correlación entre variables del perfil")
    fig.update_layout(template=TEMPLATE, height=400)
    st.plotly_chart(fig, use_container_width=True)
    altas = [(corr.index[i], corr.columns[j], corr.iloc[i, j])
             for i in range(len(corr)) for j in range(i + 1, len(corr))
             if abs(corr.iloc[i, j]) > 0.7]
    if altas:
        st.caption("Correlaciones fuertes (|r|>0.7): " +
                   ", ".join(f"{a}↔{b} ({r:.2f})" for a, b, r in altas))
    else:
        st.caption("✅ Ninguna correlación fuerte (|r|>0.7) → las variables aportan información no redundante.")

    # ---------- Clustering ----------
    st.markdown("---")
    st.subheader("Clustering de unidades región × sector")
    st.caption("Cada punto es una combinación región×sector con ≥50 accidentes. "
               "Agrupamos por su *perfil de siniestralidad*.")

    # >>> Perilla para modificar EN VIVO durante la exposición <<<
    col_k, col_info = st.columns([1, 3])
    k = col_k.slider("Número de clusters (k)", 2, 8, 3,
                     help="Modifica k y observa cómo cambia la silueta")

    # Mismo perfil que arriba: una sola definición evita que el EDA y el
    # clustering se describan sobre poblaciones distintas.
    perfil = perfil_numerico(dfp)
    FEATURES = ["tasa_permanente", "prop_masculino", "concentracion_forma"]
    X = StandardScaler().fit_transform(perfil[FEATURES])

    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
    perfil["cluster"] = km.labels_
    sil = silhouette_score(X, km.labels_)

    m1, m2, m3 = col_info.columns(3)
    m1.metric("Coeficiente de silueta", f"{sil:.3f}",
              delta="cumple meta (≥0.5)" if sil >= 0.5 else "por debajo de 0.5")
    m2.metric("Inercia", f"{km.inertia_:.1f}")
    m3.metric("Unidades", len(perfil))

    g1, g2 = st.columns(2)

    with g1:  # codo + silueta
        ks = range(2, 11)
        iner, sils = [], []
        for kk in ks:
            m = KMeans(n_clusters=kk, random_state=42, n_init=10).fit(X)
            iner.append(m.inertia_)
            sils.append(silhouette_score(X, m.labels_))
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Método del codo", "Silueta"))
        fig.add_trace(go.Scatter(x=list(ks), y=iner, mode="lines+markers",
                                 line=dict(color=GRANATE, width=3)), row=1, col=1)
        fig.add_trace(go.Scatter(x=list(ks), y=sils, mode="lines+markers",
                                 line=dict(color=DORADO, width=3)), row=1, col=2)
        fig.add_vline(x=k, line_dash="dash", line_color=VERDE, row=1, col=1)
        fig.add_vline(x=k, line_dash="dash", line_color=VERDE, row=1, col=2)
        fig.update_layout(template=TEMPLATE, height=330, showlegend=False,
                          title="Selección de k (métricas de calidad)")
        st.plotly_chart(fig, use_container_width=True)

    with g2:  # PCA
        comp = PCA(n_components=2, random_state=42).fit_transform(X)
        perfil["PC1"], perfil["PC2"] = comp[:, 0], comp[:, 1]
        fig = px.scatter(perfil, x="PC1", y="PC2", color=perfil["cluster"].astype(str),
                         size="n_accidentes", hover_name="unidad",
                         hover_data={"tasa_permanente": ":.1f", "PC1": False, "PC2": False},
                         color_discrete_sequence=PALETA, labels={"color": "Cluster"},
                         title="Clusters en el espacio PCA")
        fig.update_layout(template=TEMPLATE, height=330)
        st.plotly_chart(fig, use_container_width=True)

    # ---------- Interpretación ----------
    st.subheader("¿Qué significa cada cluster?")
    resumen = perfil.groupby("cluster")[FEATURES + ["n_accidentes"]].mean().round(1)
    resumen["n_unidades"] = perfil["cluster"].value_counts().sort_index()
    st.dataframe(resumen.style.background_gradient(subset=["tasa_permanente"], cmap="Reds"),
                 use_container_width=True)

    peor = resumen["tasa_permanente"].idxmax()
    ejemplos = perfil[perfil["cluster"] == peor].nlargest(5, "n_accidentes")["unidad"].tolist()
    st.error(f"🔴 **Cluster {peor} — el más peligroso**: "
             f"{resumen.loc[peor, 'tasa_permanente']:.1f}% de secuelas permanentes "
             f"(vs {perfil['tasa_permanente'].mean():.1f}% promedio).  \n"
             f"Ejemplos: {', '.join(ejemplos[:3])}")

# ═══════════════════════════════════════════════════════════ PANEL 2 · PREDICTIVO
with TABS[1]:
    st.header("Panel 2 · Modelo predictivo de severidad")
    st.caption("¿Este accidente dejará al trabajador con una **secuela permanente**?")

    st.info("🛡️ **Anti-*data leakage*:** el modelo NO usa la naturaleza de la lesión ni la parte "
            "del cuerpo (son la *consecuencia*). Predice desde el **contexto laboral** → sirve para **PREVENIR**.")

    # ══ COMBOBOX 1: partición temporal (responde: "¿qué pasa si cambias el train/test?") ══
    st.markdown("#### ⚙️ Configuración del experimento")
    col_p, col_a = st.columns(2)
    nombre_part = col_p.selectbox(
        "🗓️ Partición temporal (train / test)",
        list(PARTICIONES.keys()),
        help="Cambia el corte temporal y observa cómo cambian TODAS las métricas. "
             "Es la respuesta a la pregunta '¿qué pasa si cambias el train/test?'")
    paquete = cargar_paquete(PARTICIONES[nombre_part])

    if paquete is None:
        st.error("No se encontró el .pkl de esta partición. Ejecuta el notebook 03 y copia "
                 "`models/modelo_p0.pkl` y `modelo_p1.pkl`.")
        st.stop()

    puente = cargar_puente(PARTICIONES[nombre_part])

    # ══ COMBOBOX 2: algoritmo (los 5 modelos) ══
    algoritmo = col_a.selectbox("🤖 Algoritmo", list(paquete["modelos"].keys()),
                                help="Los 5 algoritmos entrenados con esta partición")

    met_df = pd.DataFrame(paquete["metricas"]).set_index("Modelo")
    cm_dict = {m["Modelo"]: m["cm"] for m in paquete["metricas"]}

    # Info de la partición activa
    i1, i2, i3, i4 = st.columns(4)
    i1.metric("Entrena con", f"{paquete['n_train']:,}")
    i2.metric("Prueba con", f"{paquete['n_test']:,}")
    i3.metric("% permanente (test)", f"{paquete['tasa_test']:.1f}%")
    mejor_recall = met_df["Recall"].idxmax()
    i4.metric("Mejor recall", f"{met_df['Recall'].max():.3f}", delta=mejor_recall)

    st.markdown("---")

    # ---------- Comparación de los 5 modelos (de la partición elegida) ----------
    st.subheader(f"Comparación de los 5 algoritmos · {nombre_part}")
    cc1, cc2 = st.columns([3, 2])
    with cc1:
        st.dataframe(met_df[["Accuracy", "Precisión", "Recall", "F1", "ROC-AUC"]].style
                     .background_gradient(subset=["Recall", "F1", "ROC-AUC"], cmap="Greens")
                     .format("{:.3f}"), use_container_width=True)
        st.caption("⚠️ Un modelo que dijera *'nunca es permanente'* tendría alto accuracy y "
                   "**recall = 0** → inútil. Por eso el criterio es **RECALL**.")
    with cc2:
        fig = go.Figure()
        for met in ["Precisión", "Recall", "F1", "ROC-AUC"]:
            fig.add_trace(go.Bar(name=met, x=met_df.index, y=met_df[met]))
        fig.update_layout(template=TEMPLATE, height=300, barmode="group",
                          yaxis_range=[0, 1], title="Métricas por modelo",
                          legend=dict(orientation="h", y=1.15))
        st.plotly_chart(fig, use_container_width=True)

    # ---------- Matriz de confusión del algoritmo elegido ----------
    st.subheader(f"Matriz de confusión · {algoritmo}")
    mc1, mc2 = st.columns([1, 2])
    with mc1:
        cm = np.array(cm_dict[algoritmo])
        fig = px.imshow(cm, text_auto=True, color_continuous_scale="Reds",
                        x=["Pred: No perm.", "Pred: Permanente"],
                        y=["Real: No perm.", "Real: Permanente"])
        fig.update_layout(template=TEMPLATE, height=320, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
    with mc2:
        st.markdown(f"""
        **Métricas de {algoritmo}** (partición {nombre_part}):
        - **Recall:** {met_df.loc[algoritmo, 'Recall']:.3f} · **Precisión:** {met_df.loc[algoritmo, 'Precisión']:.3f}
        - **F1:** {met_df.loc[algoritmo, 'F1']:.3f} · **ROC-AUC:** {met_df.loc[algoritmo, 'ROC-AUC']:.3f}

        **Justificación del criterio (recall):** el costo de los errores es asimétrico.
        Un **falso negativo** (predecir "leve" y que el trabajador quede discapacitado de por vida)
        es humano e irreversible; un **falso positivo** solo cuesta una inspección de más.
        Con clases desbalanceadas el **accuracy engaña**, por eso priorizamos **recall**.
        """)

    st.info("💡 **Para la exposición:** cambia la **partición** arriba y observa cómo el recall "
            "baja de ~0.89 (2018-21/2022) a ~0.75 (2018-20/2021). Con más años de entrenamiento el "
            "modelo generaliza mejor. Nota que el **accuracy casi no cambia pero el recall sí** → "
            "por eso el accuracy no es fiable con clases desbalanceadas.")

    # ---------- SHAP ----------
    st.subheader("SHAP · ¿qué empuja a que un accidente sea permanente?")
    shap_img = os.path.join(BASE, "assets", "shap_summary.png")
    if os.path.exists(shap_img):
        st.image(shap_img, use_container_width=True)
    else:
        st.warning("Guarda el `summary_plot` del notebook 03 como `assets/shap_summary.png`.")

    # ---------- Predictor en vivo ----------
    st.markdown("---")
    st.subheader("🔮 Predice un caso")
    st.caption(f"Usa el algoritmo **{algoritmo}** de la partición **{nombre_part}**.")

    # Los selectores se llenan con el vocabulario de la PARTICIÓN, no con el
    # del dataset completo. Antes ofrecían las 25 categorías de los 13 años;
    # al elegir una que el modelo nunca vio (p. ej. 'Punzocortantes', que solo
    # existe desde 2023) el reindex la convertía en ceros y el modelo predecía
    # sobre la categoría base sin avisar.
    def opciones(columna):
        return sorted(puente[columna].keys())

    p1, p2, p3 = st.columns(3)
    v_region = p1.selectbox("Región", opciones("REGION"), key="p2_reg")
    v_sector = p1.selectbox("Sector económico", opciones("ACTIVIDAD_ECONOMICA"), key="p2_sec")
    v_sexo = p2.selectbox("Sexo", opciones("SEXO"), key="p2_sex")
    v_ocup = p2.selectbox("Categoría ocupacional", opciones("CATEGORIA_OCUPACIONAL"), key="p2_ocu")
    v_forma = p3.selectbox("Forma del accidente", opciones("FORMA_DEL_ACCIDENTE_G"), key="p2_for")
    v_agente = p3.selectbox("Agente causante", opciones("AGENTE_CAUSANTE_G"), key="p2_age")
    v_mes = st.slider("Mes", 1, 12, 6, key="p2_mes")
    umbral = st.slider("Umbral de decisión", 0.1, 0.9, 0.5, 0.05,
                       help="Bájalo para detectar más casos graves (↑recall, ↓precisión)")

    if st.button("Predecir", type="primary"):
        prob = predecir(paquete, puente, algoritmo,
                        dict(region=v_region, sector=v_sector, sexo=v_sexo,
                             ocupacion=v_ocup, forma=v_forma,
                             agente=v_agente, mes=v_mes))
        es_perm = prob >= umbral
        r1, r2 = st.columns([1, 2])
        r1.metric("Probabilidad de secuela permanente", f"{prob*100:.1f}%")
        if es_perm:
            r2.error(f"⚠️ **RIESGO ALTO** — se predice **secuela PERMANENTE** "
                     f"(probabilidad {prob*100:.1f}% ≥ umbral {umbral*100:.0f}%)")
        else:
            r2.success(f"✅ Riesgo bajo — no se predice secuela permanente "
                       f"(probabilidad {prob*100:.1f}% < umbral {umbral*100:.0f}%)")
        st.session_state["ultima_prediccion"] = {
            "region": v_region, "sector": v_sector, "sexo": v_sexo, "ocupacion": v_ocup,
            "forma": v_forma, "agente": v_agente, "mes": v_mes,
            "prediccion": "PERMANENTE" if es_perm else "NO PERMANENTE",
            "probabilidad": round(float(prob), 4),
        }
        # El Panel 4 los necesita para recalcular si se edita la consulta.
        st.session_state["contexto_prediccion"] = {
            "particion": nombre_part, "algoritmo": algoritmo, "umbral": umbral}
        st.info("💾 Ve al **Panel 4** para guardar esta consulta.")


# ═══════════════════════════════════════════════════════════ PANEL 3 · PRONÓSTICO
with TABS[2]:
    st.header("Panel 3 · Pronóstico de accidentes mensuales")

    pron, met = cargar_pronostico(df)
    if pron is None:
        st.error("Falta `data set/pronostico_panel3.csv`. Ejecuta el notebook 05 primero.")
    else:
        imputados = pron[pron["IMPUTADO"]]
        # Los meses sin respaldo en el dataset se descartan del histórico:
        # eran una interpolación lineal (+36.5 accidentes exactos cada mes,
        # con conteos fraccionarios) que se dibujaba como dato observado.
        hist = pron[pron["accidentes"].notna() & ~pron["IMPUTADO"]]
        fut = pron.dropna(subset=["pronostico"])

        if len(imputados):
            meses = ", ".join(imputados["PERIODO"].dt.strftime("%Y-%m"))
            st.warning(
                f"⚠️ Se ocultaron **{len(imputados)} meses rellenados por interpolación** "
                f"({meses}): no existen en el dataset y no son observaciones reales. "
                "El corte se ve como un hueco en la serie.")

        if met is not None:
            mejor = met.index[0]
            k1, k2, k3 = st.columns(3)
            k1.metric("Modelo elegido", mejor)
            k2.metric("MAPE", f"{met.loc[mejor, 'MAPE (%)']:.2f}%",
                      delta="cumple meta (≤10%)" if met.loc[mejor, "MAPE (%)"] <= 10 else None)
            k3.metric("RMSE", f"{met.loc[mejor, 'RMSE']:.0f} accidentes")
            if len(imputados):
                st.caption("⚠️ MAPE y RMSE vienen del notebook 05, que evaluó sobre la "
                           "serie **con** los meses interpolados. Hay que recalcularlos "
                           "sobre la serie observada para que sean válidos.")

        # Serie + tendencia + pronóstico.
        # Se reindexa a meses continuos para que el hueco 2022-06/12 aparezca
        # como corte y Plotly no una 2022-05 con 2023-01 en línea recta.
        s = (hist.set_index("PERIODO")["accidentes"]
                 .reindex(pd.date_range(hist["PERIODO"].min(),
                                        hist["PERIODO"].max(), freq="MS")))
        # Sin min_periods: la tendencia también se corta en el hueco en vez
        # de trazar una media sobre meses que no existen.
        tend = s.rolling(12, center=True).mean()

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=s.index, y=s.values, name="Histórico",
                                 connectgaps=False,
                                 line=dict(color=GRANATE, width=1.6)))
        fig.add_trace(go.Scatter(x=tend.index, y=tend.values, name="Tendencia (MM-12)",
                                 connectgaps=False,
                                 line=dict(color=DORADO, width=3)))
        fig.add_trace(go.Scatter(x=fut["PERIODO"], y=fut["pronostico"], name="Pronóstico",
                                 mode="lines+markers",
                                 line=dict(color=VERDE, width=3, dash="dash"),
                                 marker=dict(size=9)))
        # Como cadena ISO: es la forma que plotly interpreta igual en 5.x y 6.x.
        fig.add_vline(x=s.index[-1].strftime("%Y-%m-%d"), line_dash="dot", line_color=GRIS)
        fig.update_layout(template=TEMPLATE, height=440,
                          title="Accidentes por mes: histórico, tendencia y pronóstico",
                          yaxis_title="accidentes / mes",
                          legend=dict(orientation="h", y=1.02, x=0))
        st.plotly_chart(fig, use_container_width=True)

        c1, c2 = st.columns([2, 3])
        with c1:
            st.subheader(f"Pronóstico · {len(fut)} meses")
            tabla_f = fut[["PERIODO", "pronostico"]].copy()
            tabla_f["PERIODO"] = tabla_f["PERIODO"].dt.strftime("%Y-%m")
            tabla_f.columns = ["Mes", "Accidentes esperados"]
            st.dataframe(tabla_f.round(0), use_container_width=True, hide_index=True)
        with c2:
            if met is not None:
                st.subheader("Comparación de modelos")
                st.dataframe(met.style.background_gradient(cmap="RdYlGn_r")
                             .format("{:.2f}"), use_container_width=True)
                st.caption("🔎 **Hallazgo:** el *baseline* de media móvil gana. Tras el COVID la serie "
                           "se estabilizó en un nivel casi plano; sin tendencia fuerte que explotar, "
                           "la media reciente es difícil de batir. **Navaja de Occam** → se elige el "
                           "modelo más simple.")

# ═════════════════════════════════════════════════════════════════ PANEL 4 · CRUD
with TABS[3]:
    st.header("Panel 4 · Consultas guardadas (CRUD)")
    st.caption("Guarda una consulta con sus datos de entrada + la predicción devuelta. "
               "Timestamp automático.")

    ult = st.session_state.get("ultima_prediccion")

    # ---------- CREATE ----------
    st.subheader("➕ Guardar consulta")
    if ult is None:
        st.warning("Primero realiza una predicción en el **Panel 2**.")
    else:
        cc = st.columns(4)
        cc[0].write(f"**Región:** {ult['region']}")
        cc[1].write(f"**Sector:** {acortar(ult['sector'], 24)}")
        cc[2].write(f"**Predicción:** {ult['prediccion']}")
        cc[3].write(f"**Probabilidad:** {ult['probabilidad']*100:.1f}%")
        if st.button("💾 Guardar esta consulta", type="primary"):
            crud_crear(ult)
            st.success("Consulta guardada ✓")
            st.rerun()

    # ---------- READ ----------
    st.markdown("---")
    st.subheader("📄 Consultas guardadas")
    consultas = crud_listar()

    if consultas.empty:
        st.info("Aún no hay consultas guardadas.")
    else:
        vista = consultas.copy()
        # Formateo tolerante a NULL: `(None * 100)` producía la cadena "nan%".
        vista["probabilidad"] = vista["probabilidad"].map(
            lambda p: f"{p*100:.1f}%" if pd.notna(p) else "—")
        st.dataframe(vista, use_container_width=True, hide_index=True)

        st.markdown("---")
        u1, u2 = st.columns(2)

        # ---------- UPDATE ----------
        with u1:
            st.subheader("✏️ Editar")
            # Se reutiliza la partición/algoritmo con que se hizo la predicción
            # original, para que el valor recalculado sea comparable.
            ctx = st.session_state.get("contexto_prediccion", {})
            part_edit = ctx.get("particion", next(iter(PARTICIONES)))
            paq_edit = cargar_paquete(PARTICIONES[part_edit])
            puente_edit = cargar_puente(PARTICIONES[part_edit])
            alg_edit = ctx.get("algoritmo", next(iter(paq_edit["modelos"])))
            umbral_edit = ctx.get("umbral", 0.5)

            cid = st.selectbox("Consulta a editar", consultas["id"].tolist(), key="edit_id")
            fila = consultas[consultas["id"] == cid].iloc[0]

            regiones = sorted(puente_edit["REGION"].keys())
            n_region = st.selectbox(
                "Región", regiones,
                index=regiones.index(fila["region"]) if fila["region"] in regiones else 0,
                key="edit_reg")
            n_mes = st.slider("Mes", 1, 12, int(fila["mes"]), key="edit_mes")
            st.caption(f"Se recalculará con **{alg_edit}** · {part_edit} · umbral "
                       f"{umbral_edit*100:.0f}%")

            if st.button("Actualizar"):
                campos = {"region": n_region, "sector": fila["sector"],
                          "sexo": fila["sexo"], "ocupacion": fila["ocupacion"],
                          "forma": fila["forma"], "agente": fila["agente"],
                          "mes": n_mes}
                # Una consulta guardada con vocabulario que esta partición no
                # conoce no se puede reevaluar: se marca en vez de inventar.
                desconocidos = [c for c, col in CAMPOS_CRUD.items()
                                if campos[c] not in puente_edit[col]]
                if desconocidos:
                    campos["prediccion"] = "NO RECALCULABLE"
                    campos["probabilidad"] = None
                    st.warning("Guardado sin probabilidad: "
                               f"{', '.join(desconocidos)} no existe(n) en el "
                               f"vocabulario de la partición {part_edit}.")
                else:
                    prob_edit = predecir(paq_edit, puente_edit, alg_edit,
                                         dict(region=campos["region"], sector=campos["sector"],
                                              sexo=campos["sexo"], ocupacion=campos["ocupacion"],
                                              forma=campos["forma"], agente=campos["agente"],
                                              mes=campos["mes"]))
                    campos["prediccion"] = ("PERMANENTE" if prob_edit >= umbral_edit
                                            else "NO PERMANENTE")
                    campos["probabilidad"] = round(float(prob_edit), 4)
                crud_actualizar(cid, campos)
                st.success(f"Consulta #{cid} actualizada ✓")
                st.rerun()

        # ---------- DELETE ----------
        with u2:
            st.subheader("🗑️ Eliminar")
            did = st.selectbox("Consulta a eliminar", consultas["id"].tolist(), key="del_id")
            st.warning("Esta acción no se puede deshacer.")
            if st.button("Eliminar", type="secondary"):
                crud_eliminar(did)
                st.success(f"Consulta #{did} eliminada ✓")
                st.rerun()

# ─────────────────────────────────────────────────────────────────────── FOOTER
st.markdown("---")
st.caption("Datos: MTPE · Sistema SAT (datosabiertos.gob.pe) · "
           "CRISP-DM · UNMSM-FISI · Minería de Datos 2026-I")