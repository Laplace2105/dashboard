"""
==============================================================================
 CAPA ÚNICA DE NORMALIZACIÓN  ·  Accidentes de Trabajo (MTPE - Sistema SAT)
==============================================================================

Problema que resuelve
---------------------
El dataset del SAT mezcla DOS regímenes de codificación incompatibles:

  * 2012-2022 : catálogo largo  ("CAIDA DE OBJETOS", "EXPLOTACIÓN DE MINAS…")
  * 2023-2024 : catálogo nuevo, más grueso y abreviado
                ("CADAS DE OBJETOS", "EXPLOTAC. MINAS…", "ADM.PLICA…")

Además los acentos llegaron corrompidos en el volcado original
(Ó→'Î', Ú→'\\x82', y Á/É/Í/Ñ simplemente desaparecieron), de modo que
'CONSTRUCCIÓN' aparece como 'CONSTRUCCII'/'CONSTRUCCIÎ'.

Consecuencia si no se normaliza: el mismo concepto se cuenta dos veces
(25 "sectores" que en realidad son 17), los gráficos muestran barras
duplicadas y los `value_counts()` se reparten entre variantes.

Cómo funciona
-------------
1. `clave()` reduce cualquier variante a una llave ASCII estable
   (mayúsculas, sin acentos, sin puntuación, sin caracteres corruptos).
2. Los diccionarios `SECTOR`, `FORMA`, … mapean esa llave a UNA etiqueta
   canónica legible.
3. `normalizar_datos()` se aplica UNA sola vez al cargar el parquet, así
   los 4 paneles, los selectores y el CRUD comparten el mismo vocabulario.
4. `puente_modelo()` traduce de vuelta canónico → texto crudo original,
   porque los .pkl fueron entrenados con las cadenas corruptas y sus
   columnas one-hot están congeladas.

Criterio aplicado en los mapeos
-------------------------------
Se unifican solo las variantes que son EL MISMO concepto. Cuando el
catálogo 2023-2024 agrupa varias categorías del catálogo antiguo
(p. ej. "CADAS DE PERSONAS" no distingue "a nivel" de "de altura"),
NO se inventa una desagregación: se conserva como categoría propia
marcada "(sin detalle)". Repartirla sería fabricar información.
==============================================================================
"""
import re
import unicodedata

import pandas as pd

# Etiqueta única para todos los centinelas de dato faltante.
NO_DETERMINADO = "No determinado"

# Años en los que el target PERMANENTE es comparable.
#
# PERMANENTE se derivó en el notebook 02 como
#     ACCIDENTE_INCAPACITANTE ∈ {"PARCIAL PERMANENTE", "TOTAL PERMANENTE"}
# pero esa columna cambió de catálogo fuera de 2018-2022, así que el
# `isin` deja de encontrar coincidencias y la tasa se desploma:
#
#     2012  0.5%   2015  3.3%   2018 17.2%   2021 17.7%   2024  0.3%
#     2013  1.2%   2016  1.3%   2019 18.3%   2022 20.6%
#     2014  9.4%   2017  1.9%   2020 18.7%   2023  0.2%
#
# El salto 1.9% → 17.2% → 0.2% no es un fenómeno real: es el catálogo
# cambiando. Promediar los 13 años produce un número sin significado.
ANIOS_TARGET_FIABLE = (2018, 2022)

# Columnas categóricas que pasan por normalización.
COLUMNAS_CATEGORICAS = [
    "REGION", "ACTIVIDAD_ECONOMICA", "SEXO", "CATEGORIA_OCUPACIONAL",
    "FORMA_DEL_ACCIDENTE_G", "AGENTE_CAUSANTE_G", "ESTACION",
]


def clave(valor):
    """Llave canónica de comparación: ASCII, mayúsculas, sin puntuación.

    Absorbe la corrupción de acentos del volcado original, de modo que
    'CONSTRUCCIÎ' y 'CONSTRUCCIÓN' caen en la misma llave.
    """
    texto = unicodedata.normalize("NFD", str(valor).upper())
    texto = "".join(c for c in texto if unicodedata.category(c) != "Mn")
    texto = re.sub(r"[^A-Z0-9]+", " ", texto)
    return re.sub(r"\s+", " ", texto).strip()


# ───────────────────────────────────────────────────── ACTIVIDAD ECONÓMICA
# 25 variantes crudas → 17 sectores reales + centinela.
SECTOR = {
    "ACT INMOBILIARIAS EMP Y ALQ": "Actividades inmobiliarias y empresariales",
    "ACTIVIDADES INMOBILIARIAS EMPRESARIALES Y DE ALQUILER": "Actividades inmobiliarias y empresariales",
    "ADM P LICA PLANES DE SEG SOC": "Administración pública y defensa",
    "ADMINISTRACII P LICA Y DEFENSA": "Administración pública y defensa",
    "AGRICULT GANAD CAZA Y SILVIC": "Agricultura, ganadería y silvicultura",
    "AGRICULTURA GANADERA CAZA Y SILVICULTURA": "Agricultura, ganadería y silvicultura",
    "COMERCIO AL POR MAYOR Y AL POR MENOR REP VEHC AUTOM": "Comercio y reparación de vehículos",
    "COMERCIO AL POR MAYOR Y AL POR MENOR REPARACII DE VEHCULOS AUTOMOTORES": "Comercio y reparación de vehículos",
    "CONSTRUCCII": "Construcción",
    "ENSEANZA": "Enseñanza",
    "EXPLOTACII DE MINAS Y CANTERAS": "Explotación de minas y canteras",
    "HOGARES PRIVADOS CON SERVICIO DOMSTICO": "Hogares con servicio doméstico",
    "HOTELES Y RESTAURANTES": "Hoteles y restaurantes",
    "INDUSTRIAS MANUFACTURERAS": "Industrias manufactureras",
    "INTERMEDIACII FINANCIERA": "Intermediación financiera",
    "ORGANIZACIONES Y OGANOS EXTRATERRITORIALES": "Organizaciones extraterritoriales",
    "OTRAS ACT SERV COM SOC Y PER": "Otros servicios comunitarios y personales",
    "OTRAS ACTIV SERV COMUNITARIOS SOCIALES Y PERSONALES": "Otros servicios comunitarios y personales",
    "PESCA": "Pesca",
    "SERVICIOS SOCIALES Y DE SALUD": "Servicios sociales y de salud",
    "SUMIN ELECTRICIDAD GAS Y AGUA": "Suministro de electricidad, gas y agua",
    "SUMINISTRO DE ELECTRICIDAD GAS Y AGUA": "Suministro de electricidad, gas y agua",
    "TRANSPORTE ALMACENAMIENTO Y COMUNICACIONES": "Transporte, almacenamiento y comunicaciones",
    "TRANSPORTES ALMACENAM Y COMUN": "Transporte, almacenamiento y comunicaciones",
    "NO DETERMINADO": NO_DETERMINADO,
}

# ─────────────────────────────────────────────────── FORMA DEL ACCIDENTE
# Ojo: "OTRAS FORMAS" (residual real, 2012-2022) y "OTRAS FORMAS … POR
# FALTA DE DATOS SUFICIENTES" (2023-2024) NO son lo mismo. El segundo es
# un centinela de dato faltante; fusionarlos contaminaba la categoría con
# la tasa de secuela permanente más alta del catálogo (36%).
FORMA = {
    "APRISIONAMIENTO O ATRAPAMIENTO": "Aprisionamiento o atrapamiento",
    "CADAS DE OBJETOS": "Caída de objetos",
    "CAIDA DE OBJETOS": "Caída de objetos",
    "CAIDA DE PERSONAS A NIVEL": "Caída de personas a nivel",
    "CAIDA DE PERSONAL DE ALTURA": "Caída de personas de altura",
    # El catálogo 2023-2024 no separa nivel/altura: se conserva aparte.
    "CADAS DE PERSONAS": "Caída de personas (sin detalle)",
    "CHOQUE CONTRA OBJETO": "Choque contra objeto",
    "ESFUERZOS EXCESIVOS O FALSOS MOVIMIENTOS": "Esfuerzos físicos o falsos movimientos",
    "ESFUERZOS FISICOS O FALSOS MOVIMIENTOS": "Esfuerzos físicos o falsos movimientos",
    "GOLPES POR OBJETOS EXCEPTO CAIDAS": "Golpes por objetos (excepto caídas)",
    "PISADAS SOBRE OBJETO": "Pisadas sobre objeto",
    # Agrupado 2023-2024 que cubre pisadas + choques + golpes a la vez.
    "PISADAS SOBRE CHOQUES CONTRA O GOLPES POR OBJETOS A EXCEPCII DE CADAS DE OBJETOS":
        "Pisadas, choques o golpes (agrupado)",
    "PUNZO CORTANTES": "Punzocortantes",
    "OTRAS FORMAS": "Otras formas",
    "OTROS": "Otros",
    "OTRAS FORMAS DE ACCIDENTE NO CLASIFICADAS POR FALTA DE DATOS SUFICIENTES": NO_DETERMINADO,
}

# ──────────────────────────────────────────────────────── AGENTE CAUSANTE
AGENTE = {
    "AMBIENTE DEL TRABAJO": "Ambiente de trabajo",
    "ESCALERA": "Escalera",
    "HERRAMIENTAS PORTATILES MANUALES MECNICOS ELCTRICAS NEUMTICAS ETC": "Herramientas",
    "MAQUINAS Y EQUIPOS EN GENERAL": "Máquinas y equipos",
    "MQUINAS": "Máquinas y equipos",
    "MATERIAS PRIMAS": "Materias primas",
    # Categoría más amplia del catálogo nuevo; no equivale a "materias primas".
    "MATERIALES SUSTANCIAS Y RADIACIONES": "Materiales, sustancias y radiaciones",
    "MUEBLES EN GENERAL": "Muebles en general",
    "OTROS APARATOS": "Otros aparatos",
    "PISO": "Piso",
    "SUSTANCIAS QUIMICAS PLAGUICIDAS": "Sustancias químicas y plaguicidas",
    "VEHICULOS O MEDIOS DE TRANSPORTE EN GENERAL": "Vehículos y medios de transporte",
    "OTROS": "Otros",
    "DESCONOCIDO": NO_DETERMINADO,
    "AGENTES NO CLASIFICADOS POR FALTA DE DATOS SUFICIENTES": NO_DETERMINADO,
}

# ────────────────────────────────────────────────── CATEGORÍA OCUPACIONAL
OCUPACION = {
    "AGRICULTOR": "Agricultor",
    "CAPATAZ": "Capataz",
    "EMPLEADO": "Empleado",
    "FUNCIONARIO": "Funcionario",
    "JEFE DE PLANTA": "Jefe de planta",
    "OBRERO": "Obrero",
    "OFICIAL": "Oficial",
    "OPERARIO": "Operario",
    "PEON": "Peón",
    "TECNICO": "Técnico",
    "TRABAJADOR INDEPENDIENTE": "Trabajador independiente",
    "OTROS": "Otros",
    "DESCONOCIDO": NO_DETERMINADO,
}

# ─────────────────────────────────────────────────────────────── REGIÓN
REGION = {
    "AMAZONAS": "Amazonas", "ANCASH": "Áncash", "APURIMAC": "Apurímac",
    "AREQUIPA": "Arequipa", "AYACUCHO": "Ayacucho", "CAJAMARCA": "Cajamarca",
    "CALLAO": "Callao", "CUSCO": "Cusco", "HUANCAVELICA": "Huancavelica",
    "HUANUCO": "Huánuco", "ICA": "Ica", "JUNIN": "Junín",
    "LA LIBERTAD": "La Libertad", "LAMBAYEQUE": "Lambayeque", "LIMA": "Lima",
    "LORETO": "Loreto", "MADRE DE DIOS": "Madre de Dios", "MOQUEGUA": "Moquegua",
    "PASCO": "Pasco", "PIURA": "Piura", "PUNO": "Puno",
    "SAN MARTIN": "San Martín", "TACNA": "Tacna", "TUMBES": "Tumbes",
    "UCAYALI": "Ucayali",
    "DESCONOCIDO": NO_DETERMINADO,
}

SEXO = {"MASCULINO": "Masculino", "FEMENINO": "Femenino"}

ESTACION = {"VERANO": "Verano", "OTONO": "Otoño", "OTOO": "Otoño",
            "INVIERNO": "Invierno", "PRIMAVERA": "Primavera"}

MAPAS = {
    "ACTIVIDAD_ECONOMICA": SECTOR,
    "FORMA_DEL_ACCIDENTE_G": FORMA,
    "AGENTE_CAUSANTE_G": AGENTE,
    "CATEGORIA_OCUPACIONAL": OCUPACION,
    "REGION": REGION,
    "SEXO": SEXO,
    "ESTACION": ESTACION,
}


def canonico(valor, columna):
    """Traduce un valor crudo a su etiqueta canónica.

    Si el valor no está en el catálogo se devuelve tal cual (en vez de
    descartarlo) para que un dato nuevo sea visible en vez de silencioso;
    `valores_no_mapeados()` los reporta para poder ampliar el mapa.
    """
    return MAPAS[columna].get(clave(valor), str(valor))


def valores_no_mapeados(df):
    """Valores presentes en los datos que ningún mapa cubre. Vacío = todo OK."""
    faltantes = {}
    for col, mapa in MAPAS.items():
        if col not in df.columns:
            continue
        desconocidos = sorted({str(v) for v in df[col].dropna().unique()
                               if clave(v) not in mapa})
        if desconocidos:
            faltantes[col] = desconocidos
    return faltantes


def normalizar_datos(df):
    """Normaliza el parquet crudo. Aplicar UNA vez, al cargar.

    Además de unificar el vocabulario añade las banderas de calidad que
    los paneles necesitan para no mezclar regímenes incompatibles:
      · PERIODO        - primer día del mes, como datetime
      · TARGET_FIABLE  - el año cae dentro de ANIOS_TARGET_FIABLE
      · MES_COMPLETO   - el mes tiene registros en el dataset
    """
    df = df.copy()

    for col, mapa in MAPAS.items():
        if col in df.columns:
            crudo = df[col].astype(str)
            # Mapear por llave: una sola pasada sobre los valores únicos.
            traduccion = {v: mapa.get(clave(v), v) for v in crudo.unique()}
            df[col] = crudo.map(traduccion).astype("category")

    # Tipos numéricos: ANIOS venía como float64 (2012.0) y se mostraba así.
    df["ANIOS"] = df["ANIOS"].astype(int)
    for col in ("MES_N", "TRIMESTRE", "ES_FIN_DE_ANIO", "PERMANENTE"):
        if col in df.columns:
            df[col] = df[col].astype(int)

    df["PERIODO"] = pd.to_datetime(
        {"year": df["ANIOS"], "month": df["MES_N"], "day": 1})

    lo, hi = ANIOS_TARGET_FIABLE
    df["TARGET_FIABLE"] = df["ANIOS"].between(lo, hi)

    return df


def cobertura_temporal(df):
    """Resumen de qué meses existen realmente y cuáles faltan.

    El parquet dice cubrir 2012-2024 pero le faltan jun-dic 2022 y
    jun-dic 2024. Los paneles deben decirlo en vez de presentar años
    parciales como si fueran completos.
    """
    meses = pd.PeriodIndex(df["PERIODO"].unique(), freq="M").sort_values()
    completo = pd.period_range(meses.min(), meses.max(), freq="M")
    faltantes = completo.difference(meses)
    anios_parciales = sorted({
        int(a) for a in df["ANIOS"].unique()
        if df.loc[df["ANIOS"] == a, "MES_N"].nunique() < 12
    })
    return {
        "inicio": meses.min(),
        "fin": meses.max(),
        "meses_faltantes": faltantes,
        "anios_parciales": anios_parciales,
    }


def normalizar_pronostico(pron, df):
    """Marca los meses del CSV de pronóstico que no existen en los datos.

    `pronostico_panel3.csv` trae jun-dic 2022 rellenados por interpolación
    lineal (3076.5, 3113.0, 3149.5 … +36.5 exacto cada mes, con conteos de
    accidentes fraccionarios). Esos 7 meses no existen en el dataset y se
    estaban dibujando como "Histórico" real.

    Devuelve el mismo DataFrame con la columna booleana IMPUTADO.
    """
    pron = pron.copy()
    reales = set(pd.PeriodIndex(df["PERIODO"].unique(), freq="M"))
    periodo_mes = pd.PeriodIndex(pron["PERIODO"], freq="M")
    pron["IMPUTADO"] = pron["accidentes"].notna() & ~periodo_mes.isin(reales)
    return pron


def puente_modelo(df_crudo, paquete):
    """Mapa canónico → texto crudo, por columna, para un .pkl dado.

    Los modelos se entrenaron con las cadenas corruptas del parquet y sus
    columnas one-hot están congeladas (`paquete["columnas"]`), así que hay
    que traducir de vuelta antes de predecir.

    El vocabulario sale de los años de ENTRENAMIENTO, no de train+test:
    `get_dummies` solo vio el train. En la partición 2018-2020/2021 hay dos
    valores ('MADRE DE DIOS' y 'HOGARES PRIVADOS CON SERVICIO DOMÉSTICO')
    que aparecen únicamente en 2021, así que no tienen columna propia; si se
    ofrecieran en el selector, `reindex` los volvería ceros y el modelo
    respondería con la categoría base sin avisar de la sustitución.
    """
    sub = df_crudo[df_crudo["ANIOS"].astype(int).isin(set(paquete["train_anios"]))]

    puente = {}
    for col in MAPAS:
        if col not in sub.columns:
            continue
        crudos = {str(v) for v in sub[col].dropna().unique()}
        # `drop_first=True` deja una categoría sin columna: es la base y se
        # representa con todos los dummies en cero, así que sigue siendo
        # seleccionable. Cualquier otro valor sin columna sí sería inválido.
        dummies = {c[len(col) + 1:] for c in paquete["columnas"]
                   if c.startswith(col + "_")}
        sin_columna = crudos - dummies
        if len(sin_columna) > 1:
            # No se puede saber cuál es la base: se descartan todas menos
            # las que sí tienen columna, para no predecir sobre una
            # sustitución silenciosa.
            crudos &= dummies
        puente[col] = {canonico(v, col): v for v in sorted(crudos)}
    return puente
