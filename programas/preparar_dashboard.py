"""
Genera los artefactos ligeros que consume el dashboard (app.py).
Ejecutar UNA VEZ, después de correr los notebooks 02, 03, 04 y 05.

    python programas/preparar_dashboard.py

Convierte limpio.csv (76 MB) → datos.parquet (~1 MB) para que quepa en GitHub.
"""
import os
import sys

import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data set")
sys.path.insert(0, BASE)
import normalizacion as nz  # noqa: E402

COLS = ["ANIOS", "MES_N", "REGION", "ACTIVIDAD_ECONOMICA", "SEXO",
        "CATEGORIA_OCUPACIONAL", "FORMA_DEL_ACCIDENTE_G", "AGENTE_CAUSANTE_G",
        "ESTACION", "TRIMESTRE", "ES_FIN_DE_ANIO", "PERMANENTE"]

print("Leyendo limpio.csv ...")
df = pd.read_csv(os.path.join(DATA, "limpio.csv"), usecols=COLS)

# El parquet se guarda SIN normalizar a propósito: los .pkl del Panel 2 se
# entrenaron con estas cadenas y sus columnas one-hot están congeladas.
# La normalización se aplica al cargar (app.py -> nz.normalizar_datos) y el
# camino de vuelta lo da nz.puente_modelo. Lo que sí se comprueba aquí es
# que el catálogo de normalizacion.py cubra todo lo que trae el CSV; si el
# MTPE vuelve a cambiar de codificación, salta en este punto y no como
# categorías duplicadas dentro del dashboard.
sin_mapear = nz.valores_no_mapeados(df)
if sin_mapear:
    print("\n!! Valores sin regla de normalización. Añádelos a normalizacion.py:")
    for col, valores in sin_mapear.items():
        print(f"   {col}:")
        for v in valores:
            print(f"      [{nz.clave(v)}]  <- {v!r}")
    sys.exit(1)
print("OK  catálogo de normalización cubre todos los valores")

for c in df.columns:
    if df[c].dtype == object:
        df[c] = df[c].astype("category")

salida = os.path.join(DATA, "datos.parquet")
df.to_parquet(salida, index=False)

mb_in = os.path.getsize(os.path.join(DATA, "limpio.csv")) / 1e6
mb_out = os.path.getsize(salida) / 1e6
print(f"OK  limpio.csv ({mb_in:.1f} MB)  ->  datos.parquet ({mb_out:.2f} MB)")
print("\nArchivos que el dashboard necesita en 'data set/':")
for f in ["datos.parquet", "clusters_panel1.csv", "pronostico_panel3.csv", "metricas_panel3.csv"]:
    ruta = os.path.join(DATA, f)
    print(f"  {'OK ' if os.path.exists(ruta) else 'FALTA'}  {f}")
print("\nY en 'models/':")
for f in ["modelo_p0.pkl", "modelo_p1.pkl"]:
    ruta = os.path.join(BASE, "models", f)
    print(f"  {'OK ' if os.path.exists(ruta) else 'FALTA'}  {f}")
