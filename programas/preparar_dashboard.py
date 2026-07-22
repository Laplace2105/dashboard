"""
Genera los artefactos ligeros que consume el dashboard (app.py).
Ejecutar UNA VEZ, después de correr los notebooks 02, 03, 04 y 05.

    python programas/preparar_dashboard.py

Convierte limpio.csv (76 MB) → datos.parquet (~1 MB) para que quepa en GitHub.
"""
import os
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data set")

COLS = ["ANIOS", "MES_N", "REGION", "ACTIVIDAD_ECONOMICA", "SEXO",
        "CATEGORIA_OCUPACIONAL", "FORMA_DEL_ACCIDENTE_G", "AGENTE_CAUSANTE_G",
        "ESTACION", "TRIMESTRE", "ES_FIN_DE_ANIO", "PERMANENTE"]

print("Leyendo limpio.csv ...")
df = pd.read_csv(os.path.join(DATA, "limpio.csv"), usecols=COLS)

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
