# 🏭 Accidentes de Trabajo en el Perú — Dashboard CRISP-DM

Trabajo Final · **Minería de Datos** · UNMSM-FISI · 2026-I
Datos: **MTPE — Sistema SAT** (datosabiertos.gob.pe) · 276,646 accidentes · 2012–2024

**Pregunta:** ¿bajo qué condiciones laborales un accidente deja al trabajador con una **secuela permanente**?

---

## 📁 Estructura del proyecto

```
proyecto/
├── app.py                    ← EL DASHBOARD (4 paneles)
├── requirements.txt
├── data set/
│   ├── Dataset__Noti_AT_No_Mort_SAT_0.csv   (crudo, NO se sube a GitHub)
│   ├── limpio.csv            (genera notebook 02, NO se sube: 76 MB)
│   ├── datos.parquet         ← el dashboard lee ESTE (0.9 MB) ✅ sí se sube
│   ├── clusters_panel1.csv   (genera notebook 04)
│   ├── pronostico_panel3.csv (genera notebook 05)
│   └── metricas_panel3.csv   (genera notebook 05)
├── models/
│   ├── modelo_p0.pkl         (partición 2018-2021/2022 · genera notebook 03)
│   └── modelo_p1.pkl         (partición 2018-2020/2021 · genera notebook 03)
├── assets/
│   └── shap_summary.png      (exportar del notebook 03)
└── programas/
    ├── 01_eda_accidentes_trabajo.ipynb     Fase 2 · Comprensión
    ├── 02_preparacion_accidentes_trabajo.ipynb  Fase 3 · Preparación
    ├── 03_modelado_panel2.ipynb            Panel 2 · 5 modelos + SHAP
    ├── 04_panel1_eda_clustering.ipynb      Panel 1 · K-means
    ├── 05_panel3_pronostico.ipynb          Panel 3 · Pronóstico
    └── preparar_dashboard.py               genera datos.parquet
```

---

## ▶️ Cómo ejecutarlo

### 1. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 2. Ejecutar los notebooks EN ORDEN
`01` → `02` → `03` → `04` → `05`
(generan `limpio.csv`, `modelo_p0.pkl`, `modelo_p1.pkl`, `clusters_panel1.csv`, `pronostico_panel3.csv`)

### 3. Comprimir los datos para el dashboard
```bash
python programas/preparar_dashboard.py
```
Convierte `limpio.csv` (76 MB) → `datos.parquet` (0.9 MB), que sí cabe en GitHub.

### 4. Levantar el dashboard
```bash
streamlit run app.py
```
Se abre en `http://localhost:8501`

---

## ☁️ Desplegar en Streamlit Cloud (obligatorio: debe estar EN LÍNEA)

1. Sube el proyecto a un repositorio de **GitHub** (el `.gitignore` ya excluye los CSV pesados).
2. Entra a **share.streamlit.io** → *New app* → conecta el repo.
3. *Main file path:* `app.py`
4. **Deploy.** Obtendrás una URL pública para la presentación.

> ⚠️ Verifica que `data set/datos.parquet`, `data set/clusters_panel1.csv`,
> `data set/pronostico_panel3.csv`, `data set/metricas_panel3.csv` y `models/modelo_panel2.pkl`
> **sí estén** en el repo. Sin ellos el dashboard no arranca.

---

## 📊 Los 4 paneles

| Panel | Contenido | Métricas |
|---|---|---|
| **1 · EDA + Clustering** | Estadísticas, barras, K-means con **método del codo**, PCA 2D, interpretación | **Silueta 0.541** · Inercia |
| **2 · Predictivo** | **5 algoritmos** × **2 particiones** (combobox), matriz de confusión, SHAP, predictor | **Recall 0.894** · ROC-AUC 0.907 |
| **3 · Pronóstico** | Serie + tendencia, 4 modelos, pronóstico a 6 meses | **MAPE 6.13%** · RMSE 222 |
| **4 · CRUD** | Guardar / listar / editar / eliminar consultas + timestamp | SQLite |

---

## 🎯 Decisiones metodológicas clave (para la exposición)

1. **Anti-*data leakage* de variables** — se **excluyen** `NATURALEZA_DE_LA_LESION` y
   `PARTE_DEL_CUERPO_LESIONADA`: describen la **consecuencia** (una *amputación* es obviamente
   permanente). El modelo predice desde el **contexto laboral** → sirve para **PREVENIR**.

2. **Anti-*concept drift*** — el MTPE **cambió su criterio de clasificación en 2023**
   (aparece "PARCIAL TEMPORAL" con 13% y "PARCIAL PERMANENTE" cae de 20% a 0.2%).
   El modelado se restringe al régimen estable **2018–2022**.

3. **Anti-leakage temporal** — *split* por año (train 2018-21 / test 2022);
   las tasas de riesgo y el `StandardScaler` se ajustan **solo con el train**;
   **SMOTE solo en el train** (el test conserva la proporción real).

4. **Recall > Accuracy** — con clases 18/82, un modelo que diga *"nunca es permanente"*
   logra **79% de accuracy** y **recall = 0**. El **falso negativo** (no detectar un accidente
   que discapacitará de por vida) es el error más caro.

5. **Navaja de Occam en el pronóstico** — el *baseline* de media móvil **gana** a Holt-Winters
   y SARIMA. Se reporta con honestidad: un modelo complejo que no mejora al baseline no debe usarse.

---

## 🔧 Modificación de código en vivo (requisito de la exposición)

| Integrante | Panel | Qué puede modificar en vivo |
|---|---|---|
| **A** | Panel 1 | El slider de **k** → ver cómo cambian la silueta y los clusters |
| **B** | Panel 2 | La **partición** (2018-21/2022 vs 2018-20/2021) y el **umbral** → ver cómo cambian las métricas |
| **C** | Panel 3 + 4 | El modelo de pronóstico · el flujo CRUD completo |
