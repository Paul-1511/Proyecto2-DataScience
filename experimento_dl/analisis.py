"""
Análisis Exploratorio de Datos (EDA) - RSNA 2022 Cervical Spine Fracture Detection

"""

import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
TRAIN_PATH = 'train.csv'
BBOX_PATH = 'train_bounding_boxes.csv'
IMG_DIR = 'imgs'                     # carpeta donde se guardarán los gráficos
VERT_COLS = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']

os.makedirs(IMG_DIR, exist_ok=True)
pd.set_option('display.width', 150)
plt.rcParams['figure.dpi'] = 110
sns.set_style('whitegrid')

# ---------------------------------------------------------------------------
# 1. Carga de datos
# ---------------------------------------------------------------------------
train = pd.read_csv(TRAIN_PATH)
bbox = pd.read_csv(BBOX_PATH)

# ---------------------------------------------------------------------------
# 2. Estructura general
# ---------------------------------------------------------------------------
print("=== TRAIN.CSV ===")
print("Shape:", train.shape)
print(train.dtypes)
print(train.info())
print()

print("=== BBOX.CSV ===")
print("Shape:", bbox.shape)
print(bbox.dtypes)
print(bbox.info())
print()

print("Nulos en train:\n", train.isnull().sum())
print("Nulos en bbox:\n", bbox.isnull().sum())
print()

print("Duplicados en train:", train['StudyInstanceUID'].duplicated().sum())
print("Estudios únicos en bbox:", bbox['StudyInstanceUID'].nunique())
print("Estudios únicos en train:", train['StudyInstanceUID'].nunique())
print()

# ---------------------------------------------------------------------------
# 3. Resúmenes numéricos y tablas de frecuencia
# ---------------------------------------------------------------------------
print("=== Frecuencias patient_overall (0=sin fractura, 1=con fractura) ===")
print(train['patient_overall'].value_counts())
print(train['patient_overall'].value_counts(normalize=True).round(3))
print()

print("=== Frecuencia de fractura por vértebra ===")
for c in VERT_COLS:
    vc = train[c].value_counts()
    pct = train[c].mean() * 100
    print(f"{c}: fracturada={vc.get(1, 0)} ({pct:.1f}%)  sana={vc.get(0, 0)}")
print()

train['n_fracturas'] = train[VERT_COLS].sum(axis=1)
print("=== Número de vértebras fracturadas por paciente ===")
print(train['n_fracturas'].describe())
print(train['n_fracturas'].value_counts().sort_index())
print()

print("=== Resumen numérico de bounding boxes ===")
print(bbox[['x', 'y', 'width', 'height', 'slice_number']].describe())
print()

boxes_per_study = bbox.groupby('StudyInstanceUID').size()
print("=== Cajas por estudio (solo estudios anotados) ===")
print(boxes_per_study.describe())
print()

# ---------------------------------------------------------------------------
# 4. Cruces de variables clave
# ---------------------------------------------------------------------------

# 4.1 Consistencia entre patient_overall y suma de vértebras
inconsist_1 = train[(train['patient_overall'] == 0) & (train['n_fracturas'] > 0)]
inconsist_2 = train[(train['patient_overall'] == 1) & (train['n_fracturas'] == 0)]
print("=== Consistencia patient_overall vs suma de vértebras ===")
print("Casos patient_overall=0 con alguna vértebra=1:", len(inconsist_1))
print("Casos patient_overall=1 sin ninguna vértebra=1:", len(inconsist_2))
print()

# 4.2 Correlación (co-ocurrencia) entre vértebras
print("=== Correlación entre vértebras (co-ocurrencia de fracturas) ===")
print(train[VERT_COLS].corr().round(2))
print()

# 4.3 Relación entre bboxes y patient_overall
studies_with_bbox = set(bbox['StudyInstanceUID'].unique())
train['tiene_bbox'] = train['StudyInstanceUID'].isin(studies_with_bbox)
print("=== Cruce: estudios con bbox vs patient_overall ===")
print(pd.crosstab(train['tiene_bbox'], train['patient_overall'], margins=True))
print(train.groupby('tiene_bbox')['patient_overall'].mean())
print()

# 4.4 Relación entre severidad (n_fracturas) y cantidad de bboxes por estudio
bbox_count = bbox.groupby('StudyInstanceUID').size().rename('n_boxes')
merged = train.merge(bbox_count, on='StudyInstanceUID', how='left')
merged['n_boxes'] = merged['n_boxes'].fillna(0)
print("=== Promedio de bboxes por estudio según n_fracturas ===")
print(merged.groupby('n_fracturas')['n_boxes'].mean())
print()

# ---------------------------------------------------------------------------
# 5. Generación de gráficos (guardados en IMG_DIR)
# ---------------------------------------------------------------------------

# 1. Distribución de patient_overall
fig, ax = plt.subplots(figsize=(5, 4))
counts = train['patient_overall'].value_counts().sort_index()
ax.bar(['Sin fractura (0)', 'Con fractura (1)'], counts.values, color=['#4C72B0', '#C44E52'])
for i, v in enumerate(counts.values):
    ax.text(i, v + 15, f"{v}\n({v/len(train)*100:.1f}%)", ha='center')
ax.set_title('Distribución de patient_overall')
ax.set_ylabel('N° de pacientes')
plt.tight_layout()
plt.savefig(f'{IMG_DIR}/01_patient_overall.png')
plt.close()

# 2. Frecuencia de fractura por vértebra
fig, ax = plt.subplots(figsize=(7, 4))
pct = train[VERT_COLS].mean() * 100
bars = ax.bar(VERT_COLS, pct.values, color='#55A868')
ax.set_title('Porcentaje de fractura por vértebra cervical')
ax.set_ylabel('% de pacientes con fractura')
for b, v in zip(bars, pct.values):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.3, f"{v:.1f}%", ha='center')
plt.tight_layout()
plt.savefig(f'{IMG_DIR}/02_frac_por_vertebra.png')
plt.close()

# 3. Histograma del número de fracturas por paciente
fig, ax = plt.subplots(figsize=(6, 4))
vc = train['n_fracturas'].value_counts().sort_index()
ax.bar(vc.index.astype(str), vc.values, color='#8172B2')
ax.set_title('Número de vértebras fracturadas por paciente')
ax.set_xlabel('N° de vértebras fracturadas')
ax.set_ylabel('N° de pacientes')
for i, v in enumerate(vc.values):
    ax.text(i, v + 10, str(v), ha='center')
plt.tight_layout()
plt.savefig(f'{IMG_DIR}/03_n_fracturas.png')
plt.close()

# 4. Heatmap de correlación entre vértebras
fig, ax = plt.subplots(figsize=(6, 5))
corr = train[VERT_COLS].corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax, vmin=-0.4, vmax=0.4)
ax.set_title('Correlación entre fracturas de vértebras (co-ocurrencia)')
plt.tight_layout()
plt.savefig(f'{IMG_DIR}/04_corr_vertebras.png')
plt.close()

# 5. Histograma de cajas por estudio
bpc = bbox.groupby('StudyInstanceUID').size()
fig, ax = plt.subplots(figsize=(6, 4))
ax.hist(bpc.values, bins=25, color='#CCB974', edgecolor='black')
ax.set_title('N° de bounding boxes (slices anotados) por estudio\n(solo 235 estudios con bbox)')
ax.set_xlabel('N° de cajas por estudio')
ax.set_ylabel('Frecuencia')
plt.tight_layout()
plt.savefig(f'{IMG_DIR}/05_boxes_por_estudio.png')
plt.close()

# 6. Relación n_fracturas vs promedio de bboxes
agg = merged.groupby('n_fracturas')['n_boxes'].mean()
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(agg.index, agg.values, marker='o', color='#C44E52')
ax.set_title('Promedio de bounding boxes según N° de vértebras fracturadas')
ax.set_xlabel('N° de vértebras fracturadas (patient)')
ax.set_ylabel('Promedio de bboxes por estudio')
plt.tight_layout()
plt.savefig(f'{IMG_DIR}/06_boxes_vs_fracturas.png')
plt.close()

# 7. Distribución de slice_number
fig, ax = plt.subplots(figsize=(6, 4))
ax.hist(bbox['slice_number'], bins=40, color='#64B5CD', edgecolor='black')
ax.set_title('Distribución del número de slice anotado (bbox)')
ax.set_xlabel('slice_number')
ax.set_ylabel('Frecuencia')
plt.tight_layout()
plt.savefig(f'{IMG_DIR}/07_slice_number.png')
plt.close()

# 8. Distribución de width y height
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].hist(bbox['width'], bins=30, color='#DD8452', edgecolor='black')
axes[0].set_title('Ancho (width) de bounding boxes')
axes[1].hist(bbox['height'], bins=30, color='#937860', edgecolor='black')
axes[1].set_title('Alto (height) de bounding boxes')
plt.tight_layout()
plt.savefig(f'{IMG_DIR}/08_width_height.png')
plt.close()

# 9. Presencia de bbox vs patient_overall (barras apiladas)
train['tiene_bbox_label'] = train['tiene_bbox'].map({True: 'Con bbox', False: 'Sin bbox'})
ct = pd.crosstab(train['tiene_bbox_label'], train['patient_overall'])
fig, ax = plt.subplots(figsize=(6, 4))
ct.plot(kind='bar', stacked=True, ax=ax, color=['#4C72B0', '#C44E52'])
ax.set_title('Presencia de bounding boxes vs diagnóstico general')
ax.set_ylabel('N° de estudios')
ax.legend(title='patient_overall', labels=['Sin fractura', 'Con fractura'])
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(f'{IMG_DIR}/09_bbox_vs_overall.png')
plt.close()

print("Gráficos generados en:", IMG_DIR, "->", sorted(os.listdir(IMG_DIR)))