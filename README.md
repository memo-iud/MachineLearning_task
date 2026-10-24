# MachineLearning_task

Repositorio del curso de Machine Learning (IU Digital de Antioquia). Cada actividad evaluativa tiene su propia carpeta y su propio workflow de GitHub Actions, que la ejecuta y genera sus resultados de forma automática.

## Integrantes

- Jaider Morales Bautista
- Aleicer Vesga Rueda
- Guillerno Leon Loaiza Mesa

## Dataset

| | |
|---|---|
| Identificador | `pointe77/credit-card-transaction` |
| Enlace | https://huggingface.co/datasets/pointe77/credit-card-transaction |
| Split | `train` |
| Contenido | Transacciones de tarjeta de crédito simuladas: fecha y hora, comercio, categoría, monto, datos del cliente (género, ciudad, estado, profesión, fecha de nacimiento), ubicación del cliente y del comercio, y la marca `is_fraud` (1 = fraude). |

El dataset es exclusivo de nuestro grupo.

## Estructura

```
.github/workflows/
  ea1.yml                  Ejecuta la actividad 1
  ea2.yml                  Ejecuta la actividad 2
src/
  clases_po/               Clases compartidas por todas las actividades
    hf_loader.py           HFDatasetLoader: carga un dataset de Hugging Face como DataFrame
    reportes.py            Reportes: resume un DataFrame en un archivo .txt
  actividades/
    EA1/
      EA1.ipynb            Notebook que coordina carga -> transformación -> resumen
      reportes/            Resumen .txt generado por el workflow
    EA2/
      main.py              Flujo completo de la actividad 2
      salidas/             Archivos .txt generados por el workflow
requirements.txt
```

## Ejecución automática (GitHub Actions)

Cada vez que se hace `push` a `main` con cambios en una actividad, su workflow:

1. Instala las dependencias de `requirements.txt`.
2. Descarga el dataset desde Hugging Face y ejecuta el flujo de la actividad.
3. Publica los resultados como **artefacto descargable** de la ejecución.
4. Guarda una copia de los resultados en el repositorio con un commit automático de `github-actions[bot]` (su mensaje lleva `[skip ci]` para no volver a disparar el workflow).

Cada workflow se activa solo cuando cambian archivos de su actividad, de las clases compartidas o de las dependencias. También se pueden lanzar a mano desde la pestaña **Actions → (workflow) → Run workflow**.

> **Importante para el equipo:** como el bot sube commits con los resultados, antes de seguir trabajando hay que traer esos cambios con `git pull`. Si no, el siguiente `push` será rechazado.

Los resultados no se generan ni se suben a mano: los archivos `.txt` y las salidas del notebook siempre corresponden a la última ejecución en GitHub.

## Actividad 1 — Obtener y preparar datos desde Hugging Face

- Notebook: [`src/actividades/EA1/EA1.ipynb`](src/actividades/EA1/EA1.ipynb). Allí se explica cada transformación.
- Datos cargados (100 filas): `src/actividades/EA1/datos/pointe77/credit-card-transaction.csv`.
- Resumen generado: `src/actividades/EA1/reportes/pointe77/credit-card-transaction.txt`.
- Workflow: [`.github/workflows/ea1.yml`](.github/workflows/ea1.yml).

Transformaciones aplicadas: renombrar las columnas al español, eliminar el índice `Unnamed: 0`, rellenar los nulos de `codigo_postal_comerciante` con -1 y calcular la `edad` del cliente.

## Actividad 2 — Modelos de predicción y clasificación

> En construcción: componentes 0, 1 y 2 listos; faltan la clasificación (3) y el resumen final (4).

- Código: [`src/actividades/EA2/main.py`](src/actividades/EA2/main.py) (orquesta el flujo), [`src/clases_po/preprocesador.py`](src/clases_po/preprocesador.py) y [`src/clases_po/regresion_monto.py`](src/clases_po/regresion_monto.py).
- Salidas: `src/actividades/EA2/salidas/`.
- Workflow: [`.github/workflows/ea2.yml`](.github/workflows/ea2.yml).

| Evidencia | Archivo generado |
|---|---|
| 0. DataFrame construido desde Hugging Face | `salidas/00_dataframe_inicial.txt` |
| 1. Preprocesamiento (antes, operaciones, después) | `salidas/01_preprocesamiento.txt` |
| 2. Regresión lineal (configuración, métricas, coeficientes, interpretación) | `salidas/02_regresion.txt` |

### 0. Construcción del DataFrame

`main.py` descarga el split `train` completo con `HFDatasetLoader`, sin archivos manuales. El reporte `00_dataframe_inicial.txt` muestra el tamaño, las columnas, los tipos, una muestra y cuántas transacciones son fraude.

En la ejecución real el dataset tiene **1.296.675 transacciones y 24 columnas**. Solo **7.506 son fraude (0,579 %)**, así que las clases están muy desbalanceadas. Además, la única columna con nulos es `merch_zipcode`, con 195.973 (15 %).

Con este dataset planteamos dos problemas:

- **Regresión:** predecir el **monto** de una transacción (variable numérica continua).
- **Clasificación:** predecir si una transacción es **fraude o no** (`is_fraud`, variable categórica 0/1). Esta etiqueta ya viene en el dataset; no la construimos a partir de otra columna.

### 1. Preprocesamiento

Todas las operaciones están en la clase `PreprocesadorTransacciones`. Cada ejecución deja en `01_preprocesamiento.txt` el antes y el después, y el motivo y el efecto de cada paso.

| Paso | Qué hacemos | Por qué |
|---|---|---|
| Muestreo | Tomamos 300.000 filas estratificadas por fraude, con semilla 42. | El dataset completo hace muy lenta cada ejecución en GitHub. Al estratificar, la muestra conserva el mismo 0,579 % de fraudes. |
| Selección de variables | Eliminamos 12 columnas. | Unas son datos personales o identificadores (nombre, apellido, calle, número de tarjeta, número de transacción). Otras repiten información (`unix_time`, `zip`) o tienen cientos de valores distintos (comercio, ciudad, profesión), lo que no le sirve a un modelo lineal. También sale `merch_zipcode`, que tiene muchos nulos y repite la ubicación del comercio. |
| Renombrar | Pasamos las columnas al español. | Mantener el mismo idioma de la EA1. |
| Duplicados | Revisamos y eliminamos filas repetidas. | Una fila repetida pesaría doble al entrenar. |
| Tipos | Convertimos las fechas de texto a fecha. | Así se puede calcular con ellas. |
| Variables nuevas | `hora`, `dia_semana`, `edad`, `distancia_km`, `log_poblacion` y `log_monto`. | La hora y el día pueden mostrar patrones de fraude. La edad ahora se calcula exacta y no solo por año. La distancia resume en un dato las coordenadas del cliente y del comercio. El logaritmo reduce el efecto de unos pocos valores muy grandes en la población y el monto. |
| Entrenamiento / prueba | 80 % / 20 %, estratificado por fraude, con semilla 42. | Evaluar con datos que el modelo no vio. Los dos modelos usan la misma división. |
| Escalado y codificación | `StandardScaler` para las numéricas y One-hot (`drop="first"`) para `categoria`, `genero`, `estado`, `hora` y `dia_semana`. | Los modelos lineales necesitan números en escalas parecidas. Estos pasos **se ajustan solo con el conjunto de entrenamiento** y después se aplican al de prueba. El reporte lo demuestra: en entrenamiento las numéricas quedan con media 0 y desviación 1, y en prueba solo se acercan a esos valores. |

### 2. Regresión lineal: estimar el monto de una compra

**Pregunta:** ¿qué tanto podemos estimar el monto de una compra a partir del tipo de comercio, el momento de la compra, el perfil del cliente y la distancia al comercio?

| | |
|---|---|
| Variable objetivo | `log_monto` = log(1 + monto). El monto tiene pocas compras muy grandes y muchas pequeñas; con el logaritmo esas compras grandes no dominan la recta. |
| Predictoras | `categoria`, `hora`, `dia_semana`, `genero`, `estado` (categóricas) y `edad`, `distancia_km`, `log_poblacion` (numéricas). |
| Excluidas a propósito | `monto` y `log_monto`, porque son la respuesta, y `es_fraude`, porque al momento de la compra no se sabe si es fraude. |
| Algoritmo | `LinearRegression()` de scikit-learn con **parámetros por defecto** (`fit_intercept=True`). |
| Configuración | `Pipeline` = transformador del componente 1 + regresión. Usamos la misma división 80/20 del componente 1, con semilla 42. |
| Línea base | `DummyRegressor(strategy="mean")`: siempre predice el promedio. Sirve para medir cuánto aporta el modelo. |
| Métricas | MAE, RMSE y R², en escala logarítmica y también en dólares (las predicciones se devuelven a dólares con exp(x) − 1). |

**Cómo leemos los coeficientes.** Como el objetivo es un logaritmo, cada coeficiente se convierte en un cambio porcentual del monto con **efecto % = (e^coeficiente − 1) × 100**:

- En una **categoría** (por ejemplo, `categoria = shopping_net`), el efecto se compara con la categoría de referencia, que es la primera en orden alfabético y la que quita `drop="first"`.
- En una **numérica**, el efecto es el de aumentar una desviación estándar, porque las variables están escaladas. El reporte muestra cuánto vale esa desviación en unidades reales (años o km).
- El **intercepto** es el monto típico del caso de referencia con las numéricas en su promedio.

Validamos el código con datos simulados en los que pusimos efectos conocidos (+200 % y −50 %), y el modelo los recuperó casi exactos (+200,6 % y −49,5 %).

Los resultados y la interpretación de cada ejecución se calculan solos y quedan en `02_regresion.txt`. **Limitaciones:** el dataset no dice qué se compró, que es lo que más define el monto. Además, el modelo lineal solo suma efectos y no capta combinaciones, y con el logaritmo tiende a quedarse corto en las compras grandes.

## Ejecución local

```bash
python -m venv .venv
source .venv/bin/activate          # En Windows (PowerShell): .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt

# Actividad 1
cd src/actividades/EA1
jupyter nbconvert --to notebook --execute --inplace EA1.ipynb

# Actividad 2
cd ../EA2
python main.py
```

### Autenticación en Hugging Face (opcional)

El dataset es público, así que no hace falta token. Si en algún momento se necesita uno:

```bash
python -m pip install --upgrade huggingface_hub
hf auth login       # pegar el token de solo lectura
hf auth whoami      # verificar la sesión
```

Nunca escribas el token en el código ni lo subas a GitHub.
