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

**Flujo:** Hugging Face → DataFrame → preprocesamiento → regresión lineal + evaluación → clasificación + evaluación → resumen automático `.txt`. GitHub Actions ejecuta todo el flujo en cada `push`.

> 📄 **Resumen de los dos modelos:** [`src/actividades/EA2/salidas/resumen_modelos.txt`](src/actividades/EA2/salidas/resumen_modelos.txt), generado automáticamente por la última ejecución. También se puede descargar como artefacto `resumen-modelos-ea2` desde la ejecución en GitHub Actions.

- Código: [`src/actividades/EA2/main.py`](src/actividades/EA2/main.py) (orquesta el flujo), [`src/clases_po/preprocesador.py`](src/clases_po/preprocesador.py), [`src/clases_po/regresion_monto.py`](src/clases_po/regresion_monto.py), [`src/clases_po/clasificacion_fraude.py`](src/clases_po/clasificacion_fraude.py) y [`src/clases_po/resumen_modelos.py`](src/clases_po/resumen_modelos.py).
- Salidas: `src/actividades/EA2/salidas/`.
- Workflow: [`.github/workflows/ea2.yml`](.github/workflows/ea2.yml).

| Evidencia | Archivo generado |
|---|---|
| 0. DataFrame construido desde Hugging Face | `salidas/00_dataframe_inicial.txt` |
| 1. Preprocesamiento (antes, operaciones, después) | `salidas/01_preprocesamiento.txt` |
| 2. Regresión lineal (configuración, métricas, coeficientes, interpretación) | `salidas/02_regresion.txt` |
| 3. Clasificación (configuración, matriz de confusión, métricas, umbrales, errores) | `salidas/03_clasificacion.txt` |
| 4. **Resumen de los dos modelos** | `salidas/resumen_modelos.txt` |
| 5. Workflow, artefactos y commits del bot | [`.github/workflows/ea2.yml`](.github/workflows/ea2.yml) y pestaña **Actions** |

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

**Dos ajustes después de la primera ejecución real:**

- **Categorías raras agrupadas en "otros".** En la primera ejecución, el estado de Delaware (DE) salió con un efecto de +1002 %. El motivo es que en el dataset casi no tiene transacciones, y las pocas que tiene son compras grandes, así que el modelo memorizaba esos casos. Por eso el One-hot ahora agrupa en "otros" toda categoría con menos de 100 casos en entrenamiento (`min_frequency=100`). El reporte muestra cuáles se agruparon.
- **Importancia por permutación.** Para decidir qué variable pesa más ya no miramos el coeficiente más grande, porque lo puede inflar una categoría rara. En cambio, medimos cuánto baja el R² en prueba cuando se mezclan al azar los valores de cada variable.

También vale la pena aclarar que el R² en dólares puede salir negativo aunque el MAE mejore. Al devolver el logaritmo a dólares, el modelo estima el monto *típico* y no el promedio, y el R² en dólares lo dominan las pocas compras gigantes. Por eso, para leer el error real usamos el MAE en dólares.

Validamos el código con datos simulados en los que pusimos efectos conocidos (+200 % y −50 %), y el modelo los recuperó casi exactos (+200,6 % y −49,5 %).

Los resultados y la interpretación de cada ejecución se calculan solos y quedan en `02_regresion.txt`. **Limitaciones:** el dataset no dice qué se compró, que es lo que más define el monto. Además, el modelo lineal solo suma efectos y no capta combinaciones, y con el logaritmo tiende a quedarse corto en las compras grandes.

### 3. Clasificación: ¿la transacción es fraude?

**Qué clasificamos:** cada transacción en **legítima (0)** o **fraude (1)**, usando la columna `is_fraud` del dataset. Las clases están muy desbalanceadas: solo cerca del 0,58 % de las transacciones son fraude.

| | |
|---|---|
| Predictoras | Las mismas de la regresión más `log_monto`. Aquí el monto sí entra, porque al momento de la compra ya se conoce y es una señal clave de fraude. |
| Excluida | `es_fraude`, porque es la respuesta. |
| Algoritmo | `LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)`. El resto de parámetros por defecto (`penalty="l2"`, `C=1.0`, `solver="lbfgs"`). Umbral de decisión: 0,5. |
| Configuración | `Pipeline` = transformador del componente 1 + modelo. Usamos la misma división 80/20 estratificada, con semilla 42. |
| Línea base | `DummyClassifier(strategy="most_frequent")`: siempre dice "legítima". |

**¿Por qué regresión logística?** Por tres razones:

- **Es interpretable.** Cada coeficiente se convierte en una razón de probabilidades (e^coeficiente). Si es mayor que 1, esa característica sube el riesgo de fraude; si es menor que 1, lo baja.
- **Entrega probabilidades.** Así podemos analizar qué pasa al mover el umbral de decisión.
- **`class_weight="balanced"` compensa el desbalance.** Sin este ajuste, el modelo aprendería a decir siempre "legítima", porque así acierta el 99,4 %. Con el ajuste, cada fraude pesa como muchas transacciones legítimas al entrenar.

**¿Por qué estas métricas y no solo la exactitud?** Porque la línea base, que nunca detecta un fraude, ya tiene **99,4 % de exactitud**. Por eso nos fijamos en otras:

- **Recall de fraude:** de los fraudes reales, cuántos detecta. Es la principal, porque un fraude perdido es dinero perdido.
- **Precisión de fraude:** de las alertas que da el modelo, cuántas son fraude de verdad. Mide el costo de las falsas alarmas.
- **F1:** el equilibrio entre recall y precisión.
- **PR-AUC:** qué tan bien ordena las transacciones por riesgo, sin depender del umbral. Un modelo al azar obtiene el mismo porcentaje de fraudes que hay en los datos.

El reporte también incluye:

- La **matriz de confusión** del modelo y de la línea base.
- Una tabla del **efecto del umbral** (0,5 a 0,99): al subir el umbral hay menos falsas alarmas, pero se pierden más fraudes.
- La **importancia de cada variable**, por permutación con PR-AUC.
- Un **análisis de errores**: cuántos fraudes se pierden, de qué monto y en qué categorías.

Validamos el código con datos simulados en los que el fraude ocurría de noche, en `shopping_net` y con montos altos. El modelo identificó justo esas variables como las más importantes.

**Limitaciones:**

- El balanceo sube el recall, pero genera muchas falsas alarmas.
- El modelo es lineal, así que no capta combinaciones como "compra grande **y** de madrugada".
- No usa el historial de cada cliente.
- El dataset es simulado.

### 4. Resumen automático de los dos modelos

La clase `ResumenModelos` arma `resumen_modelos.txt` a partir de los objetos ya ejecutados, sin texto escrito a mano. Incluye:

- La **fecha y hora de la ejecución**. Si corre en GitHub Actions, también el enlace a esa ejecución y el commit.
- La **identificación del dataset** y un resumen del **preprocesamiento**.
- Para cada modelo: el objetivo, la variable objetivo, las predictoras, el algoritmo, sus parámetros, la configuración de entrenamiento y prueba, las métricas comparadas con la línea base y una interpretación breve.
- Las **conclusiones y limitaciones**, calculadas con los resultados de esa ejecución.

### 5. Ejecución automática y control de versiones

El workflow [`ea2.yml`](.github/workflows/ea2.yml) se activa solo con cada `push` a `main` que cambie archivos de la EA2, las clases compartidas o las dependencias. También tiene un botón manual (`workflow_dispatch`), pero ese no es el único disparador. El workflow hace lo siguiente:

1. Instala las dependencias.
2. Ejecuta `python main.py`. Ese script descarga el dataset desde Hugging Face, construye el DataFrame, lo preprocesa, entrena y evalúa los dos modelos, y genera el resumen.
3. Publica dos artefactos descargables:
   - **`resumen-modelos-ea2`**: el resumen de los dos modelos.
   - **`evidencias-ea2`**: todos los `.txt`.
4. Guarda una copia de las salidas en el repositorio con un commit de `github-actions[bot]`.

**Cómo descargar el resumen:** pestaña **Actions** → "EA2 - Regresión y clasificación" → la última ejecución en verde → sección **Artifacts** → `resumen-modelos-ea2`.

## Checklist de entrega (EA2)

- [x] Dataset exclusivo del grupo, con identificador y enlace documentados.
- [x] DataFrame construido automáticamente desde Hugging Face.
- [x] Preprocesamiento justificado, con evidencias antes y después.
- [x] Regresión lineal con variables, parámetros, evaluación e interpretación.
- [x] Clasificación con justificación del algoritmo, variables, parámetros, evaluación e interpretación.
- [x] Resumen automático en un `.txt` con los resultados de ambos modelos.
- [x] Ejecución automática en GitHub Actions, con el resumen disponible como artefacto.
- [x] Control de versiones e instrucciones de ejecución.
- [ ] Acceso para el docente (si el repositorio es privado) y enlaces subidos a Canvas.

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
