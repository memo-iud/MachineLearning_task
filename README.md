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

> En construcción.

- Código: [`src/actividades/EA2/main.py`](src/actividades/EA2/main.py).
- Salidas: `src/actividades/EA2/salidas/`.
- Workflow: [`.github/workflows/ea2.yml`](.github/workflows/ea2.yml).

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
