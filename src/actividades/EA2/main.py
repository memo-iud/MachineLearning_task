"""EA2 - Modelos de predicción y clasificación.

Flujo completo que ejecuta GitHub Actions (``.github/workflows/ea2.yml``):

    Hugging Face -> DataFrame -> preprocesamiento
                             -> regresión lineal + evaluación
                             -> clasificación + evaluación
                             -> resumen automático .txt

Ejecución local (desde esta carpeta):

    python main.py
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# Permite importar las clases compartidas de src/clases_po.
CARPETA_EA2 = Path(__file__).resolve().parent
RAIZ = CARPETA_EA2.parents[2]
sys.path.insert(0, str(RAIZ / "src"))

from clases_po import (  # noqa: E402
    CATEGORICAS,
    NUMERICAS,
    ClasificadorFraude,
    HFDatasetLoader,
    PreprocesadorTransacciones,
    RegresionMonto,
    Reportes,
    ResumenModelos,
)

# ---------------------------------------------------------------------------
# Configuración del experimento
# ---------------------------------------------------------------------------
DATASET = "pointe77/credit-card-transaction"
URL_DATASET = f"https://huggingface.co/datasets/{DATASET}"
SPLIT = "train"
OBJETIVO_CLASIFICACION = "is_fraud"
TAMANO_MUESTRA = 300_000
TEST_SIZE = 0.2
SEMILLA = 42
SALIDAS = CARPETA_EA2 / "salidas"
MODELOS = CARPETA_EA2 / "modelos"


def construir_dataframe() -> pd.DataFrame:
    """Componente 0: descarga el dataset y lo devuelve como DataFrame."""
    loader = HFDatasetLoader(dataset_name=DATASET, split=SPLIT)
    df = loader.load()
    print(f"[0] Dataset {DATASET} (split={SPLIT}) cargado: {df.shape}")
    return df


def evidencia_dataframe(df: pd.DataFrame) -> str:
    """Texto con tamaño, columnas, tipos, muestra y balance de clases."""
    lineas = [
        "EA2 - COMPONENTE 0: DATAFRAME CONSTRUIDO DESDE HUGGING FACE",
        "=" * 70,
        f"Fecha de ejecución: {datetime.now().isoformat(timespec='seconds')}",
        f"Dataset: {DATASET}",
        f"Enlace: {URL_DATASET}",
        f"Split: {SPLIT}",
        "",
        Reportes().generar_reporte(df),
        "Muestra del contenido (5 filas):",
        df.head().to_string(),
        "",
    ]
    if OBJETIVO_CLASIFICACION in df.columns:
        conteo = df[OBJETIVO_CLASIFICACION].value_counts().sort_index()
        porcentaje = (conteo / conteo.sum() * 100).round(3)
        lineas += [
            f"Distribución de '{OBJETIVO_CLASIFICACION}' (0 = legítima, 1 = fraude):",
            pd.DataFrame({"transacciones": conteo, "%": porcentaje}).to_string(),
            "",
        ]
    return "\n".join(lineas)


def main() -> None:
    SALIDAS.mkdir(parents=True, exist_ok=True)

    df = construir_dataframe()
    texto = evidencia_dataframe(df)
    print(texto)
    ruta = SALIDAS / "00_dataframe_inicial.txt"
    ruta.write_text(texto, encoding="utf-8")
    print(f"[0] Evidencia guardada en {ruta}")

    # Componente 1: preprocesamiento.
    preprocesador = PreprocesadorTransacciones(
        tamano_muestra=TAMANO_MUESTRA, test_size=TEST_SIZE, random_state=SEMILLA
    )
    preprocesador.ejecutar(df)
    del df  # libera memoria: desde aquí se trabaja con la muestra
    evidencia = preprocesador.generar_evidencia()
    print(evidencia)
    ruta = SALIDAS / "01_preprocesamiento.txt"
    ruta.write_text(evidencia, encoding="utf-8")
    print(f"[1] Evidencia guardada en {ruta}")

    # Componente 2: regresión lineal del monto.
    regresion = RegresionMonto(
        numericas=NUMERICAS, categoricas=CATEGORICAS, directorio_modelos=MODELOS
    )
    regresion.ejecutar(preprocesador.df_train, preprocesador.df_test)
    evidencia = regresion.generar_evidencia()
    print(evidencia)
    ruta = SALIDAS / "02_regresion.txt"
    ruta.write_text(evidencia, encoding="utf-8")
    print(f"[2] Evidencia guardada en {ruta}")

    # Componente 3: clasificación de fraude (el monto sí entra como predictora).
    clasificador = ClasificadorFraude(
        numericas=NUMERICAS + ["log_monto"],
        categoricas=CATEGORICAS,
        random_state=SEMILLA,
        directorio_modelos=MODELOS,
    )
    clasificador.ejecutar(preprocesador.df_train, preprocesador.df_test)
    evidencia = clasificador.generar_evidencia()
    print(evidencia)
    ruta = SALIDAS / "03_clasificacion.txt"
    ruta.write_text(evidencia, encoding="utf-8")
    print(f"[3] Evidencia guardada en {ruta}")

    # Componente 4: resumen automático de los dos modelos.
    resumen = ResumenModelos(
        dataset=DATASET,
        url_dataset=URL_DATASET,
        split=SPLIT,
        preprocesador=preprocesador,
        regresion=regresion,
        clasificador=clasificador,
    )
    ruta = resumen.guardar(SALIDAS / "resumen_modelos.txt")
    print(ruta.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
