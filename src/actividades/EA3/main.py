"""EA3 - Modelo de Machine Learning para responder una pregunta de negocio.

Este es el archivo que corre GitHub Actions (.github/workflows/ea3.yml).
El flujo completo va así:

    Pregunta de negocio
        -> Hugging Face -> DataFrame -> preprocesamiento
        -> Random Forest -> métricas y evaluación
        -> resumen automático .txt
        -> gráficas para el documento de hallazgos

Al final queda salidas/resumen_ea3.txt con todo lo de la ejecución. El documento
de hallazgos (HALLAZGOS.md) lo escribimos nosotros a partir de estos resultados.

Para correrlo en local, desde esta carpeta:

    python main.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Rutas: esta carpeta (para importar los componentes de la EA3) y src/
# (para reutilizar las clases que ya teníamos de la EA1 y la EA2).
CARPETA_EA3 = Path(__file__).resolve().parent
RAIZ = CARPETA_EA3.parents[2]
sys.path.insert(0, str(CARPETA_EA3))
sys.path.insert(0, str(RAIZ / "src"))

from clases_po import HFDatasetLoader  # noqa: E402

import pregunta as pn  # noqa: E402
from componentes import exploracion  # noqa: E402
from componentes.evaluacion import EvaluacionEA3  # noqa: E402
from componentes.modelo import ModeloFraudeRF  # noqa: E402
from componentes.resumen import ResumenEA3  # noqa: E402
from componentes.preprocesamiento import (  # noqa: E402
    CATEGORICAS,
    NUMERICAS,
    OBJETIVO,
    PreprocesadorEA3,
)

SALIDAS = CARPETA_EA3 / "salidas"
GRAFICAS = CARPETA_EA3 / "graficas"


def main() -> None:
    # Componente 0: traemos el dataset completo desde Hugging Face.
    # No hay archivos descargados a mano: todo sale de esta llamada.
    df = HFDatasetLoader(dataset_name=pn.DATASET, split=pn.SPLIT).load()
    print(f"[0] Dataset {pn.DATASET} (split={pn.SPLIT}) cargado: {df.shape}")
    exploracion.ejecutar(df, SALIDAS, GRAFICAS)

    # Componente 1: preprocesamiento (historial de cada tarjeta y división temporal).
    preprocesador = PreprocesadorEA3()
    preprocesador.ejecutar(df)
    del df  # ya no necesitamos el DataFrame crudo y así liberamos memoria
    evidencia = preprocesador.generar_evidencia()
    print(evidencia)
    (SALIDAS / "01_preprocesamiento.txt").write_text(evidencia, encoding="utf-8")
    preprocesador.graficar(GRAFICAS / "01_historial_tarjeta.png")

    # Componente 2: Random Forest. Elegimos parámetros y umbral con la validación.
    modelo = ModeloFraudeRF(NUMERICAS, CATEGORICAS, OBJETIVO)
    modelo.ejecutar(preprocesador.train, preprocesador.validacion)
    evidencia = modelo.generar_evidencia()
    print(evidencia)
    (SALIDAS / "02_modelo.txt").write_text(evidencia, encoding="utf-8")
    modelo.graficar_ajuste(GRAFICAS / "02_eleccion_parametros.png")
    modelo.graficar_arbol(GRAFICAS / "02_arbol_ejemplo.png")

    # Componente 3: evaluación final con los meses de prueba.
    evaluacion = EvaluacionEA3(modelo, preprocesador.validacion, preprocesador.prueba)
    evaluacion.ejecutar()
    evidencia = evaluacion.generar_evidencia()
    print(evidencia)
    (SALIDAS / "03_evaluacion.txt").write_text(evidencia, encoding="utf-8")
    evaluacion.graficar(GRAFICAS)

    # Componente 4: resumen automático de toda la ejecución.
    ruta = ResumenEA3(preprocesador, modelo, evaluacion).guardar(SALIDAS / "resumen_ea3.txt")
    print(ruta.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
