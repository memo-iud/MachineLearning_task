"""EA3 - Modelo de Machine Learning para responder una pregunta de negocio.

Este es el archivo que corre GitHub Actions (.github/workflows/ea3.yml).
El flujo completo va así:

    Pregunta de negocio
        -> Hugging Face -> DataFrame -> preprocesamiento
        -> Random Forest -> métricas y evaluación
        -> resumen automático .txt
        -> gráficas para el documento de hallazgos

Por ahora tiene el componente 0 (pregunta + DataFrame) y el 1 (preprocesamiento).
Los demás los vamos agregando por fases.

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
from componentes.preprocesamiento import PreprocesadorEA3  # noqa: E402

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

    # Próximas fases:
    # 2-3. Random Forest, validación cruzada, métricas y gráficas
    # 4. resumen_ea3.txt


if __name__ == "__main__":
    main()
