"""Componente 4: resumen automático de toda la ejecución en un .txt.

Todo lo que aparece aquí sale de los objetos que se acaban de ejecutar
(preprocesador, modelo y evaluación). No escribimos ningún número a mano,
así que el archivo siempre corresponde a la última corrida.
"""
from __future__ import annotations

import os
import platform
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import sklearn

import configuracion as cfg
import pregunta as pn
from componentes.evaluacion import EvaluacionEA3
from componentes.modelo import BUSQUEDA, ModeloFraudeRF
from componentes.preprocesamiento import PreprocesadorEA3


class ResumenEA3:
    def __init__(self, pre: PreprocesadorEA3, modelo: ModeloFraudeRF, evaluacion: EvaluacionEA3) -> None:
        self.pre = pre
        self.modelo = modelo
        self.ev = evaluacion

    @staticmethod
    def _ejecucion() -> list[str]:
        ahora = datetime.now(timezone.utc)
        lineas = [
            f"- Fecha de ejecución: {ahora.astimezone(ZoneInfo('America/Bogota')):%Y-%m-%d %H:%M:%S} "
            f"(hora de Bogotá) / {ahora:%Y-%m-%d %H:%M:%S} UTC",
            f"- Entorno: Python {platform.python_version()}, scikit-learn {sklearn.__version__}, "
            f"pandas {pd.__version__}",
        ]
        if os.getenv("GITHUB_ACTIONS") == "true":
            lineas += [
                "- Ejecutado por: GitHub Actions",
                f"- Ejecución: {os.getenv('GITHUB_SERVER_URL')}/{os.getenv('GITHUB_REPOSITORY')}"
                f"/actions/runs/{os.getenv('GITHUB_RUN_ID')}",
                f"- Commit: {os.getenv('GITHUB_SHA', '')[:7]}",
            ]
        else:
            lineas.append("- Ejecutado por: equipo local (no GitHub Actions)")
        return lineas

    def _respuesta(self) -> list[str]:
        """Respuesta corta a la pregunta, calculada con esta ejecución."""
        b = self.ev.metricas["bosque (umbral elegido)"]
        cap = self.ev.capacidad.set_index("capacidad_dia")
        top = list(self.ev.importancia.sort_values(ascending=False).index[:4])
        lineas = [
            "Las transacciones con más riesgo son las que el modelo pone arriba de la lista. "
            f"Lo que más pesa para ubicarlas ahí es: {', '.join(top)}.",
            f"- Con el umbral recomendado ({self.modelo.umbral:.3f}) el equipo revisaría unas "
            f"{b['alertas_por_dia']:.1f} alertas al día y frenaría el {b['recall'] * 100:.1f} % de los "
            f"fraudes, que son el {b['dinero_fraude_detectado_%']:.1f} % del dinero en fraude. "
            f"{b['precision'] * 100:.0f} de cada 100 alertas son fraude de verdad.",
            "- Según cuántas alertas pueda revisar el equipo al día:",
        ]
        for k, f in cap.iterrows():
            lineas.append(
                f"    {int(k):>3} alertas/día -> {f['fraudes_detectados_%']:5.1f} % de los fraudes, "
                f"{f['dinero_detectado_%']:5.1f} % del dinero, {f['alertas_que_son_fraude_%']:5.1f} % de "
                f"las alertas son fraude (umbral {f['umbral']:.3f})")
        return lineas

    def generar(self) -> str:
        m = self.modelo
        b = self.ev.metricas
        tabla = pd.DataFrame(b).T.reindex(
            columns=["exactitud", "precision", "recall", "f1", "pr_auc", "roc_auc",
                     "alertas_por_dia", "dinero_fraude_detectado_%"]).astype(float).round(4)
        conjuntos = self.pre._tabla_conjuntos()
        secciones = [
            ("PREGUNTA DE NEGOCIO", [
                pn.PREGUNTA,
                f"- Quién la necesita: {pn.BENEFICIARIO}",
                f"- Qué decisión apoya: {pn.DECISION}",
            ]),
            ("EJECUCIÓN", self._ejecucion()),
            ("DATASET", [
                f"- Identificador: {pn.DATASET}",
                f"- Enlace: {pn.URL_DATASET}",
                f"- Split: {pn.SPLIT}",
                f"- Tamaño: {self.pre.antes['filas']:,} filas y {self.pre.antes['columnas']} columnas "
                f"({self.pre.antes['nulos']:,} nulos).",
                "- Contenido: transacciones de tarjeta de crédito simuladas, con fecha y hora, comercio, "
                "categoría, monto, datos y ubicación del cliente y del comercio, y la marca de fraude.",
            ]),
            ("PREPROCESAMIENTO", [
                f"- Después: {self.pre.despues['filas']:,} filas, {self.pre.despues['columnas']} columnas, "
                f"{self.pre.despues['nulos']:,} nulos.",
                *[f"  {i}. {p['operacion']}: {p['efecto']}" for i, p in enumerate(self.pre.registro, start=1)],
                "- Detalle: salidas/01_preprocesamiento.txt",
            ]),
            ("MODELO", [
                f"- Tipo de aprendizaje: {pn.TIPO_APRENDIZAJE}",
                "- Algoritmo: Random Forest (sklearn.ensemble.RandomForestClassifier).",
                f"- Variable objetivo: {m.objetivo} (0 = legítima, 1 = fraude).",
                f"- Predictoras numéricas (sin escalar): {', '.join(m.numericas)}",
                f"- Predictoras categóricas (one-hot): {', '.join(m.categoricas)}",
                "- Parámetros finales:",
                *[f"    {k} = {v!r}" for k, v in m.parametros_finales().items()],
                f"- Cómo se eligieron: {len(m.busqueda)} combinaciones de "
                + ", ".join(f"{k} {v}" for k, v in BUSQUEDA.items())
                + ", medidas con PR-AUC en validación; entre las casi empatadas, la de menor brecha "
                f"entre entrenamiento y validación. Número de árboles por curva (10 a 400).",
                f"- Umbral de alerta: {m.umbral:.4f}, el de mejor F1 en validación "
                f"(F1 {m.info_umbral['f1']:.3f}).",
                "- Detalle: salidas/02_modelo.txt",
            ]),
            ("CONFIGURACIÓN DEL EXPERIMENTO", [
                f"- Semilla: {cfg.SEMILLA}",
                f"- División temporal: entrenamiento < {cfg.INICIO_VALIDACION}; validación "
                f"{cfg.INICIO_VALIDACION} a {cfg.INICIO_PRUEBA}; prueba >= {cfg.INICIO_PRUEBA}.",
                f"- Entrenamiento con todos los fraudes y el {cfg.FRACCION_LEGITIMAS_TRAIN:.0%} de las legítimas.",
                *[f"    {linea}" for linea in conjuntos.to_string().splitlines()],
            ]),
            ("MÉTRICAS EN PRUEBA", [
                *tabla.to_string().splitlines(),
                "",
                "Matriz de confusión (umbral elegido):",
                *[f"    {linea}" for linea in self.ev.matriz.to_string().splitlines()],
                "- Detalle: salidas/03_evaluacion.txt",
            ]),
            ("RESPUESTA A LA PREGUNTA", self._respuesta()),
            ("INTERPRETACIÓN", self.ev.conclusiones()),
            ("LIMITACIONES", [
                "- El dataset es simulado; en un banco real el fraude es más variado y estas cifras "
                "seguramente serían más bajas.",
                "- El fraude cambia con el tiempo: el modelo habría que reentrenarlo y vigilarlo cada mes.",
                "- El umbral se eligió con F1; si el banco conoce el costo de cada error, debe ajustarlo "
                "con la tabla de capacidad.",
                "- Los fraudes pequeños, que se parecen a compras normales, son los que más se escapan.",
            ]),
        ]
        lineas = ["=" * 70, "RESUMEN EA3 - RANDOM FOREST PARA PRIORIZAR ALERTAS DE FRAUDE", "=" * 70]
        for titulo, contenido in secciones:
            lineas += ["", titulo, "-" * 70, *contenido]
        return "\n".join(lineas) + "\n"

    def guardar(self, ruta: Path) -> Path:
        ruta.parent.mkdir(parents=True, exist_ok=True)
        ruta.write_text(self.generar(), encoding="utf-8")
        print(f"[4] Resumen guardado en {ruta}")
        return ruta
