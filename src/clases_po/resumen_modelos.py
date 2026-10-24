"""Resumen automático de los dos modelos en un .txt (EA2, componente 4).

Reúne en un solo archivo lo que pide la actividad: identificación del dataset,
preprocesamiento, configuración y métricas de cada modelo, interpretación,
fecha de ejecución y conclusiones. Todo sale de los objetos ya ejecutados;
nada se escribe a mano.
"""
from __future__ import annotations

import os
import platform
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import sklearn

from .clasificacion_fraude import ClasificadorFraude
from .preprocesador import PreprocesadorTransacciones
from .regresion_monto import RegresionMonto


class ResumenModelos:
    """Construye y guarda ``resumen_modelos.txt``.

    Args:
        dataset: identificador del dataset en Hugging Face.
        url_dataset: enlace del dataset.
        split: split utilizado.
        preprocesador: instancia ya ejecutada.
        regresion: instancia ya ejecutada.
        clasificador: instancia ya ejecutada.
    """

    def __init__(
        self,
        dataset: str,
        url_dataset: str,
        split: str,
        preprocesador: PreprocesadorTransacciones,
        regresion: RegresionMonto,
        clasificador: ClasificadorFraude,
    ) -> None:
        if regresion.metricas is None or clasificador.metricas is None:
            raise ValueError("Ejecuta los dos modelos antes de generar el resumen.")
        if preprocesador.estado_final is None:
            raise ValueError("Ejecuta el preprocesamiento antes de generar el resumen.")
        self.dataset = dataset
        self.url_dataset = url_dataset
        self.split = split
        self.pre = preprocesador
        self.reg = regresion
        self.clf = clasificador

    @staticmethod
    def _ejecucion() -> list[str]:
        ahora = datetime.now(timezone.utc)
        bogota = ahora.astimezone(ZoneInfo("America/Bogota"))
        lineas = [
            f"- Fecha de ejecución: {bogota:%Y-%m-%d %H:%M:%S} (hora de Bogotá) / "
            f"{ahora:%Y-%m-%d %H:%M:%S} UTC",
            f"- Entorno: Python {platform.python_version()}, scikit-learn {sklearn.__version__}",
        ]
        if os.getenv("GITHUB_ACTIONS") == "true":
            url = (f"{os.getenv('GITHUB_SERVER_URL')}/{os.getenv('GITHUB_REPOSITORY')}"
                   f"/actions/runs/{os.getenv('GITHUB_RUN_ID')}")
            lineas += [
                "- Ejecutado por: GitHub Actions",
                f"- Ejecución: {url}",
                f"- Commit: {os.getenv('GITHUB_SHA', '')[:7]}",
            ]
        else:
            lineas.append("- Ejecutado por: equipo local (no GitHub Actions)")
        return lineas

    def _preprocesamiento(self) -> list[str]:
        ini, fin = self.pre.estado_inicial, self.pre.estado_final
        lineas = [
            f"- Antes: {ini['filas']:,} filas, {ini['columnas']} columnas, {ini['nulos']:,} nulos.",
            f"- Después: {fin['filas']:,} filas, {fin['columnas']} columnas, {fin['nulos']:,} nulos.",
            "- Operaciones, en orden:",
        ]
        lineas += [f"  {i}. {p['operacion']}: {p['efecto']}"
                   for i, p in enumerate(self.pre.registro, start=1)]
        lineas += [
            "  9. Escalado (StandardScaler) y one-hot (OneHotEncoder con drop='first' y "
            "categorías raras agrupadas en 'otros') dentro del Pipeline de cada modelo, "
            "ajustados solo con entrenamiento.",
            "- Detalle completo: salidas/01_preprocesamiento.txt",
        ]
        return lineas

    def _regresion(self) -> list[str]:
        m = self.reg.metricas
        return [
            "- Objetivo: estimar el monto de una compra a partir del tipo de comercio, el "
            "momento, el perfil del cliente y la distancia al comercio.",
            f"- Variable objetivo: {self.reg.objetivo} = log(1 + {self.reg.columna_monto})",
            f"- Predictoras: {', '.join(self.reg.predictoras)}",
            "- Algoritmo: LinearRegression (scikit-learn), parámetros por defecto "
            "(fit_intercept=True).",
            f"- Entrenamiento / prueba: {self.reg.filas_train:,} / {self.reg.filas_test:,} filas "
            f"({int(self.pre.test_size * 100)} % prueba, semilla {self.pre.random_state}).",
            "- Línea base: DummyRegressor(strategy='mean').",
            "- Métricas en prueba:",
            f"    R2 (log)  = {m['modelo_log']['r2']:.4f}   (línea base {m['base_log']['r2']:.4f})",
            f"    MAE (log) = {m['modelo_log']['mae']:.4f}   (línea base {m['base_log']['mae']:.4f})",
            f"    RMSE (log)= {m['modelo_log']['rmse']:.4f}   (línea base {m['base_log']['rmse']:.4f})",
            f"    MAE (USD) = ${m['modelo_usd']['mae']:,.2f}  (línea base ${m['base_usd']['mae']:,.2f})",
            f"    RMSE (USD)= ${m['modelo_usd']['rmse']:,.2f} (línea base ${m['base_usd']['rmse']:,.2f})",
            f"- Intercepto: {self.reg.intercepto:.4f} (log), unos ${float(np.expm1(self.reg.intercepto)):,.2f}.",
            "- Interpretación:",
            *[f"  {c}" for c in self.reg._conclusiones()],
            "- Detalle completo (coeficientes de todas las variables): salidas/02_regresion.txt",
        ]

    def _clasificacion(self) -> list[str]:
        m, b = self.clf.metricas["modelo"], self.clf.metricas["base"]
        fr = m["por_clase"].loc["fraude"]
        mat = m["matriz"]
        params = ", ".join(f"{k}={v!r}" for k, v in self.clf.parametros_modelo.items())
        return [
            "- Objetivo: identificar si una transacción es fraude.",
            f"- Clases: 0 = legítima, 1 = fraude (columna '{self.clf.objetivo}').",
            f"- Predictoras: {', '.join(self.clf.predictoras)}",
            f"- Algoritmo: LogisticRegression ({params}; resto por defecto). "
            f"Umbral de decisión: {self.clf.umbral}.",
            "- Por qué: es interpretable, entrega probabilidades y compensa el desbalance de clases.",
            f"- Entrenamiento / prueba: misma división que la regresión "
            f"({self.reg.filas_train:,} / {self.reg.filas_test:,} filas, estratificada).",
            "- Línea base: DummyClassifier(strategy='most_frequent').",
            "- Métricas en prueba:",
            f"    Exactitud        = {m['exactitud']:.4f}  (línea base {b['exactitud']:.4f})",
            f"    Recall fraude    = {fr['recall']:.4f}  (línea base 0.0000)",
            f"    Precisión fraude = {fr['precision']:.4f}  (línea base 0.0000)",
            f"    F1 fraude        = {fr['f1']:.4f}  (línea base 0.0000)",
            f"    PR-AUC           = {m['pr_auc']:.4f}  (azar {b['pr_auc']:.4f})",
            f"    ROC-AUC          = {m['roc_auc']:.4f}  (azar 0.5000)",
            "- Matriz de confusión:",
            *[f"    {linea}" for linea in mat.to_string().splitlines()],
            "- Interpretación:",
            *[f"  {c}" for c in self.clf._conclusiones()],
            "- Detalle completo (umbrales, coeficientes, errores): salidas/03_clasificacion.txt",
        ]

    def _conclusiones(self) -> list[str]:
        r2 = self.reg.metricas["modelo_log"]["r2"]
        fr = self.clf.metricas["modelo"]["por_clase"].loc["fraude"]
        umb = self.clf.tabla_umbrales
        mejor = umb.loc[umb["f1_fraude"].idxmax()]
        pr, azar = self.clf.metricas["modelo"]["pr_auc"], self.clf.metricas["base"]["pr_auc"]
        top_reg = list(self.reg.importancia.index[:3])
        top_clf = list(self.clf.importancia.index[:3])
        comunes = [v for v in top_clf if v in top_reg]
        filas_ini = self.pre.estado_inicial["filas"]
        filas_fin = self.pre.estado_final["filas"]
        return [
            f"- El mismo dataset sirvió para los dos problemas. Predecir el monto exacto es "
            f"difícil (R2 = {r2:.2f}): la variable '{top_reg[0]}' explica la mayor parte de "
            "lo que el modelo logra, y falta saber qué se compró.",
            f"- El clasificador da un resultado más útil: encuentra el "
            f"{fr['recall'] * 100:.0f} % de los fraudes y ordena las transacciones por riesgo "
            f"{pr / azar:.0f} veces mejor que el azar (PR-AUC). Su costo son las falsas "
            f"alarmas (precisión {fr['precision'] * 100:.1f} %).",
            f"- El umbral cambia mucho el resultado. En esta ejecución, el mejor F1 de fraude "
            f"({mejor['f1_fraude']:.3f}) aparece con umbral {mejor['umbral']}: detecta "
            f"{int(mejor['fraudes_detectados'])} fraudes con {int(mejor['falsas_alarmas'])} "
            "falsas alarmas. Elegir el umbral es una decisión de negocio (cuánto cuesta un "
            "fraude perdido frente a una alerta falsa). Esta tabla se calculó con el conjunto "
            "de prueba, así que es un análisis y no un umbral ya validado.",
            f"- Variables más importantes (por permutación): regresión = {', '.join(top_reg)}; "
            f"clasificación = {', '.join(top_clf)}."
            + (f" Aparecen en ambos: {', '.join(comunes)}." if comunes else ""),
            "",
            "LIMITACIONES",
            (f"- Se usó una muestra de {filas_fin:,} de las {filas_ini:,} filas del dataset "
             "(estratificada), no el total." if filas_fin < filas_ini
             else "- Se usó el dataset completo."),
            "- Ambos modelos son lineales: no captan combinaciones entre variables.",
            "- Falta información clave: qué se compró (para el monto) y el historial de "
            "cada cliente (para el fraude).",
            "- El dataset es simulado, así que los patrones pueden ser más limpios que los reales.",
            "- Los coeficientes muestran asociaciones, no causas.",
        ]

    def generar(self) -> str:
        secciones = [
            ("RESUMEN DE MODELOS - EA2 MACHINE LEARNING", None),
            ("EJECUCIÓN", self._ejecucion()),
            ("DATASET", [
                f"- Identificador: {self.dataset}",
                f"- Enlace: {self.url_dataset}",
                f"- Split: {self.split}",
                "- Contenido: transacciones de tarjeta de crédito simuladas, con datos del "
                "comercio, del cliente, ubicación y la marca de fraude.",
            ]),
            ("PREPROCESAMIENTO", self._preprocesamiento()),
            ("MODELO 1 - REGRESIÓN LINEAL", self._regresion()),
            ("MODELO 2 - CLASIFICACIÓN (REGRESIÓN LOGÍSTICA)", self._clasificacion()),
            ("CONCLUSIONES", self._conclusiones()),
        ]
        lineas: list[str] = []
        for titulo, contenido in secciones:
            if contenido is None:
                lineas += ["=" * 70, titulo, "=" * 70]
                continue
            lineas += ["", titulo, "-" * 70, *contenido]
        return "\n".join(lineas) + "\n"

    def guardar(self, ruta: str | Path) -> Path:
        ruta = Path(ruta)
        ruta.parent.mkdir(parents=True, exist_ok=True)
        ruta.write_text(self.generar(), encoding="utf-8")
        print(f"[resumen] guardado en {ruta}")
        return ruta
