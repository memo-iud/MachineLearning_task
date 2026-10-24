"""Regresión lineal para estimar el monto de una transacción (EA2, componente 2).

Pregunta: ¿qué tanto podemos estimar el monto de una compra a partir del tipo
de comercio, el momento de la compra, el perfil del cliente y la distancia al
comercio?

- Objetivo: ``log_monto`` = log(1 + monto). El monto tiene unas pocas compras
  muy grandes; con el logaritmo la relación con las variables es más lineal y
  esas compras no dominan el ajuste.
- El escalado y la codificación se ajustan solo con entrenamiento, porque van
  dentro del ``Pipeline`` junto con el modelo.
- Se compara contra una línea base que siempre predice el promedio.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from .preprocesador import PreprocesadorTransacciones


class RegresionMonto:
    """Entrena, evalúa e interpreta una regresión lineal del monto.

    Args:
        numericas: predictoras numéricas (se escalan).
        categoricas: predictoras categóricas (one-hot, primera categoría
            como referencia).
        objetivo: columna a predecir, en escala logarítmica.
        columna_monto: monto original en dólares, para reportar el error en
            unidades reales.
        directorio_modelos: carpeta donde se guarda el ``.pkl``.
    """

    def __init__(
        self,
        numericas: list[str],
        categoricas: list[str],
        objetivo: str = "log_monto",
        columna_monto: str = "monto",
        directorio_modelos: str | Path = "modelos",
    ) -> None:
        prohibidas = {objetivo, columna_monto} & set(numericas + categoricas)
        if prohibidas:
            raise ValueError(f"Estas columnas revelan la respuesta: {sorted(prohibidas)}.")
        self.numericas = list(numericas)
        self.categoricas = list(categoricas)
        self.objetivo = objetivo
        self.columna_monto = columna_monto
        self.directorio_modelos = Path(directorio_modelos)
        self.pipeline: Pipeline | None = None
        self.linea_base: DummyRegressor | None = None
        self.metricas: dict[str, dict[str, float]] | None = None
        self.tabla_coeficientes: pd.DataFrame | None = None
        self.intercepto: float | None = None
        self.referencias: dict[str, Any] = {}
        self.ruta_modelo: Path | None = None
        self.importancia: pd.Series | None = None
        self.agrupadas: dict[str, list[str]] = {}
        self.filas_train = self.filas_test = 0

    @property
    def predictoras(self) -> list[str]:
        return self.numericas + self.categoricas

    # ------------------------------------------------------------------ #
    # entrenamiento
    # ------------------------------------------------------------------ #
    def entrenar(self, train: pd.DataFrame) -> Pipeline:
        """Ajusta transformador + LinearRegression solo con entrenamiento."""
        X, y = train[self.predictoras], train[self.objetivo]
        self.pipeline = Pipeline(
            [
                ("preparacion", PreprocesadorTransacciones.construir_transformador(
                    self.numericas, self.categoricas)),
                ("modelo", LinearRegression()),  # parámetros por defecto
            ]
        )
        self.pipeline.fit(X, y)
        self.linea_base = DummyRegressor(strategy="mean").fit(X, y)
        self.filas_train = len(train)

        self.directorio_modelos.mkdir(parents=True, exist_ok=True)
        self.ruta_modelo = self.directorio_modelos / "regresion_lineal.pkl"
        with self.ruta_modelo.open("wb") as archivo:
            pickle.dump(self.pipeline, archivo)
        print(f"[regresion] entrenada con {len(train):,} filas; modelo en {self.ruta_modelo}")
        return self.pipeline

    # ------------------------------------------------------------------ #
    # evaluación
    # ------------------------------------------------------------------ #
    @staticmethod
    def _metricas(real: np.ndarray, pred: np.ndarray) -> dict[str, float]:
        return {
            "mae": float(mean_absolute_error(real, pred)),
            "rmse": float(np.sqrt(mean_squared_error(real, pred))),
            "r2": float(r2_score(real, pred)),
        }

    def evaluar(self, test: pd.DataFrame) -> dict[str, dict[str, float]]:
        """Métricas en escala log y en dólares, para el modelo y la línea base."""
        if self.pipeline is None or self.linea_base is None:
            raise RuntimeError("Primero entrena el modelo.")
        X, y_log = test[self.predictoras], test[self.objetivo].to_numpy()
        dolares = test[self.columna_monto].to_numpy()
        pred_log = self.pipeline.predict(X)
        base_log = self.linea_base.predict(X)
        self.metricas = {
            "modelo_log": self._metricas(y_log, pred_log),
            "base_log": self._metricas(y_log, base_log),
            "modelo_usd": self._metricas(dolares, np.expm1(pred_log)),
            "base_usd": self._metricas(dolares, np.expm1(base_log)),
        }
        self.filas_test = len(test)
        # Importancia por permutación: cuánto baja el R2 si se mezclan al azar
        # los valores de una variable (en prueba). Es más confiable que mirar
        # el coeficiente más grande, que puede venir de una categoría rara.
        resultado = permutation_importance(
            self.pipeline, X, y_log, scoring="r2", n_repeats=3, random_state=42
        )
        self.importancia = pd.Series(
            resultado.importances_mean, index=self.predictoras
        ).sort_values(ascending=False)
        m = self.metricas["modelo_log"]
        print(f"[regresion] prueba: R2={m['r2']:.4f}; MAE(log)={m['mae']:.4f}; "
              f"MAE(USD)={self.metricas['modelo_usd']['mae']:.2f}")
        return self.metricas

    # ------------------------------------------------------------------ #
    # interpretación
    # ------------------------------------------------------------------ #
    def interpretar(self) -> pd.DataFrame:
        """Tabla de coeficientes con su efecto en % sobre el monto."""
        if self.pipeline is None:
            raise RuntimeError("Primero entrena el modelo.")
        prep = self.pipeline.named_steps["preparacion"]
        modelo = self.pipeline.named_steps["modelo"]
        escalador = prep.named_transformers_["num"]
        codificador = prep.named_transformers_["cat"]
        self.agrupadas = {
            var: [str(v) for v in inf]
            for var, inf in zip(self.categoricas, codificador.infrequent_categories_)
            if inf is not None and len(inf)
        }
        self.referencias = {}
        for var, cats, idx in zip(self.categoricas, codificador.categories_, codificador.drop_idx_):
            ref = cats[idx]
            self.referencias[var] = "otros" if str(ref) in self.agrupadas.get(var, []) else ref

        filas = []
        nombres = prep.get_feature_names_out()
        for nombre, coef in zip(nombres, modelo.coef_):
            grupo, columna = nombre.split("__", 1)
            if grupo == "num":
                variable, valor = columna, ""
                desv = float(escalador.scale_[self.numericas.index(columna)])
                lectura = f"+1 desviación ({desv:,.2f})"
            else:
                variable = next(v for v in self.categoricas if columna.startswith(f"{v}_"))
                valor = columna[len(variable) + 1:]
                if valor == "infrequent_sklearn":
                    valor = "otros"
                lectura = f"'{valor}' vs '{self.referencias[variable]}'"
            filas.append({
                "variable": variable,
                "valor": valor,
                "lectura": lectura,
                "coeficiente": float(coef),
                "efecto_%": float(np.expm1(coef) * 100),
            })
        self.tabla_coeficientes = pd.DataFrame(filas)
        self.intercepto = float(modelo.intercept_)
        return self.tabla_coeficientes

    # ------------------------------------------------------------------ #
    # evidencia
    # ------------------------------------------------------------------ #
    def _conclusiones(self) -> list[str]:
        m = self.metricas
        r2 = m["modelo_log"]["r2"]
        mejora = (1 - m["modelo_usd"]["mae"] / m["base_usd"]["mae"]) * 100
        tabla = self.tabla_coeficientes
        imp = self.importancia
        cat = tabla[(tabla["variable"] == "categoria") & (tabla["valor"] != "otros")]
        cat = cat.sort_values("efecto_%")
        nivel = "bajo" if r2 < 0.3 else "moderado" if r2 < 0.6 else "alto"
        return [
            f"- El modelo explica el {r2 * 100:.1f} % de la variación del monto "
            f"(en escala logarítmica). Es un nivel {nivel}.",
            f"- En dólares, se equivoca en promedio por ${m['modelo_usd']['mae']:,.2f}, "
            f"frente a ${m['base_usd']['mae']:,.2f} de la línea base que siempre "
            f"predice el promedio: {mejora:.1f} % menos error.",
            f"- La variable más importante es '{imp.index[0]}': si se mezclan sus valores, "
            f"el R2 baja {imp.iloc[0]:.3f}. Le siguen '{imp.index[1]}' ({imp.iloc[1]:.3f}) "
            f"y '{imp.index[2]}' ({imp.iloc[2]:.3f}).",
            f"- Por categoría de comercio, el monto esperado más alto es en "
            f"'{cat.iloc[-1]['valor']}' ({cat.iloc[-1]['efecto_%']:+.1f} %) y el más "
            f"bajo en '{cat.iloc[0]['valor']}' ({cat.iloc[0]['efecto_%']:+.1f} %), "
            f"comparados con '{self.referencias['categoria']}'."
            if not cat.empty else "- No hay coeficientes de categoría.",
            (
                f"- El RMSE en dólares (${m['modelo_usd']['rmse']:,.2f}) es "
                f"{m['modelo_usd']['rmse'] / m['modelo_usd']['mae']:.1f} veces el MAE: unas "
                "pocas compras muy grandes generan errores enormes que el modelo no anticipa."
                if m["modelo_usd"]["rmse"] > 1.5 * m["modelo_usd"]["mae"]
                else "- El RMSE y el MAE en dólares son parecidos: no hay errores extremos dominando."
            ),
        ] + (
            [
                f"- El R2 en dólares ({m['modelo_usd']['r2']:.3f}) sale más bajo que en escala "
                f"log{', incluso negativo' if m['modelo_usd']['r2'] < 0 else ''}. No es una contradicción: al volver a dólares el modelo "
                "estima el monto típico de una compra, no el promedio, y el R2 en dólares lo "
                "dominan las pocas compras gigantes. Por eso el MAE en dólares es la métrica "
                "más útil para leer el error real."
            ]
            if m["modelo_usd"]["r2"] < m["modelo_log"]["r2"] - 0.05 else []
        )

    def generar_evidencia(self) -> str:
        if self.metricas is None:
            raise RuntimeError("Primero evalúa el modelo.")
        if self.tabla_coeficientes is None:
            self.interpretar()
        m = self.metricas
        tabla_metricas = pd.DataFrame(
            {
                "modelo (log)": m["modelo_log"],
                "linea_base (log)": m["base_log"],
                "modelo (USD)": m["modelo_usd"],
                "linea_base (USD)": m["base_usd"],
            }
        ).T.round(4)
        tabla = self.tabla_coeficientes
        num = tabla[tabla["variable"].isin(self.numericas)]
        cat = tabla[~tabla["variable"].isin(self.numericas)]
        base_usd = float(np.expm1(self.intercepto))
        ref = ", ".join(f"{k}='{v}'" for k, v in self.referencias.items())

        lineas = [
            "EA2 - COMPONENTE 2: REGRESIÓN LINEAL",
            "=" * 70,
            "",
            "PREGUNTA",
            "¿Qué tanto podemos estimar el monto de una compra a partir del tipo de "
            "comercio, el momento de la compra, el perfil del cliente y la distancia al comercio?",
            "",
            "CONFIGURACIÓN",
            f"- Variable objetivo: {self.objetivo} = log(1 + {self.columna_monto})",
            f"- Predictoras numéricas: {', '.join(self.numericas)}",
            f"- Predictoras categóricas: {', '.join(self.categoricas)}",
            f"- Excluidas a propósito: {self.columna_monto} y {self.objetivo} (son la respuesta); "
            "es_fraude (no se conoce al momento de la compra).",
            "- Algoritmo: sklearn.linear_model.LinearRegression con parámetros por defecto "
            "(fit_intercept=True, positive=False).",
            "- Preparación dentro del Pipeline: StandardScaler (numéricas) y "
            "OneHotEncoder(drop='first') (categóricas), ajustados solo con entrenamiento.",
            f"- Entrenamiento: {self.filas_train:,} filas | Prueba: {self.filas_test:,} filas "
            "(misma división del componente 1, semilla 42).",
            "- Línea base: DummyRegressor(strategy='mean'), siempre predice el promedio de entrenamiento.",
            "",
            "EVALUACIÓN EN PRUEBA",
            tabla_metricas.to_string(),
            "",
            "Cómo leer las métricas:",
            "- MAE: error promedio, sin importar el signo.",
            "- RMSE: castiga más los errores grandes.",
            "- R2: proporción de la variación del objetivo que explica el modelo "
            "(1 = perfecto, 0 = igual que predecir el promedio, negativo = peor que eso).",
            "- En USD las predicciones se devuelven a dólares con exp(x) - 1.",
            "",
            "INTERCEPTO",
            f"- Valor: {self.intercepto:.4f} en escala log, unos ${base_usd:,.2f}.",
            f"- Es el monto típico estimado para el caso de referencia ({ref}) "
            "con las variables numéricas en su promedio.",
            "",
            "IMPORTANCIA DE CADA VARIABLE (por permutación, en prueba)",
            "Cuánto baja el R2 si se mezclan al azar los valores de esa variable "
            "(3 repeticiones, semilla 42):",
            self.importancia.round(4).rename("baja_del_R2").to_string(),
            "",
            "COEFICIENTES",
            "Como el objetivo es un logaritmo, cada coeficiente se lee como un cambio "
            "porcentual: efecto_% = (exp(coeficiente) - 1) * 100.",
            "",
            "Numéricas (escaladas): efecto de aumentar una desviación estándar.",
            num[["variable", "lectura", "coeficiente", "efecto_%"]].round(4).to_string(index=False),
            "",
            "Categóricas: efecto frente a la categoría de referencia.",
            "Categorías agrupadas en 'otros' por tener muy pocos casos: "
            f"{self.agrupadas if self.agrupadas else 'ninguna'}.",
        ]
        for variable in self.categoricas:
            sub = cat[cat["variable"] == variable].sort_values("efecto_%", ascending=False)
            lineas += [
                f"* {variable} (referencia: '{self.referencias[variable]}')",
                sub[["valor", "coeficiente", "efecto_%"]].round(4).to_string(index=False),
                "",
            ]
        lineas += [
            "INTERPRETACIÓN (calculada con esta ejecución)",
            *self._conclusiones(),
            "",
            "LIMITACIONES",
            "- El monto depende sobre todo de qué se compra, y ese dato no está en el dataset.",
            "- Un modelo lineal suma efectos: no capta combinaciones (por ejemplo, que la hora "
            "importe distinto en cada tipo de comercio).",
            "- Al usar logaritmo, el modelo estima un monto típico (más cerca de la mediana que "
            "del promedio), así que tiende a quedarse corto en las compras grandes.",
            "- Los efectos son asociaciones, no causas.",
            "",
        ]
        return "\n".join(lineas)

    def ejecutar(self, train: pd.DataFrame, test: pd.DataFrame) -> dict[str, Any]:
        self.entrenar(train)
        metricas = self.evaluar(test)
        coeficientes = self.interpretar()
        return {
            "metricas": metricas,
            "coeficientes": coeficientes,
            "intercepto": self.intercepto,
            "conclusiones": self._conclusiones(),
            "ruta_modelo": self.ruta_modelo,
        }
