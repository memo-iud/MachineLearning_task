"""Clasificación de transacciones en legítimas o fraude (EA2, componente 3).

- Clases: 0 = legítima, 1 = fraude (``es_fraude``, viene en el dataset).
- Algoritmo: regresión logística con ``class_weight="balanced"``.
  * Es interpretable: cada coeficiente dice si una característica sube o baja
    el riesgo de fraude (razón de probabilidades = exp(coeficiente)).
  * Entrega probabilidades, así que se puede analizar qué pasa al mover el
    umbral de decisión.
  * Con solo ~0,6 % de fraudes, un modelo sin ajuste aprendería a decir
    siempre "legítima". ``class_weight="balanced"`` hace que cada fraude pese
    tanto como muchas transacciones legítimas al entrenar.
- Se compara con una línea base que siempre predice la clase mayoritaria,
  para mostrar por qué la exactitud sola engaña.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from .preprocesador import PreprocesadorTransacciones

NOMBRES_CLASES = {0: "legitima", 1: "fraude"}
UMBRALES_ANALISIS = (0.5, 0.7, 0.9, 0.95, 0.99)


class ClasificadorFraude:
    """Entrena, evalúa e interpreta un clasificador de fraude.

    Args:
        numericas: predictoras numéricas (se escalan).
        categoricas: predictoras categóricas (one-hot).
        objetivo: columna con la clase (0/1).
        umbral: probabilidad a partir de la cual se predice fraude.
        random_state: semilla del modelo.
        directorio_modelos: carpeta donde se guarda el ``.pkl``.
    """

    def __init__(
        self,
        numericas: list[str],
        categoricas: list[str],
        objetivo: str = "es_fraude",
        umbral: float = 0.5,
        random_state: int = 42,
        directorio_modelos: str | Path = "modelos",
    ) -> None:
        if objetivo in numericas + categoricas:
            raise ValueError("La variable objetivo no puede ser predictora.")
        if not 0 < umbral < 1:
            raise ValueError("umbral debe estar entre 0 y 1.")
        self.numericas = list(numericas)
        self.categoricas = list(categoricas)
        self.objetivo = objetivo
        self.umbral = umbral
        self.random_state = random_state
        self.directorio_modelos = Path(directorio_modelos)
        self.parametros_modelo = {
            "class_weight": "balanced",
            "max_iter": 1000,
            "random_state": random_state,
        }
        self.pipeline: Pipeline | None = None
        self.linea_base: DummyClassifier | None = None
        self.metricas: dict[str, Any] | None = None
        self.tabla_umbrales: pd.DataFrame | None = None
        self.importancia: pd.Series | None = None
        self.tabla_coeficientes: pd.DataFrame | None = None
        self.errores: dict[str, Any] | None = None
        self.ruta_modelo: Path | None = None
        self.distribucion: dict[str, pd.Series] = {}
        self.referencias: dict[str, Any] = {}

    @property
    def predictoras(self) -> list[str]:
        return self.numericas + self.categoricas

    # ------------------------------------------------------------------ #
    # entrenamiento
    # ------------------------------------------------------------------ #
    def entrenar(self, train: pd.DataFrame) -> Pipeline:
        X, y = train[self.predictoras], train[self.objetivo]
        self.distribucion["train"] = y.value_counts().sort_index()
        self.pipeline = Pipeline(
            [
                ("preparacion", PreprocesadorTransacciones.construir_transformador(
                    self.numericas, self.categoricas)),
                ("modelo", LogisticRegression(**self.parametros_modelo)),
            ]
        )
        self.pipeline.fit(X, y)
        self.linea_base = DummyClassifier(strategy="most_frequent").fit(X, y)

        self.directorio_modelos.mkdir(parents=True, exist_ok=True)
        self.ruta_modelo = self.directorio_modelos / "clasificador_fraude.pkl"
        with self.ruta_modelo.open("wb") as archivo:
            pickle.dump(self.pipeline, archivo)
        iteraciones = int(self.pipeline.named_steps["modelo"].n_iter_[0])
        print(f"[clasificacion] entrenado con {len(train):,} filas en {iteraciones} "
              f"iteraciones; modelo en {self.ruta_modelo}")
        return self.pipeline

    # ------------------------------------------------------------------ #
    # evaluación
    # ------------------------------------------------------------------ #
    @staticmethod
    def _metricas(real: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
        p, r, f, n = precision_recall_fscore_support(real, pred, labels=[0, 1], zero_division=0)
        por_clase = pd.DataFrame(
            {"precision": p, "recall": r, "f1": f, "casos": n},
            index=[NOMBRES_CLASES[0], NOMBRES_CLASES[1]],
        )
        matriz = pd.DataFrame(
            confusion_matrix(real, pred, labels=[0, 1]),
            index=["real legitima", "real fraude"],
            columns=["pred legitima", "pred fraude"],
        )
        return {
            "exactitud": float(accuracy_score(real, pred)),
            "por_clase": por_clase,
            "f1_macro": float(por_clase["f1"].mean()),
            "matriz": matriz,
        }

    def evaluar(self, test: pd.DataFrame) -> dict[str, Any]:
        if self.pipeline is None or self.linea_base is None:
            raise RuntimeError("Primero entrena el modelo.")
        X, y = test[self.predictoras], test[self.objetivo].to_numpy()
        self.distribucion["test"] = test[self.objetivo].value_counts().sort_index()
        proba = self.pipeline.predict_proba(X)[:, 1]
        pred = (proba >= self.umbral).astype(int)

        modelo = self._metricas(y, pred)
        modelo["pr_auc"] = float(average_precision_score(y, proba))
        modelo["roc_auc"] = float(roc_auc_score(y, proba))
        base = self._metricas(y, self.linea_base.predict(X))
        base["pr_auc"] = float(y.mean())  # PR-AUC de un modelo al azar = % de fraudes
        base["roc_auc"] = 0.5
        self.metricas = {"modelo": modelo, "base": base}

        filas = []
        for u in UMBRALES_ANALISIS:
            m = self._metricas(y, (proba >= u).astype(int))
            fr = m["por_clase"].loc["fraude"]
            mat = m["matriz"]
            filas.append({
                "umbral": u,
                "precision_fraude": fr["precision"],
                "recall_fraude": fr["recall"],
                "f1_fraude": fr["f1"],
                "fraudes_detectados": int(mat.iloc[1, 1]),
                "fraudes_perdidos": int(mat.iloc[1, 0]),
                "falsas_alarmas": int(mat.iloc[0, 1]),
            })
        self.tabla_umbrales = pd.DataFrame(filas)

        # Importancia por permutación con PR-AUC: cuánto empeora la detección
        # de fraude si se mezclan al azar los valores de cada variable.
        resultado = permutation_importance(
            self.pipeline, X, y, scoring="average_precision",
            n_repeats=3, random_state=self.random_state,
        )
        self.importancia = pd.Series(
            resultado.importances_mean, index=self.predictoras
        ).sort_values(ascending=False)

        self._analizar_errores(test, y, pred, proba)
        fr = modelo["por_clase"].loc["fraude"]
        print(f"[clasificacion] prueba: recall fraude={fr['recall']:.4f}; precisión "
              f"fraude={fr['precision']:.4f}; PR-AUC={modelo['pr_auc']:.4f}")
        return self.metricas

    def _analizar_errores(self, test, y, pred, proba) -> None:
        datos = test.assign(_real=y, _pred=pred, _proba=proba)
        tipo = np.select(
            [(y == 1) & (pred == 1), (y == 1) & (pred == 0), (y == 0) & (pred == 1)],
            ["fraude detectado", "fraude perdido", "falsa alarma"],
            default="legitima correcta",
        )
        datos["tipo"] = tipo
        monto = datos.groupby("tipo")["monto"].median().rename("monto_mediano")
        casos = datos["tipo"].value_counts().rename("casos")
        resumen = pd.concat([casos, monto], axis=1).reindex(
            ["fraude detectado", "fraude perdido", "falsa alarma", "legitima correcta"]
        )
        fraudes = datos[datos["_real"] == 1]
        por_categoria = (
            fraudes.groupby("categoria")
            .agg(fraudes=("_pred", "size"), detectados=("_pred", "sum"))
            .assign(recall=lambda d: d["detectados"] / d["fraudes"])
            .sort_values("fraudes", ascending=False)
        )
        self.errores = {"resumen": resumen, "por_categoria": por_categoria}

    # ------------------------------------------------------------------ #
    # interpretación
    # ------------------------------------------------------------------ #
    def interpretar(self) -> pd.DataFrame:
        if self.pipeline is None:
            raise RuntimeError("Primero entrena el modelo.")
        prep = self.pipeline.named_steps["preparacion"]
        modelo = self.pipeline.named_steps["modelo"]
        codificador = prep.named_transformers_["cat"]
        agrupadas = {
            var: [str(v) for v in inf]
            for var, inf in zip(self.categoricas, codificador.infrequent_categories_)
            if inf is not None and len(inf)
        }
        self.referencias = {}
        for var, cats, idx in zip(self.categoricas, codificador.categories_, codificador.drop_idx_):
            ref = cats[idx]
            self.referencias[var] = "otros" if str(ref) in agrupadas.get(var, []) else ref

        filas = []
        for nombre, coef in zip(prep.get_feature_names_out(), modelo.coef_[0]):
            grupo, columna = nombre.split("__", 1)
            if grupo == "num":
                etiqueta = f"{columna} (+1 desviación)"
            else:
                variable = next(v for v in self.categoricas if columna.startswith(f"{v}_"))
                valor = columna[len(variable) + 1:]
                valor = "otros" if valor == "infrequent_sklearn" else valor
                etiqueta = f"{variable} = {valor} (vs {self.referencias[variable]})"
            filas.append({
                "caracteristica": etiqueta,
                "coeficiente": float(coef),
                "razon_probabilidades": float(np.exp(coef)),
            })
        self.tabla_coeficientes = pd.DataFrame(filas).sort_values("coeficiente", ascending=False)
        return self.tabla_coeficientes

    # ------------------------------------------------------------------ #
    # evidencia
    # ------------------------------------------------------------------ #
    def _conclusiones(self) -> list[str]:
        m, b = self.metricas["modelo"], self.metricas["base"]
        fr = m["por_clase"].loc["fraude"]
        le = m["por_clase"].loc["legitima"]
        mat = m["matriz"]
        perdidos, detectados = int(mat.iloc[1, 0]), int(mat.iloc[1, 1])
        alarmas = int(mat.iloc[0, 1])
        imp = self.importancia
        res = self.errores["resumen"]
        cat = self.errores["por_categoria"]
        cat = cat[cat["fraudes"] >= 5]
        coef = self.tabla_coeficientes
        lineas = [
            f"- La línea base acierta el {b['exactitud'] * 100:.2f} % sin detectar ni un fraude "
            f"(recall de fraude = {b['por_clase'].loc['fraude', 'recall']:.2f}). Por eso la "
            "exactitud no sirve para juzgar este problema.",
            f"- Con umbral {self.umbral}, el modelo detecta {detectados:,} de "
            f"{detectados + perdidos:,} fraudes (recall {fr['recall'] * 100:.1f} %). "
            f"A cambio genera {alarmas:,} falsas alarmas: de cada 100 alertas, "
            f"{fr['precision'] * 100:.1f} son fraude de verdad.",
            f"- Reconoce mejor la clase 'legitima' (F1 {le['f1']:.3f}) que la clase 'fraude' "
            f"(F1 {fr['f1']:.3f}). Su error más frecuente son "
            + (f"las falsas alarmas ({alarmas:,}) más que los fraudes perdidos ({perdidos:,})."
               if alarmas >= perdidos else
               f"los fraudes perdidos ({perdidos:,}) más que las falsas alarmas ({alarmas:,})."),
            f"- PR-AUC = {m['pr_auc']:.3f} frente a {b['pr_auc']:.4f} de un modelo al azar: "
            f"ordena las transacciones por riesgo {m['pr_auc'] / b['pr_auc']:.0f} veces mejor que el azar.",
            f"- La variable más importante para detectar fraude es '{imp.index[0]}' (si se "
            f"mezcla, el PR-AUC baja {imp.iloc[0]:.3f}), seguida de '{imp.index[1]}' "
            f"({imp.iloc[1]:.3f}) y '{imp.index[2]}' ({imp.iloc[2]:.3f}).",
            f"- La característica que más sube el riesgo es {coef.iloc[0]['caracteristica']} "
            f"(multiplica las probabilidades por {coef.iloc[0]['razon_probabilidades']:.2f}).",
        ]
        if pd.notna(res.loc["fraude perdido", "monto_mediano"]):
            lineas.append(
                f"- Los fraudes que se le escapan tienen un monto mediano de "
                f"${res.loc['fraude perdido', 'monto_mediano']:,.2f}, frente a "
                f"${res.loc['fraude detectado', 'monto_mediano']:,.2f} de los que sí detecta: "
                "los fraudes pequeños se parecen más a compras normales."
            )
        if not cat.empty:
            lineas.append(
                f"- Por categoría, detecta mejor el fraude en '{cat['recall'].idxmax()}' "
                f"({cat['recall'].max() * 100:.0f} %) y peor en '{cat['recall'].idxmin()}' "
                f"({cat['recall'].min() * 100:.0f} %)."
            )
        return lineas

    def generar_evidencia(self) -> str:
        if self.metricas is None:
            raise RuntimeError("Primero evalúa el modelo.")
        if self.tabla_coeficientes is None:
            self.interpretar()
        m, b = self.metricas["modelo"], self.metricas["base"]
        dist = pd.DataFrame({
            "entrenamiento": self.distribucion["train"],
            "prueba": self.distribucion["test"],
        }).rename(index=NOMBRES_CLASES)
        dist["% prueba"] = (dist["prueba"] / dist["prueba"].sum() * 100).round(3)
        resumen = pd.DataFrame(
            {
                "modelo": [m["exactitud"], m["por_clase"].loc["fraude", "precision"],
                           m["por_clase"].loc["fraude", "recall"], m["por_clase"].loc["fraude", "f1"],
                           m["f1_macro"], m["pr_auc"], m["roc_auc"]],
                "linea_base": [b["exactitud"], b["por_clase"].loc["fraude", "precision"],
                               b["por_clase"].loc["fraude", "recall"], b["por_clase"].loc["fraude", "f1"],
                               b["f1_macro"], b["pr_auc"], b["roc_auc"]],
            },
            index=["exactitud", "precision fraude", "recall fraude", "f1 fraude",
                   "f1 macro", "PR-AUC", "ROC-AUC"],
        ).round(4)
        coef = self.tabla_coeficientes
        lineas = [
            "EA2 - COMPONENTE 3: CLASIFICACIÓN DE FRAUDE",
            "=" * 70,
            "",
            "QUÉ SE CLASIFICA",
            "Cada transacción en una de dos clases: 0 = legítima, 1 = fraude "
            f"(columna '{self.objetivo}', viene en el dataset; no se construyó a partir de otra variable).",
            "",
            "Distribución de clases:",
            dist.to_string(),
            "",
            "CONFIGURACIÓN",
            f"- Predictoras numéricas: {', '.join(self.numericas)}",
            f"- Predictoras categóricas: {', '.join(self.categoricas)}",
            f"- Excluida: '{self.objetivo}' (es la respuesta). 'monto' entra como log_monto.",
            "- Algoritmo: sklearn.linear_model.LogisticRegression con "
            + ", ".join(f"{k}={v!r}" for k, v in self.parametros_modelo.items())
            + "; el resto por defecto (penalty='l2', C=1.0, solver='lbfgs').",
            "- Por qué este algoritmo: es interpretable (cada coeficiente sube o baja el "
            "riesgo), entrega probabilidades y, con class_weight='balanced', compensa que "
            "haya tan pocos fraudes.",
            "- Preparación dentro del Pipeline: StandardScaler + OneHotEncoder, ajustados solo "
            "con entrenamiento (misma división 80/20 estratificada, semilla 42).",
            f"- Umbral de decisión: {self.umbral} (si la probabilidad de fraude es >= {self.umbral}, "
            "se predice fraude).",
            "- Línea base: DummyClassifier(strategy='most_frequent'), siempre dice 'legítima'.",
            "",
            "MÉTRICAS ELEGIDAS Y POR QUÉ",
            "- La exactitud engaña: con ~0,6 % de fraudes, decir siempre 'legítima' acierta ~99,4 %.",
            "- Recall de fraude: de todos los fraudes reales, cuántos detecta. Es la más "
            "importante, porque un fraude perdido es plata perdida.",
            "- Precisión de fraude: de todas las alertas, cuántas son fraude de verdad. Mide el "
            "costo de las falsas alarmas (clientes molestos, revisiones manuales).",
            "- F1: equilibrio entre precisión y recall.",
            "- PR-AUC: qué tan bien ordena las transacciones por riesgo, sin depender del umbral. "
            "Un modelo al azar obtiene el % de fraudes.",
            "",
            "EVALUACIÓN EN PRUEBA",
            resumen.to_string(),
            "",
            "Matriz de confusión del modelo:",
            m["matriz"].to_string(),
            "",
            "Métricas por clase del modelo:",
            m["por_clase"].round(4).to_string(),
            "",
            "Matriz de confusión de la línea base:",
            b["matriz"].to_string(),
            "",
            "EFECTO DEL UMBRAL (análisis en prueba; el modelo usa el umbral indicado arriba)",
            self.tabla_umbrales.round(4).to_string(index=False),
            "",
            "IMPORTANCIA DE CADA VARIABLE (por permutación, en prueba)",
            "Cuánto baja el PR-AUC si se mezclan al azar los valores de esa variable "
            "(3 repeticiones, semilla 42):",
            self.importancia.round(4).rename("baja_del_PR_AUC").to_string(),
            "",
            "COEFICIENTES",
            "razon_probabilidades = exp(coeficiente). Mayor que 1 sube el riesgo de fraude; "
            "menor que 1 lo baja. Las numéricas están escaladas (efecto de +1 desviación).",
            "",
            "Las 10 que más suben el riesgo:",
            coef.head(10).round(4).to_string(index=False),
            "",
            "Las 10 que más bajan el riesgo:",
            coef.tail(10).iloc[::-1].round(4).to_string(index=False),
            "",
            f"Intercepto: {float(self.pipeline.named_steps['modelo'].intercept_[0]):.4f}",
            "",
            "ANÁLISIS DE ERRORES",
            self.errores["resumen"].round(2).to_string(),
            "",
            "Fraudes detectados por categoría de comercio:",
            self.errores["por_categoria"].round(3).to_string(),
            "",
            "INTERPRETACIÓN (calculada con esta ejecución)",
            *self._conclusiones(),
            "",
            "LIMITACIONES",
            "- class_weight='balanced' sube mucho el recall pero produce muchas falsas alarmas; "
            "en la práctica habría que escoger el umbral según el costo de cada error.",
            "- El modelo es lineal: no capta combinaciones como 'compra grande Y de madrugada' "
            "salvo como suma de efectos por separado.",
            "- No usa el historial del cliente (por ejemplo, si gasta mucho más de lo normal), "
            "que suele ser la señal más fuerte de fraude.",
            "- El dataset es simulado; los patrones pueden ser más limpios que en la vida real.",
            "",
        ]
        return "\n".join(lineas)

    def ejecutar(self, train: pd.DataFrame, test: pd.DataFrame) -> dict[str, Any]:
        self.entrenar(train)
        metricas = self.evaluar(test)
        self.interpretar()
        return {
            "metricas": metricas,
            "umbrales": self.tabla_umbrales,
            "importancia": self.importancia,
            "conclusiones": self._conclusiones(),
            "ruta_modelo": self.ruta_modelo,
        }
