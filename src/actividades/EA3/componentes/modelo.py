"""Componente 2: el Random Forest y cómo elegimos sus parámetros.

Un Random Forest es un grupo de árboles de decisión. Cada árbol aprende con una
muestra al azar de las filas y, en cada pregunta que hace, solo puede escoger
entre algunas variables al azar. Al final, la probabilidad de fraude es el
promedio de lo que dicen todos los árboles. Como cada árbol se equivoca de una
forma distinta, al promediarlos los errores se compensan.

Para elegir los parámetros NO usamos la prueba: comparamos varias combinaciones
contra el conjunto de validación (dic-2019 a feb-2020), que el modelo no vio al
entrenar. La métrica para comparar es el PR-AUC, porque el fraude es muy poco
frecuente y lo que le importa al equipo es que el modelo ponga los fraudes
arriba de la lista de alertas.
"""
from __future__ import annotations

import time
from itertools import product
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.tree import plot_tree

import configuracion as cfg
from componentes.preprocesamiento import construir_transformador

# Valores que probamos. Son los parámetros que más cambian el comportamiento del bosque.
BUSQUEDA = {
    "max_depth": [5, 10, 15, 20, None],          # qué tan profundo puede crecer cada árbol
    "min_samples_leaf": [1, 10, 50],             # cuántas transacciones mínimo por hoja
    "class_weight": [None, "balanced_subsample"],  # si le damos más peso al fraude
}
ARBOLES_BUSQUEDA = 100                      # árboles por combinación mientras buscamos
CURVA_ARBOLES = [10, 25, 50, 100, 200, 300, 400]
TOLERANCIA_ARBOLES = 0.002                  # si con menos árboles casi no se pierde nada, usamos menos
TOLERANCIA_PARAMETROS = 0.005               # combinaciones "empatadas" con la mejor


class ModeloFraudeRF:
    """Busca parámetros, entrena el bosque final y elige el umbral de alerta."""

    def __init__(self, numericas: list[str], categoricas: list[str], objetivo: str) -> None:
        self.numericas = numericas
        self.categoricas = categoricas
        self.objetivo = objetivo
        self.busqueda: pd.DataFrame | None = None
        self.mejores: dict[str, Any] = {}
        self.curva: pd.DataFrame | None = None
        self.n_arboles: int | None = None
        self.pipeline: Pipeline | None = None
        self.umbral: float | None = None
        self.info_umbral: dict[str, float] = {}
        self.segundos_entrenamiento: float = 0.0
        self.fila_elegida: pd.Series | None = None
        self.n_empatadas: int = 0

    @property
    def predictoras(self) -> list[str]:
        return self.numericas + self.categoricas

    def _pipeline(self, n_estimators: int, **params) -> Pipeline:
        return Pipeline([
            ("prep", construir_transformador(escalar=False)),
            ("bosque", RandomForestClassifier(
                n_estimators=n_estimators, random_state=cfg.SEMILLA, n_jobs=-1, **params)),
        ])

    def _xy(self, datos: pd.DataFrame):
        return datos[self.predictoras], datos[self.objetivo].to_numpy()

    # ------------------------------------------------------------ parámetros
    def buscar_parametros(self, train: pd.DataFrame, validacion: pd.DataFrame) -> pd.DataFrame:
        X_tr, y_tr = self._xy(train)
        X_va, y_va = self._xy(validacion)
        filas = []
        combinaciones = list(product(*BUSQUEDA.values()))
        for i, valores in enumerate(combinaciones, start=1):
            params = dict(zip(BUSQUEDA, valores))
            inicio = time.time()
            modelo = self._pipeline(ARBOLES_BUSQUEDA, **params).fit(X_tr, y_tr)
            p_tr = modelo.predict_proba(X_tr)[:, 1]
            p_va = modelo.predict_proba(X_va)[:, 1]
            fila = {
                **{k: ("sin límite" if v is None and k == "max_depth" else
                       "ninguno" if v is None else v) for k, v in params.items()},
                "pr_auc_train": average_precision_score(y_tr, p_tr),
                "pr_auc_validacion": average_precision_score(y_va, p_va),
                "roc_auc_validacion": roc_auc_score(y_va, p_va),
                "segundos": time.time() - inicio,
                "_params": params,
            }
            fila["brecha"] = fila["pr_auc_train"] - fila["pr_auc_validacion"]
            filas.append(fila)
            print(f"[2] {i}/{len(combinaciones)} {params} -> PR-AUC validación "
                  f"{fila['pr_auc_validacion']:.4f} ({fila['segundos']:.1f} s)")
        tabla = pd.DataFrame(filas).sort_values("pr_auc_validacion", ascending=False)
        # Entre las combinaciones casi empatadas con la mejor, nos quedamos con la de
        # menor brecha: es la que menos memoriza y por eso debería aguantar mejor
        # con transacciones nuevas.
        mejor = tabla["pr_auc_validacion"].max()
        empatadas = tabla[tabla["pr_auc_validacion"] >= mejor - TOLERANCIA_PARAMETROS]
        elegida = empatadas["brecha"].idxmin()
        tabla["elegida"] = np.where(tabla.index == elegida, "<-- elegida", "")
        self.busqueda = tabla.reset_index(drop=True)
        self.fila_elegida = self.busqueda[self.busqueda["elegida"] != ""].iloc[0]
        self.n_empatadas = len(empatadas)
        self.mejores = dict(self.fila_elegida["_params"])
        print(f"[2] Mejores parámetros: {self.mejores}")
        return self.busqueda

    def curva_de_arboles(self, train: pd.DataFrame, validacion: pd.DataFrame) -> pd.DataFrame:
        """¿Cuántos árboles hacen falta? Entrenamos un bosque grande y medimos
        el PR-AUC usando solo los primeros k árboles (el promedio de sus votos)."""
        X_tr, y_tr = self._xy(train)
        X_va, y_va = self._xy(validacion)
        grande = self._pipeline(max(CURVA_ARBOLES), **self.mejores).fit(X_tr, y_tr)
        X_va_t = grande.named_steps["prep"].transform(X_va)
        suma = np.zeros(len(X_va_t))
        filas = []
        for k, arbol in enumerate(grande.named_steps["bosque"].estimators_, start=1):
            suma += arbol.predict_proba(X_va_t)[:, 1]
            if k in CURVA_ARBOLES:
                filas.append({"arboles": k, "pr_auc_validacion": average_precision_score(y_va, suma / k)})
        self.curva = pd.DataFrame(filas)
        mejor = self.curva["pr_auc_validacion"].max()
        suficiente = self.curva[self.curva["pr_auc_validacion"] >= mejor - TOLERANCIA_ARBOLES]
        self.n_arboles = int(suficiente["arboles"].min())
        print(f"[2] Árboles elegidos: {self.n_arboles}")
        return self.curva

    # ------------------------------------------------------------ modelo final
    def entrenar(self, train: pd.DataFrame) -> Pipeline:
        X_tr, y_tr = self._xy(train)
        inicio = time.time()
        self.pipeline = self._pipeline(self.n_arboles, **self.mejores).fit(X_tr, y_tr)
        self.segundos_entrenamiento = time.time() - inicio
        print(f"[2] Bosque final entrenado en {self.segundos_entrenamiento:.1f} s")
        return self.pipeline

    def probabilidades(self, datos: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(datos[self.predictoras])[:, 1]

    def elegir_umbral(self, validacion: pd.DataFrame) -> float:
        """El umbral que mejor equilibra precisión y recall (F1) en VALIDACIÓN.

        La prueba no se usa para esto: así el resultado de prueba es una medida
        honesta de lo que pasaría con transacciones nuevas.
        """
        y_va = validacion[self.objetivo].to_numpy()
        p_va = self.probabilidades(validacion)
        precision, recall, umbrales = precision_recall_curve(y_va, p_va)
        f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
        mejor = int(np.argmax(f1[:-1]))
        self.umbral = float(umbrales[mejor])
        self.info_umbral = {
            "f1": float(f1[mejor]), "precision": float(precision[mejor]), "recall": float(recall[mejor]),
            "pr_auc": float(average_precision_score(y_va, p_va)),
        }
        print(f"[2] Umbral elegido en validación: {self.umbral:.3f} (F1 {f1[mejor]:.3f})")
        return self.umbral

    def ejecutar(self, train: pd.DataFrame, validacion: pd.DataFrame) -> Pipeline:
        self.buscar_parametros(train, validacion)
        self.curva_de_arboles(train, validacion)
        self.entrenar(train)
        self.elegir_umbral(validacion)
        return self.pipeline

    # ------------------------------------------------------------ evidencia
    def nombres_variables(self) -> list[str]:
        nombres = self.pipeline.named_steps["prep"].get_feature_names_out()
        return [n.split("__", 1)[1] for n in nombres]

    def parametros_finales(self) -> dict[str, Any]:
        bosque = self.pipeline.named_steps["bosque"]
        claves = ["n_estimators", "max_depth", "min_samples_leaf", "class_weight",
                  "max_features", "criterion", "bootstrap", "random_state"]
        return {k: bosque.get_params()[k] for k in claves}

    def generar_evidencia(self) -> str:
        tabla = self.busqueda.drop(columns="_params").round(4)
        params = self.parametros_finales()
        explicacion = {
            "n_estimators": "cuántos árboles tiene el bosque. Más árboles = votación más estable, pero más lento.",
            "max_depth": "cuántas preguntas seguidas puede hacer cada árbol. Muy profundo memoriza; muy bajo se queda corto.",
            "min_samples_leaf": "mínimo de transacciones en cada hoja. Más alto = reglas más generales.",
            "class_weight": "si es 'balanced_subsample', cada fraude pesa más al entrenar cada árbol.",
            "max_features": "cuántas variables al azar puede mirar cada pregunta ('sqrt' = raíz cuadrada del total). Es lo que hace distintos a los árboles.",
            "criterion": "cómo mide qué tan buena es una pregunta (Gini = qué tan mezcladas quedan las clases).",
            "bootstrap": "cada árbol aprende con una muestra al azar, con reemplazo, de las filas.",
            "random_state": "semilla para que el resultado se pueda repetir.",
        }
        por_defecto = {"max_features", "criterion", "bootstrap"}
        lineas = [
            "EA3 - COMPONENTE 2: MODELO (RANDOM FOREST) Y ELECCIÓN DE PARÁMETROS",
            "=" * 70,
            "",
            "TIPO DE APRENDIZAJE: supervisado, clasificación binaria (0 = legítima, 1 = fraude).",
            f"Variables del modelo: {', '.join(self.predictoras)}",
            f"Después del one-hot quedan {len(self.nombres_variables())} columnas.",
            "",
            "CÓMO ELEGIMOS LOS PARÁMETROS",
            f"- Probamos {len(tabla)} combinaciones con {ARBOLES_BUSQUEDA} árboles cada una.",
            "- Cada combinación se entrena con el entrenamiento (ene-nov 2019) y se mide en "
            "validación (dic-2019 a feb-2020). La prueba no se toca.",
            "- Métrica: PR-AUC en validación. 'brecha' = PR-AUC en entrenamiento menos en "
            "validación: si es grande, el modelo memoriza en vez de aprender.",
            "",
            "Resultados (de mejor a peor):",
            tabla.to_string(index=False),
            "",
            f"Combinaciones a menos de {TOLERANCIA_PARAMETROS} del mejor PR-AUC: {self.n_empatadas}. "
            "Entre ellas elegimos la de menor brecha, porque es la que menos memoriza.",
            f"Combinación elegida: {self.mejores}",
            "",
            "¿CUÁNTOS ÁRBOLES?",
            self.curva.round(4).to_string(index=False),
            f"Elegimos {self.n_arboles} árboles: es el número más pequeño que queda a menos de "
            f"{TOLERANCIA_ARBOLES} del mejor PR-AUC. Más árboles casi no mejoran y tardan más.",
            "",
            "PARÁMETROS FINALES",
            *[f"- {k} = {v!r}{' (valor por defecto)' if k in por_defecto else ''}: {explicacion[k]}"
              for k, v in params.items()],
            "- n_jobs = -1: usa todos los núcleos del computador (no cambia el resultado).",
            f"- Tiempo de entrenamiento del bosque final: {self.segundos_entrenamiento:.1f} s",
            "",
            "UMBRAL DE ALERTA (elegido en validación, no en prueba)",
            f"- Umbral = {self.umbral:.4f}: si la probabilidad de fraude es mayor o igual, se genera alerta.",
            f"- Es el que da el mejor F1 en validación: F1 {self.info_umbral['f1']:.3f}, precisión "
            f"{self.info_umbral['precision']:.3f}, recall {self.info_umbral['recall']:.3f}.",
            f"- PR-AUC del bosque final en validación: {self.info_umbral['pr_auc']:.4f}",
            "",
        ]
        return "\n".join(lineas)

    def graficar_ajuste(self, ruta) -> None:
        tabla = self.busqueda.copy()
        tabla["prof"] = tabla["max_depth"].astype(str)
        orden = [str(v) if v is not None else "sin límite" for v in BUSQUEDA["max_depth"]]
        # Dibujamos con posiciones numéricas (0, 1, 2...) y luego ponemos las etiquetas.
        # Con NumPy 2.5, matplotlib 3.9 falla si se anota sobre un eje de texto.
        x = np.arange(len(orden))
        fig, ejes = plt.subplots(1, 3, figsize=(19, 4.8), gridspec_kw={"width_ratios": [1.2, 1.2, 1]})
        for eje, peso, titulo in ((ejes[0], "ninguno", "Sin peso extra al fraude"),
                                  (ejes[1], "balanced_subsample", "Con class_weight='balanced_subsample'")):
            sub = tabla[tabla["class_weight"] == peso]
            for hoja, color in zip(BUSQUEDA["min_samples_leaf"], ["#4C78A8", "#F58518", "#54A24B"]):
                s = sub[sub["min_samples_leaf"] == hoja].set_index("prof").reindex(orden)
                eje.plot(x, s["pr_auc_validacion"].to_numpy(float), "o-", color=color,
                         label=f"validación, hoja ≥ {hoja}")
                eje.plot(x, s["pr_auc_train"].to_numpy(float), ":", color=color, alpha=.7,
                         label=f"entrenamiento, hoja ≥ {hoja}")
            eje.set_xticks(x, orden)
            eje.set_title(titulo)
            eje.set_xlabel("max_depth (profundidad de cada árbol)")
            eje.set_ylabel("PR-AUC")
            eje.set_ylim(0, 1.05)
            eje.legend(fontsize=7, loc="lower right")
        mejor = self.fila_elegida
        ejes[0 if mejor["class_weight"] == "ninguno" else 1].annotate(
            "elegido", (orden.index(str(mejor["max_depth"])), float(mejor["pr_auc_validacion"])),
            xytext=(0, -28), textcoords="offset points", ha="center",
            arrowprops={"arrowstyle": "->"}, fontsize=9)
        ejes[2].plot(self.curva["arboles"], self.curva["pr_auc_validacion"], "o-", color="#E45756")
        ejes[2].axvline(self.n_arboles, ls="--", color="gray", label=f"elegido: {self.n_arboles} árboles")
        ejes[2].set_xlabel("número de árboles")
        ejes[2].set_ylabel("PR-AUC en validación")
        ejes[2].set_title("¿Cuántos árboles hacen falta?")
        ejes[2].legend(fontsize=8)
        fig.suptitle("Elección de parámetros del Random Forest (línea punteada = entrenamiento; "
                     "si se separa mucho de la continua, el modelo memoriza)")
        fig.tight_layout()
        ruta.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(ruta, dpi=120)
        plt.close(fig)
        print(f"[2] Gráfica guardada en {ruta}")

    def graficar_arbol(self, ruta) -> None:
        """Los primeros niveles de UNO de los árboles del bosque, para mostrar cómo pregunta."""
        arbol = self.pipeline.named_steps["bosque"].estimators_[0]
        fig, eje = plt.subplots(figsize=(22, 8))
        plot_tree(arbol, max_depth=3, feature_names=self.nombres_variables(),
                  class_names=["legítima", "fraude"], filled=True, proportion=True,
                  impurity=True, fontsize=8, ax=eje)
        eje.set_title(f"Primeros 3 niveles del árbol 1 de {self.n_arboles}. Cada caja es una pregunta; "
                      "'value' = proporción de legítimas y fraudes que llegan ahí (en la muestra de ese árbol)")
        fig.tight_layout()
        fig.savefig(ruta, dpi=110)
        plt.close(fig)
        print(f"[2] Gráfica guardada en {ruta}")
