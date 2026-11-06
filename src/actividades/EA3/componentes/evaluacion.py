"""Componente 3: evaluar el bosque con los meses de prueba (mar-jun 2020).

Estos meses no se usaron ni para entrenar ni para elegir parámetros ni el
umbral, así que funcionan como "transacciones del futuro".

Además de las métricas de siempre, medimos lo que le importa al equipo de
fraude: cuántas alertas al día tendría que revisar y cuánto dinero en fraude
alcanzaría a frenar. Y lo comparamos contra una regla sencilla que podría usar
el banco sin modelo: revisar las compras más grandes.
"""
from __future__ import annotations

from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

import configuracion as cfg
from componentes.modelo import ModeloFraudeRF

CAPACIDADES = [5, 10, 25, 50, 100, 200]   # alertas que el equipo podría revisar al día
MUESTRA_IMPORTANCIA = 60_000
COLOR_FRAUDE, COLOR_LEGITIMA, COLOR_REGLA = "#E45756", "#4C78A8", "#9D755D"


def _dias(datos: pd.DataFrame) -> int:
    return int(datos["fecha"].dt.normalize().nunique())


class EvaluacionEA3:
    def __init__(self, modelo: ModeloFraudeRF, validacion: pd.DataFrame, prueba: pd.DataFrame) -> None:
        self.modelo = modelo
        self.validacion = validacion
        self.prueba = prueba
        self.objetivo = modelo.objetivo
        self.metricas: dict[str, dict[str, float]] = {}
        self.matriz: pd.DataFrame | None = None
        self.capacidad: pd.DataFrame | None = None
        self.importancia: pd.Series | None = None
        self.importancia_impureza: pd.Series | None = None
        self.errores: dict[str, Any] = {}
        self.p_te: np.ndarray | None = None

    # ------------------------------------------------------------ métricas
    @staticmethod
    def _metricas(y, pred, monto) -> dict[str, float]:
        return {
            "exactitud": accuracy_score(y, pred),
            "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0),
            "f1": f1_score(y, pred, zero_division=0),
            "alertas": int(pred.sum()),
            "fraudes_detectados": int((pred & y).sum()),
            "dinero_fraude_detectado_%": float(monto[(pred & y) == 1].sum() / monto[y == 1].sum() * 100),
        }

    def calcular_metricas(self) -> None:
        y = self.prueba[self.objetivo].to_numpy()
        monto = self.prueba["monto"].to_numpy()
        self.p_te = self.modelo.probabilidades(self.prueba)
        dias = _dias(self.prueba)

        pred = (self.p_te >= self.modelo.umbral).astype(int)
        self.metricas["bosque (umbral elegido)"] = self._metricas(y, pred, monto)
        pred_05 = (self.p_te >= 0.5).astype(int)
        self.metricas["bosque (umbral 0.5)"] = self._metricas(y, pred_05, monto)
        # Línea base 1: decir siempre "legítima".
        self.metricas["siempre legítima"] = self._metricas(y, np.zeros_like(y), monto)
        # Línea base 2: una regla sin modelo con la MISMA cantidad de alertas:
        # revisar las compras más grandes.
        n = int(pred.sum())
        regla = np.zeros_like(y)
        regla[np.argsort(-monto, kind="stable")[:n]] = 1
        self.metricas["regla: las compras más grandes"] = self._metricas(y, regla, monto)

        for m in self.metricas.values():
            m["alertas_por_dia"] = m["alertas"] / dias
        self.metricas["bosque (umbral elegido)"]["pr_auc"] = average_precision_score(y, self.p_te)
        self.metricas["bosque (umbral elegido)"]["roc_auc"] = roc_auc_score(y, self.p_te)
        self.metricas["regla: las compras más grandes"]["pr_auc"] = average_precision_score(y, monto)
        self.metricas["siempre legítima"]["pr_auc"] = float(y.mean())

        vn, fp, fn, vp = confusion_matrix(y, pred).ravel()
        self.matriz = pd.DataFrame(
            [[vn, fp], [fn, vp]],
            index=["real legítima", "real fraude"],
            columns=["sin alerta", "con alerta"],
        )
        print(f"[3] Prueba: recall {self.metricas['bosque (umbral elegido)']['recall']:.3f}, "
              f"precisión {self.metricas['bosque (umbral elegido)']['precision']:.3f}, "
              f"PR-AUC {self.metricas['bosque (umbral elegido)']['pr_auc']:.3f}")

    def calcular_capacidad(self) -> None:
        """Si el equipo solo puede revisar K alertas al día, ¿qué pasa?

        El umbral para cada K se fija en VALIDACIÓN (las K*días transacciones con
        más riesgo) y luego se aplica tal cual a la prueba.
        """
        p_va = self.modelo.probabilidades(self.validacion)
        dias_va, dias_te = _dias(self.validacion), _dias(self.prueba)
        y = self.prueba[self.objetivo].to_numpy()
        monto = self.prueba["monto"].to_numpy()
        orden_va = np.sort(p_va)[::-1]
        filas = []
        for k in CAPACIDADES:
            umbral = orden_va[min(k * dias_va, len(orden_va)) - 1]
            pred = (self.p_te >= umbral).astype(int)
            m = self._metricas(y, pred, monto)
            filas.append({
                "capacidad_dia": k,
                "umbral": umbral,
                "alertas_dia_prueba": m["alertas"] / dias_te,
                "fraudes_detectados_%": m["recall"] * 100,
                "dinero_detectado_%": m["dinero_fraude_detectado_%"],
                "alertas_que_son_fraude_%": m["precision"] * 100,
            })
        self.capacidad = pd.DataFrame(filas)

    def calcular_importancia(self) -> None:
        """Importancia por permutación: cuánto empeora el PR-AUC si mezclamos una variable."""
        muestra = self.prueba.sample(min(MUESTRA_IMPORTANCIA, len(self.prueba)), random_state=cfg.SEMILLA)
        X, y = muestra[self.modelo.predictoras], muestra[self.objetivo].to_numpy()
        r = permutation_importance(self.modelo.pipeline, X, y, scoring="average_precision",
                                   n_repeats=3, random_state=cfg.SEMILLA)
        self.importancia = pd.Series(r.importances_mean, index=self.modelo.predictoras).sort_values()
        # La importancia "interna" del bosque, sumando las columnas one-hot de cada variable.
        bruta = pd.Series(self.modelo.pipeline.named_steps["bosque"].feature_importances_,
                          index=self.modelo.nombres_variables())
        agrupada = {}
        for nombre, valor in bruta.items():
            base = next((c for c in self.modelo.categoricas if nombre.startswith(f"{c}_")), nombre)
            agrupada[base] = agrupada.get(base, 0) + valor
        self.importancia_impureza = pd.Series(agrupada).sort_values(ascending=False)

    def analizar_errores(self) -> None:
        d = self.prueba.assign(
            alerta=(self.p_te >= self.modelo.umbral).astype(int),
            prob=self.p_te,
        )
        tipo = np.select(
            [(d[self.objetivo] == 1) & (d["alerta"] == 1), (d[self.objetivo] == 1) & (d["alerta"] == 0),
             (d[self.objetivo] == 0) & (d["alerta"] == 1)],
            ["fraude detectado", "fraude perdido", "falsa alarma"], default="legítima sin alerta")
        d["tipo"] = tipo
        perfil = d.groupby("tipo").agg(
            casos=("monto", "size"), monto_mediano=("monto", "median"),
            veces_monto_habitual=("veces_monto_habitual", "median"),
            hora_mediana=("hora", "median"), prob_mediana=("prob", "median"),
        ).reindex(["fraude detectado", "fraude perdido", "falsa alarma", "legítima sin alerta"])
        fraudes = d[d[self.objetivo] == 1]
        por_categoria = (
            fraudes.groupby("categoria")
            .agg(fraudes=("alerta", "size"), detectados=("alerta", "sum"), dinero=("monto", "sum"))
            .assign(recall_pct=lambda t: t["detectados"] / t["fraudes"] * 100)
            .sort_values("fraudes", ascending=False)
        )
        por_mes = d.groupby(d["fecha"].dt.to_period("M")).apply(
            lambda g: pd.Series({
                "fraudes": int(g[self.objetivo].sum()),
                "recall_pct": g.loc[g[self.objetivo] == 1, "alerta"].mean() * 100,
                "precision_pct": g.loc[g["alerta"] == 1, self.objetivo].mean() * 100,
            }), include_groups=False)
        self.errores = {"perfil": perfil, "por_categoria": por_categoria, "por_mes": por_mes}

    def ejecutar(self) -> None:
        self.calcular_metricas()
        self.calcular_capacidad()
        self.calcular_importancia()
        self.analizar_errores()

    # ------------------------------------------------------------ evidencia
    def conclusiones(self) -> list[str]:
        b = self.metricas["bosque (umbral elegido)"]
        r = self.metricas["regla: las compras más grandes"]
        s = self.metricas["siempre legítima"]
        perfil = self.errores["perfil"]
        top = list(self.importancia.sort_values(ascending=False).index[:3])
        cap = self.capacidad.set_index("capacidad_dia")
        k_ref = 50 if 50 in cap.index else int(cap.index[len(cap) // 2])
        lineas = [
            f"- Decir siempre 'legítima' acierta el {s['exactitud'] * 100:.2f} % sin frenar un solo "
            "fraude. Por eso la exactitud no sirve para juzgar este modelo.",
            f"- Con el umbral elegido en validación, el bosque genera {b['alertas_por_dia']:.1f} "
            f"alertas al día y detecta el {b['recall'] * 100:.1f} % de los fraudes "
            f"({b['fraudes_detectados']:,} de {int(self.matriz.loc['real fraude'].sum()):,}), que "
            f"equivalen al {b['dinero_fraude_detectado_%']:.1f} % del dinero en fraude.",
            f"- De cada 100 alertas, {b['precision'] * 100:.0f} son fraude de verdad. El resto son "
            f"falsas alarmas ({int(self.matriz.loc['real legítima', 'con alerta']):,} en total).",
            f"- Con las mismas alertas, la regla de revisar las compras más grandes detecta el "
            f"{r['recall'] * 100:.1f} % de los fraudes (precisión {r['precision'] * 100:.0f} %), frente "
            f"al {b['recall'] * 100:.1f} % del modelo. "
            + ("El modelo aporta más que una regla sencilla." if b["recall"] > r["recall"]
               else "En esta ejecución el modelo NO supera a la regla sencilla."),
            f"- PR-AUC {b['pr_auc']:.3f} (el azar daría {s['pr_auc']:.4f}) y ROC-AUC {b['roc_auc']:.3f}.",
            f"- Si el equipo solo puede revisar {k_ref} alertas al día, detectaría el "
            f"{cap.loc[k_ref, 'fraudes_detectados_%']:.1f} % de los fraudes y el "
            f"{cap.loc[k_ref, 'dinero_detectado_%']:.1f} % del dinero.",
            f"- Las variables que más pesan son {', '.join(top)}.",
        ]
        perdido = perfil.loc["fraude perdido", "monto_mediano"]
        detectado = perfil.loc["fraude detectado", "monto_mediano"]
        if pd.notna(perdido) and pd.notna(detectado):
            lineas.append(
                f"- Los fraudes que se escapan tienen un monto mediano de ${perdido:,.2f}, frente a "
                f"${detectado:,.2f} de los detectados"
                + (": el modelo falla sobre todo en fraudes pequeños, que se parecen a compras normales."
                   if perdido < detectado else "."))
        return lineas

    def generar_evidencia(self) -> str:
        tabla = pd.DataFrame(self.metricas).T
        columnas = ["exactitud", "precision", "recall", "f1", "pr_auc", "roc_auc", "alertas",
                    "alertas_por_dia", "fraudes_detectados", "dinero_fraude_detectado_%"]
        tabla = tabla.reindex(columns=columnas).astype(float).round(4)
        b = self.metricas["bosque (umbral elegido)"]
        lineas = [
            "EA3 - COMPONENTE 3: EVALUACIÓN EN PRUEBA (mar-jun 2020)",
            "=" * 70,
            "",
            f"Transacciones de prueba: {len(self.prueba):,} en {_dias(self.prueba)} días; "
            f"fraudes: {int(self.prueba[self.objetivo].sum()):,} "
            f"({self.prueba[self.objetivo].mean():.3%}).",
            f"Umbral usado: {self.modelo.umbral:.4f} (elegido en validación).",
            "",
            "POR QUÉ ESTAS MÉTRICAS",
            "- Exactitud: la mostramos, pero engaña: casi todo es legítimo.",
            "- Recall: de los fraudes reales, cuántos alertamos. Un fraude que se escapa es plata perdida.",
            "- Precisión: de las alertas, cuántas son fraude. Cada falsa alarma es tiempo de un "
            "analista y un cliente molesto porque le bloquean la tarjeta.",
            "- F1: equilibrio entre las dos.",
            "- PR-AUC: qué tan bien ordena la lista de alertas, sin depender del umbral. Es la "
            "que mejor responde la pregunta, porque el equipo revisa de arriba hacia abajo.",
            "- Dinero detectado: qué parte del dinero en fraude cae en las alertas.",
            "",
            "RESULTADOS",
            tabla.to_string(),
            "",
            "MATRIZ DE CONFUSIÓN (umbral elegido)",
            self.matriz.to_string(),
            "- Fraude con alerta: fraudes que el equipo alcanza a frenar.",
            "- Fraude sin alerta: el error más caro, el dinero se pierde.",
            "- Legítima con alerta: falsa alarma, cuesta tiempo y molesta al cliente.",
            "",
            "SI EL EQUIPO SOLO PUEDE REVISAR K ALERTAS AL DÍA",
            "(el umbral de cada fila se fija en validación y se aplica a prueba)",
            self.capacidad.round(4).to_string(index=False),
            "",
            "IMPORTANCIA DE LAS VARIABLES",
            f"Por permutación en una muestra de prueba ({MUESTRA_IMPORTANCIA:,} filas): cuánto baja "
            "el PR-AUC si se mezclan al azar los valores de cada variable.",
            self.importancia.sort_values(ascending=False).round(4).rename("baja_del_PR_AUC").to_string(),
            "",
            "Importancia interna del bosque (reducción de Gini; las columnas one-hot se suman):",
            self.importancia_impureza.round(4).rename("importancia").to_string(),
            "",
            "ANÁLISIS DE ERRORES",
            self.errores["perfil"].round(3).to_string(),
            "",
            "Fraudes por categoría de comercio:",
            self.errores["por_categoria"].round(1).to_string(),
            "",
            "Resultado por mes (¿el modelo se mantiene en el tiempo?):",
            self.errores["por_mes"].round(1).to_string(),
            "",
            "INTERPRETACIÓN (calculada con esta ejecución)",
            *self.conclusiones(),
            "",
            "LIMITACIONES",
            "- El dataset es simulado: en la vida real el fraude cambia y el modelo habría que "
            "reentrenarlo seguido.",
            "- El umbral se eligió con F1, que pesa igual un fraude perdido y una falsa alarma. "
            "Si el banco sabe cuánto le cuesta cada error, se puede mover usando la tabla de capacidad.",
            "- El bosque es menos fácil de explicar que un solo árbol: sabemos qué variables pesan, "
            "pero no hay una sola regla simple para cada alerta.",
            f"- Se entrenó con el {cfg.FRACCION_LEGITIMAS_TRAIN:.0%} de las legítimas, así que las "
            "probabilidades salen más altas que la tasa real de fraude. Por eso el umbral se "
            "eligió en validación, con la proporción real.",
            "",
        ]
        return "\n".join(lineas)

    # ------------------------------------------------------------ gráficas
    def graficar(self, carpeta) -> None:
        carpeta.mkdir(parents=True, exist_ok=True)
        y = self.prueba[self.objetivo].to_numpy()

        # 1. Matriz de confusión
        fig, eje = plt.subplots(figsize=(6.5, 5))
        ConfusionMatrixDisplay(self.matriz.to_numpy(), display_labels=["legítima", "fraude"]).plot(
            ax=eje, cmap="Blues", values_format=",d", colorbar=False)
        eje.set_xticks([0, 1], ["sin alerta", "con alerta"])
        eje.set_xlabel("lo que hizo el modelo")
        eje.set_ylabel("lo que pasó en realidad")
        eje.set_title(f"Matriz de confusión en prueba (umbral {self.modelo.umbral:.3f})")
        fig.tight_layout()
        fig.savefig(carpeta / "03_matriz_confusion.png", dpi=120)
        plt.close(fig)

        # 2. Curvas PR y ROC, con la regla de los montos como comparación
        fig, ejes = plt.subplots(1, 2, figsize=(13, 4.8))
        monto = self.prueba["monto"].to_numpy()
        for puntaje, nombre, color in ((self.p_te, "Random Forest", COLOR_FRAUDE),
                                       (monto, "regla: compras más grandes", COLOR_REGLA)):
            p, r, _ = precision_recall_curve(y, puntaje)
            ejes[0].plot(r, p, color=color, label=f"{nombre} (PR-AUC {average_precision_score(y, puntaje):.3f})")
            fpr, tpr, _ = roc_curve(y, puntaje)
            ejes[1].plot(fpr, tpr, color=color, label=f"{nombre} (ROC-AUC {roc_auc_score(y, puntaje):.3f})")
        ejes[0].axhline(y.mean(), ls="--", color="gray", label=f"azar ({y.mean():.4f})")
        b = self.metricas["bosque (umbral elegido)"]
        ejes[0].plot(b["recall"], b["precision"], "ko", label="umbral elegido")
        ejes[0].set_xlabel("recall (fraudes detectados)")
        ejes[0].set_ylabel("precisión (alertas que son fraude)")
        ejes[0].set_title("Precisión vs recall")
        ejes[0].legend(fontsize=8)
        ejes[1].plot([0, 1], [0, 1], "--", color="gray")
        ejes[1].set_xlabel("legítimas con alerta (falsas alarmas)")
        ejes[1].set_ylabel("fraudes detectados")
        ejes[1].set_title("Curva ROC")
        ejes[1].legend(fontsize=8)
        fig.suptitle("Qué tan bien ordena el modelo las transacciones por riesgo (prueba, mar-jun 2020)")
        fig.tight_layout()
        fig.savefig(carpeta / "03_curvas_pr_roc.png", dpi=120)
        plt.close(fig)

        # 3. Capacidad diaria
        fig, eje = plt.subplots(figsize=(9, 4.8))
        cap = self.capacidad
        eje.plot(cap["capacidad_dia"], cap["fraudes_detectados_%"], "o-", color=COLOR_FRAUDE,
                 label="% de fraudes detectados")
        eje.plot(cap["capacidad_dia"], cap["dinero_detectado_%"], "s-", color="#54A24B",
                 label="% del dinero en fraude detectado")
        eje.plot(cap["capacidad_dia"], cap["alertas_que_son_fraude_%"], "^--", color=COLOR_LEGITIMA,
                 label="% de alertas que son fraude")
        for _, f in cap.iterrows():
            eje.annotate(f"{f['dinero_detectado_%']:.0f}%", (f["capacidad_dia"], f["dinero_detectado_%"]),
                         xytext=(0, 7), textcoords="offset points", ha="center", fontsize=8, color="#54A24B")
        eje.set_xscale("log")
        eje.set_xticks(cap["capacidad_dia"], [str(k) for k in cap["capacidad_dia"]])
        eje.set_xlabel("alertas que el equipo puede revisar al día")
        eje.set_ylabel("%")
        eje.set_ylim(0, 105)
        eje.set_title("¿Cuántas alertas revisar al día? (umbral fijado en validación, medido en prueba)")
        eje.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(carpeta / "03_capacidad_diaria.png", dpi=120)
        plt.close(fig)

        # 4. Importancia
        fig, eje = plt.subplots(figsize=(8, 5))
        imp = self.importancia
        colores = [COLOR_FRAUDE if c in ("veces_monto_habitual", "compras_24h", "monto_24h",
                                         "horas_desde_anterior") else COLOR_LEGITIMA for c in imp.index]
        eje.barh(imp.index, imp.values, color=colores)
        eje.set_xlabel("cuánto baja el PR-AUC si se mezcla la variable")
        eje.set_title("¿Qué variables usa el modelo? (rojo = historial de la tarjeta)")
        fig.tight_layout()
        fig.savefig(carpeta / "03_importancia_variables.png", dpi=120)
        plt.close(fig)
        print(f"[3] Gráficas guardadas en {carpeta}")
