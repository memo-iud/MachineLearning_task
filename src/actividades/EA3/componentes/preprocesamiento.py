"""Componente 1: preparar los datos para el Random Forest.

Lo más importante de este paso es que agregamos variables con el historial de
cada tarjeta (cuánto suele gastar, cuántas compras hizo en las últimas 24 horas
y cuánto tiempo pasó desde la compra anterior). En la EA2 dijimos que esa era
la gran limitación del modelo, así que aquí la atacamos.

Cuidamos dos cosas para no hacer trampa (fuga de información):
1. Cada variable de historial solo mira compras ANTERIORES de la misma tarjeta.
   Lo comprobamos con una prueba: si borramos los meses de prueba y volvemos a
   calcular, los valores de entrenamiento no cambian.
2. La división es por fechas: el modelo nunca ve meses posteriores a los que
   usa para aprender.
"""
from __future__ import annotations

from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

import configuracion as cfg

OBJETIVO = "es_fraude"

# Variables que va a usar el modelo.
NUMERICAS = [
    "monto",
    "hora",
    "dia_semana",
    "edad",
    "distancia_km",
    "poblacion_ciudad",
    "veces_monto_habitual",
    "compras_24h",
    "monto_24h",
    "horas_desde_anterior",
]
CATEGORICAS = ["categoria", "genero"]
HISTORIAL = ["veces_monto_habitual", "compras_24h", "monto_24h", "horas_desde_anterior"]

# Columnas que conservamos del dataset original y su nombre en español.
CONSERVAR = {
    "trans_date_trans_time": "fecha",
    "cc_num": "tarjeta",
    "category": "categoria",
    "amt": "monto",
    "gender": "genero",
    "city_pop": "poblacion_ciudad",
    "dob": "fecha_nacimiento",
    "lat": "latitud",
    "long": "longitud",
    "merch_lat": "latitud_comerciante",
    "merch_long": "longitud_comerciante",
    "is_fraud": OBJETIVO,
}

# Las que quitamos y por qué.
DESCARTAR = {
    "Unnamed: 0": "es solo un índice del archivo original.",
    "first": "nombre del cliente: dato personal que no ayuda a predecir.",
    "last": "apellido del cliente: dato personal que no ayuda a predecir.",
    "street": "dirección del cliente: dato personal; la ubicación ya está en latitud y longitud.",
    "trans_num": "código único de cada transacción, no se repite y no enseña nada.",
    "unix_time": "repite la fecha y hora en otro formato.",
    "zip": "código postal: la ubicación ya está en latitud y longitud.",
    "city": "ciudad: cientos de valores distintos; usamos la población de la ciudad.",
    "state": "estado: en la EA2 casi no aportó y tiene 50 categorías.",
    "job": "profesión: cerca de 500 valores distintos.",
    "merchant": "nombre del comercio: cerca de 700 valores; usamos la categoría.",
    "merch_zipcode": "15 % de nulos y repite la ubicación del comercio.",
}

COLOR_FRAUDE = "#E45756"


def calcular_historial(datos: pd.DataFrame) -> pd.DataFrame:
    """Variables del historial de cada tarjeta, usando solo compras anteriores.

    Espera las columnas 'tarjeta', 'fecha' y 'monto'. Devuelve el DataFrame
    ordenado por tarjeta y fecha, con cuatro columnas nuevas.
    """
    d = datos.sort_values(["tarjeta", "fecha"]).reset_index(drop=True)
    por_tarjeta = d.groupby("tarjeta")

    # 1) ¿Cuántas veces más grande es esta compra que lo que suele gastar la tarjeta?
    #    Usamos el promedio del logaritmo de las compras anteriores (un "monto típico"
    #    que no se deja arrastrar por una sola compra gigante).
    log_monto = np.log1p(d["monto"])
    compras_previas = por_tarjeta.cumcount()
    suma_previa = log_monto.groupby(d["tarjeta"]).cumsum() - log_monto
    media_previa = suma_previa / compras_previas.replace(0, np.nan)
    d["veces_monto_habitual"] = np.exp(log_monto - media_previa).fillna(1.0)

    # 2) y 3) Compras y dinero gastado por la misma tarjeta en las 24 horas ANTERIORES.
    #    closed="left" deja por fuera la compra actual.
    ventana = (
        d.set_index("fecha").groupby("tarjeta")["monto"].rolling("24h", closed="left")
    )
    d["compras_24h"] = ventana.count().fillna(0).to_numpy()
    d["monto_24h"] = ventana.sum().fillna(0).to_numpy()

    # 4) Horas desde la compra anterior de la tarjeta (la primera compra no tiene).
    horas = por_tarjeta["fecha"].diff().dt.total_seconds() / 3600
    d["horas_desde_anterior"] = horas.fillna(cfg.TOPE_HORAS).clip(upper=cfg.TOPE_HORAS)
    return d


def _haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = (np.radians(v) for v in (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


def construir_transformador(escalar: bool = False) -> ColumnTransformer:
    """One-hot para las categóricas; las numéricas pasan tal cual.

    No escalamos porque el Random Forest está hecho de árboles, y un árbol solo
    pregunta "¿esta variable es mayor o menor que tal valor?". Escalar no cambia
    el orden de los valores, así que no cambia esas preguntas. El parámetro
    ``escalar`` existe solo para la prueba que lo demuestra.
    """
    numericas = (StandardScaler(), NUMERICAS) if escalar else ("passthrough", NUMERICAS)
    return ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAS),
            ("num", *numericas),
        ]
    )


class PreprocesadorEA3:
    """Hace todo el preprocesamiento y guarda el registro para la evidencia."""

    def __init__(self) -> None:
        self.registro: list[dict[str, str]] = []
        self.antes: dict[str, Any] = {}
        self.despues: dict[str, Any] = {}
        self.datos: pd.DataFrame | None = None
        self.train = self.validacion = self.prueba = None
        self.train_completo: pd.DataFrame | None = None
        self.chequeo_fuga: dict[str, Any] = {}
        self.prueba_escalado: dict[str, Any] = {}

    def _anotar(self, operacion: str, motivo: str, efecto: str) -> None:
        self.registro.append({"operacion": operacion, "motivo": motivo, "efecto": efecto})
        print(f"[1] {operacion}: {efecto}")

    @staticmethod
    def _foto(df: pd.DataFrame) -> dict[str, Any]:
        return {
            "filas": len(df),
            "columnas": df.shape[1],
            "nulos": int(df.isna().sum().sum()),
            "tipos": df.dtypes.astype(str).to_dict(),
        }

    # -------------------------------------------------------------- pasos
    def seleccionar(self, crudo: pd.DataFrame) -> pd.DataFrame:
        quitar = [c for c in DESCARTAR if c in crudo.columns]
        nulos = int(crudo[quitar].isna().sum().sum())
        datos = crudo[list(CONSERVAR)].rename(columns=CONSERVAR)
        self._anotar(
            "Selección de columnas",
            "Quitamos datos personales, identificadores, columnas repetidas y columnas "
            "con cientos de categorías. 'cc_num' se queda solo para agrupar el historial "
            "de cada tarjeta; no entra al modelo.",
            f"{len(quitar)} columnas fuera ({', '.join(quitar)}); con ellas se van "
            f"{nulos:,} nulos. Quedan {datos.shape[1]} columnas renombradas al español.",
        )
        return datos

    def limpiar(self, datos: pd.DataFrame) -> pd.DataFrame:
        repetidas = int(datos.duplicated().sum())
        datos = datos.drop_duplicates()
        datos["fecha"] = pd.to_datetime(datos["fecha"], format="%Y-%m-%d %H:%M:%S")
        datos["fecha_nacimiento"] = pd.to_datetime(datos["fecha_nacimiento"], format="%Y-%m-%d")
        self._anotar(
            "Duplicados y tipos",
            "Una fila repetida contaría doble. Las fechas llegan como texto y hay que "
            "convertirlas para poder calcular horas, edades e historial.",
            f"{repetidas:,} filas duplicadas eliminadas; 'fecha' y 'fecha_nacimiento' "
            "pasan de texto a fecha.",
        )
        return datos

    def enriquecer(self, datos: pd.DataFrame) -> pd.DataFrame:
        datos["hora"] = datos["fecha"].dt.hour
        datos["dia_semana"] = datos["fecha"].dt.dayofweek  # 0 = lunes ... 6 = domingo
        datos["edad"] = ((datos["fecha"] - datos["fecha_nacimiento"]).dt.days / 365.25).astype(int)
        datos["distancia_km"] = _haversine_km(
            datos["latitud"], datos["longitud"],
            datos["latitud_comerciante"], datos["longitud_comerciante"],
        )
        self._anotar(
            "Variables del momento y del cliente",
            "La hora es la señal más clara de la exploración (22 y 23 h). También "
            "calculamos el día de la semana, la edad exacta y la distancia entre cliente y "
            "comercio. Dejamos la hora y el día como números porque el árbol puede "
            "cortarlos en rangos (por ejemplo, 'hora >= 22').",
            "Nuevas: hora, dia_semana, edad, distancia_km.",
        )

        datos = calcular_historial(datos)
        self._anotar(
            "Historial de cada tarjeta",
            "Un fraude suele verse raro para ESE cliente: una compra mucho más grande de "
            "lo normal o varias compras seguidas. Cada variable usa solo compras "
            "anteriores de la misma tarjeta. La primera compra de cada tarjeta no tiene "
            f"historial, así que toma valores fijos (1 vez su monto, 0 compras, {cfg.TOPE_HORAS} "
            "horas). Es una regla, no algo que se aprenda de los datos, así que no "
            "filtra información de prueba.",
            "Nuevas: veces_monto_habitual, compras_24h, monto_24h, horas_desde_anterior.",
        )

        quitar = ["fecha_nacimiento", "latitud", "longitud", "latitud_comerciante", "longitud_comerciante"]
        datos = datos.drop(columns=quitar)
        self._anotar(
            "Quitar columnas ya transformadas",
            "Sus versiones nuevas (edad y distancia) ya tienen la información útil.",
            f"{len(quitar)} columnas eliminadas ({', '.join(quitar)}).",
        )
        return datos

    def verificar_fuga(self, datos: pd.DataFrame) -> None:
        """Prueba de que el historial no mira hacia el futuro.

        Recalculamos el historial borrando todo lo que pasa desde el inicio de la
        prueba. Si alguna variable usara compras futuras, los valores anteriores
        a esa fecha cambiarían.
        """
        base = datos[["tarjeta", "fecha", "monto"]]
        recortado = calcular_historial(base[base["fecha"] < cfg.INICIO_PRUEBA])
        original = datos[datos["fecha"] < cfg.INICIO_PRUEBA]
        cruce = original.merge(recortado, on=["tarjeta", "fecha", "monto"], suffixes=("", "_rec"))
        diferencia = max(
            float((cruce[c] - cruce[f"{c}_rec"]).abs().max()) for c in HISTORIAL
        )
        self.chequeo_fuga = {"filas_comparadas": len(cruce), "diferencia_maxima": diferencia}
        estado = "SIN fuga" if diferencia < 1e-9 else "REVISAR"
        self._anotar(
            "Prueba de fuga de información",
            "Recalculamos el historial borrando los meses de prueba. Si algo mirara hacia "
            "el futuro, los valores anteriores cambiarían.",
            f"{len(cruce):,} filas comparadas; diferencia máxima = {diferencia:.2e} -> {estado}.",
        )

    def dividir(self, datos: pd.DataFrame) -> None:
        fecha = datos["fecha"]
        self.train_completo = datos[fecha < cfg.INICIO_VALIDACION]
        self.validacion = datos[(fecha >= cfg.INICIO_VALIDACION) & (fecha < cfg.INICIO_PRUEBA)]
        self.prueba = datos[fecha >= cfg.INICIO_PRUEBA]
        self._anotar(
            "División temporal",
            "En el banco el modelo se entrena con lo que ya pasó y se usa con lo que viene. "
            "Por eso dividimos por fechas y no al azar. La validación sirve para elegir "
            "parámetros y umbral sin tocar la prueba.",
            f"entrenamiento < {cfg.INICIO_VALIDACION} ({len(self.train_completo):,} filas); "
            f"validación hasta {cfg.INICIO_PRUEBA} ({len(self.validacion):,}); "
            f"prueba desde {cfg.INICIO_PRUEBA} ({len(self.prueba):,}).",
        )

        fraudes = self.train_completo[self.train_completo[OBJETIVO] == 1]
        legitimas = self.train_completo[self.train_completo[OBJETIVO] == 0].sample(
            frac=cfg.FRACCION_LEGITIMAS_TRAIN, random_state=cfg.SEMILLA
        )
        self.train = pd.concat([legitimas, fraudes]).sort_values("fecha").reset_index(drop=True)
        self._anotar(
            "Submuestreo de legítimas en entrenamiento",
            "Hay cerca de 170 legítimas por cada fraude. Quedarnos con todos los fraudes y "
            f"con el {cfg.FRACCION_LEGITIMAS_TRAIN:.0%} de las legítimas hace que el modelo vea "
            "más ejemplos de fraude en proporción y que entrene mucho más rápido en "
            "GitHub Actions. Validación y prueba NO se tocan: conservan la proporción "
            "real, que es contra la que hay que medir. El historial se calculó antes, con "
            "todas las compras, así que este recorte no lo afecta.",
            f"{len(self.train_completo):,} -> {len(self.train):,} filas; fraude "
            f"{self.train_completo[OBJETIVO].mean():.3%} -> {self.train[OBJETIVO].mean():.3%} "
            f"(semilla {cfg.SEMILLA}).",
        )

    def probar_escalado(self) -> None:
        """Evidencia de por qué no escalamos: el mismo árbol con y sin escalado."""
        X_tr, y_tr = self.train[NUMERICAS + CATEGORICAS], self.train[OBJETIVO]
        X_va, y_va = self.validacion[NUMERICAS + CATEGORICAS], self.validacion[OBJETIVO]
        resultados = {}
        for nombre, escalar in (("sin escalar", False), ("con escalado", True)):
            arbol = Pipeline([
                ("prep", construir_transformador(escalar=escalar)),
                ("arbol", DecisionTreeClassifier(max_depth=6, random_state=cfg.SEMILLA)),
            ]).fit(X_tr, y_tr)
            resultados[nombre] = arbol.predict_proba(X_va)[:, 1]
        a, b = resultados["sin escalar"], resultados["con escalado"]
        self.prueba_escalado = {
            "coinciden_%": float(np.mean(np.isclose(a, b)) * 100),
            "auc_sin": float(roc_auc_score(y_va, a)),
            "auc_con": float(roc_auc_score(y_va, b)),
        }
        p = self.prueba_escalado
        self._anotar(
            "Prueba: ¿hace falta escalar?",
            "Entrenamos el mismo árbol de profundidad 6 con y sin StandardScaler y "
            "comparamos sus probabilidades en validación.",
            f"coinciden en el {p['coinciden_%']:.2f} % de las transacciones; ROC-AUC "
            f"{p['auc_sin']:.4f} sin escalar vs {p['auc_con']:.4f} con escalado. "
            "Conclusión: no escalamos.",
        )

    # -------------------------------------------------------------- flujo
    def ejecutar(self, crudo: pd.DataFrame) -> dict[str, pd.DataFrame]:
        self.registro = []
        self.antes = self._foto(crudo)
        datos = self.seleccionar(crudo)
        datos = self.limpiar(datos)
        datos = self.enriquecer(datos)
        self.verificar_fuga(datos)
        self.datos = datos
        self.despues = self._foto(datos)
        self.dividir(datos)
        self.probar_escalado()
        return {"train": self.train, "validacion": self.validacion, "prueba": self.prueba}

    # -------------------------------------------------------------- evidencia
    def _tabla_conjuntos(self) -> pd.DataFrame:
        filas = {}
        for nombre, d in (
            ("entrenamiento (completo)", self.train_completo),
            ("entrenamiento (submuestreado)", self.train),
            ("validación", self.validacion),
            ("prueba", self.prueba),
        ):
            filas[nombre] = {
                "desde": f"{d['fecha'].min():%Y-%m-%d}",
                "hasta": f"{d['fecha'].max():%Y-%m-%d}",
                "filas": len(d),
                "fraudes": int(d[OBJETIVO].sum()),
                "tasa_%": round(d[OBJETIVO].mean() * 100, 3),
            }
        return pd.DataFrame(filas).T

    def _historial_por_clase(self) -> pd.DataFrame:
        d = self.train_completo
        tabla = d.groupby(OBJETIVO)[HISTORIAL].median().T
        tabla.columns = ["mediana legítima", "mediana fraude"]
        return tabla.round(2)

    def generar_evidencia(self) -> str:
        comparacion = pd.DataFrame(
            {
                "antes (crudo)": [self.antes["filas"], self.antes["columnas"], self.antes["nulos"]],
                "después": [self.despues["filas"], self.despues["columnas"], self.despues["nulos"]],
            },
            index=["filas", "columnas", "valores nulos"],
        )
        lineas = [
            "EA3 - COMPONENTE 1: PREPROCESAMIENTO",
            "=" * 70,
            "",
            "ANTES Y DESPUÉS (todo el dataset, antes de dividir)",
            comparacion.to_string(),
            "",
            "Tipos antes:",
            *[f"- {c}: {t}" for c, t in self.antes["tipos"].items()],
            "",
            "Tipos después:",
            *[f"- {c}: {t}" for c, t in self.despues["tipos"].items()],
            "",
            "OPERACIONES (en orden)",
        ]
        for i, paso in enumerate(self.registro, start=1):
            lineas += [
                f"{i}. {paso['operacion']}",
                f"   Por qué: {paso['motivo']}",
                f"   Efecto: {paso['efecto']}",
            ]
        lineas += [
            "",
            "Columnas descartadas y motivo:",
            *[f"- {c}: {m}" for c, m in DESCARTAR.items()],
            "",
            "CONJUNTOS DE DATOS",
            self._tabla_conjuntos().to_string(),
            "",
            "¿LAS VARIABLES DE HISTORIAL SEPARAN FRAUDE DE LEGÍTIMA? (entrenamiento completo)",
            self._historial_por_clase().to_string(),
            "",
            "VARIABLES QUE USARÁ EL MODELO",
            f"- Numéricas (sin escalar): {', '.join(NUMERICAS)}",
            f"- Categóricas (one-hot, aprendido solo con entrenamiento): {', '.join(CATEGORICAS)}",
            f"- Objetivo: {OBJETIVO} (0 = legítima, 1 = fraude)",
            "- 'tarjeta' y 'fecha' se conservan para agrupar y analizar, pero no entran al modelo.",
            "",
            "Muestra del entrenamiento después del preprocesamiento (5 filas):",
            self.train[["fecha", *NUMERICAS, *CATEGORICAS, OBJETIVO]].head().to_string(),
            "",
        ]
        return "\n".join(lineas)

    def graficar(self, ruta) -> None:
        """Tasa de fraude según cada variable de historial (en entrenamiento)."""
        d = self.train_completo
        tramos = {
            "veces_monto_habitual": ([0, 1, 2, 5, 10, np.inf], ["< 1", "1-2", "2-5", "5-10", "> 10"],
                                     "Compra vs. gasto habitual (veces)"),
            "compras_24h": ([-0.5, 0.5, 2.5, 5.5, 10.5, np.inf], ["0", "1-2", "3-5", "6-10", "> 10"],
                            "Compras de la tarjeta en las 24 h anteriores"),
            "monto_24h": ([-0.1, 0, 100, 500, 1500, np.inf], ["$0", "$0-100", "$100-500", "$500-1.500", "> $1.500"],
                          "Dinero gastado en las 24 h anteriores"),
            "horas_desde_anterior": ([-0.1, 1, 6, 24, 72, np.inf], ["< 1 h", "1-6 h", "6-24 h", "1-3 días", "> 3 días"],
                                     "Tiempo desde la compra anterior"),
        }
        fig, ejes = plt.subplots(1, 4, figsize=(20, 4.5))
        tasa_general = d[OBJETIVO].mean() * 100
        for eje, (columna, (cortes, etiquetas, titulo)) in zip(ejes, tramos.items()):
            grupo = pd.cut(d[columna], cortes, labels=etiquetas)
            # Un tramo sin transacciones queda en 0 para que su etiqueta no desaparezca.
            tasa = (d.groupby(grupo, observed=False)[OBJETIVO].mean() * 100).fillna(0)
            barras = eje.bar(etiquetas, tasa.to_numpy(), color=COLOR_FRAUDE)
            eje.bar_label(barras, fmt="%.2f%%", fontsize=8)
            eje.axhline(tasa_general, ls="--", color="gray", label=f"tasa general ({tasa_general:.2f} %)")
            eje.set_title(titulo, fontsize=10)
            eje.set_ylabel("% de fraude")
            eje.tick_params(axis="x", rotation=20)
            eje.set_ylim(0, max(tasa.max(), tasa_general) * 1.35)  # espacio para la leyenda
            eje.legend(fontsize=8, loc="upper left")
        fig.suptitle("¿El historial de la tarjeta ayuda a reconocer el fraude? (entrenamiento, ene-nov 2019)")
        fig.tight_layout()
        ruta.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(ruta, dpi=120)
        plt.close(fig)
        print(f"[1] Gráfica guardada en {ruta}")
