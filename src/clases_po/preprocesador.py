"""Preprocesamiento del dataset de transacciones para la EA2.

La clase toma el DataFrame crudo de Hugging Face y lo deja listo para los
dos modelos (regresión lineal y clasificación). Cada operación queda
registrada con su motivo y su efecto, para generar la evidencia de
"antes y después" que pide la actividad.

Orden del flujo (método ``ejecutar``):

    muestrear -> traducir_y_limpiar -> enriquecer -> dividir

Los pasos que aprenden de los datos (escalado y codificación) NO se aplican
aquí sobre todo el DataFrame: se entregan como un ``ColumnTransformer`` sin
entrenar (``construir_transformador``) que cada modelo ajusta solo con el
conjunto de entrenamiento dentro de su ``Pipeline``.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

OBJETIVO_FRAUDE_ORIGINAL = "is_fraud"

TRADUCCION = {
    "trans_date_trans_time": "fecha_hora_transaccion",
    "category": "categoria",
    "amt": "monto",
    "gender": "genero",
    "state": "estado",
    "lat": "latitud",
    "long": "longitud",
    "city_pop": "poblacion_ciudad",
    "dob": "fecha_nacimiento",
    "merch_lat": "latitud_comerciante",
    "merch_long": "longitud_comerciante",
    "is_fraud": "es_fraude",
}

# Columnas que se eliminan antes de enriquecer, con el motivo de cada una.
COLUMNAS_A_ELIMINAR = {
    "Unnamed: 0": "Índice del archivo original; no describe la transacción.",
    "cc_num": "Número de tarjeta: identificador sensible, no un patrón de compra.",
    "first": "Nombre del cliente: dato personal sin valor predictivo.",
    "last": "Apellido del cliente: dato personal sin valor predictivo.",
    "street": "Dirección del cliente: dato personal; la ubicación ya está en latitud y longitud.",
    "trans_num": "Identificador único de cada transacción; no se repite, así que no enseña nada.",
    "unix_time": "Repite la fecha y hora de la transacción en otro formato.",
    "zip": "Código postal: es una etiqueta de ubicación con miles de valores; la ubicación ya está en latitud y longitud.",
    "merchant": "Nombre del comercio: demasiados valores distintos para un modelo lineal.",
    "city": "Ciudad: demasiados valores distintos; se conserva el estado y la población.",
    "job": "Profesión: demasiados valores distintos para un modelo lineal.",
    "merch_zipcode": "Tiene muchos nulos y repite la ubicación del comercio, que ya está en latitud y longitud.",
}

# Columnas originales que se reemplazan por variables derivadas.
COLUMNAS_REEMPLAZADAS = {
    "fecha_hora_transaccion": "Se reemplaza por 'hora' y 'dia_semana'.",
    "fecha_nacimiento": "Se reemplaza por 'edad'.",
    "latitud": "Se reemplaza por 'distancia_km'.",
    "longitud": "Se reemplaza por 'distancia_km'.",
    "latitud_comerciante": "Se reemplaza por 'distancia_km'.",
    "longitud_comerciante": "Se reemplaza por 'distancia_km'.",
    "poblacion_ciudad": "Se reemplaza por 'log_poblacion'.",
}

DIAS = ["lunes", "martes", "miercoles", "jueves", "viernes", "sabado", "domingo"]

# Una categoría con menos casos que esto en entrenamiento se agrupa en "otros".
MIN_FRECUENCIA = 100

# Variables que pueden usar los modelos.
CATEGORICAS = ["categoria", "genero", "estado", "hora", "dia_semana"]
NUMERICAS = ["edad", "distancia_km", "log_poblacion"]


class PreprocesadorTransacciones:
    """Prepara el dataset de transacciones para regresión y clasificación.

    Args:
        tamano_muestra: filas de la muestra estratificada por fraude. Si el
            DataFrame tiene menos filas, se usa completo.
        test_size: proporción del conjunto de prueba.
        random_state: semilla para que la muestra y la división se repitan.
    """

    def __init__(
        self,
        tamano_muestra: int = 300_000,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> None:
        if tamano_muestra < 10:
            raise ValueError("tamano_muestra debe ser al menos 10.")
        if not 0 < test_size < 1:
            raise ValueError("test_size debe estar entre 0 y 1, sin incluirlos.")
        self.tamano_muestra = tamano_muestra
        self.test_size = test_size
        self.random_state = random_state
        self.registro: list[dict[str, str]] = []
        self.estado_inicial: dict[str, Any] | None = None
        self.estado_final: dict[str, Any] | None = None
        self.df_train: pd.DataFrame | None = None
        self.df_test: pd.DataFrame | None = None
        self.cardinalidad_eliminadas: dict[str, int] = {}

    # ------------------------------------------------------------------ #
    # utilidades
    # ------------------------------------------------------------------ #
    def _registrar(self, operacion: str, motivo: str, efecto: str) -> None:
        self.registro.append({"operacion": operacion, "motivo": motivo, "efecto": efecto})
        print(f"[preprocesamiento] {operacion}: {efecto}")

    @staticmethod
    def _foto(df: pd.DataFrame) -> dict[str, Any]:
        """Guarda el estado del DataFrame para comparar antes y después."""
        return {
            "filas": len(df),
            "columnas": df.shape[1],
            "nulos": int(df.isna().sum().sum()),
            "duplicados": int(df.duplicated().sum()),
            "tipos": df.dtypes.astype(str).to_dict(),
            "muestra": df.head().to_string(),
        }

    # ------------------------------------------------------------------ #
    # pasos del flujo
    # ------------------------------------------------------------------ #
    def muestrear(self, df: pd.DataFrame) -> pd.DataFrame:
        """Toma una muestra estratificada que conserva el % de fraudes."""
        if OBJETIVO_FRAUDE_ORIGINAL not in df.columns:
            raise KeyError(f"Falta la columna '{OBJETIVO_FRAUDE_ORIGINAL}'.")
        filas = len(df)
        if self.tamano_muestra >= filas:
            muestra = df.copy()
        else:
            muestra, _ = train_test_split(
                df,
                train_size=self.tamano_muestra,
                stratify=df[OBJETIVO_FRAUDE_ORIGINAL],
                random_state=self.random_state,
            )
        muestra = muestra.reset_index(drop=True)
        pct_antes = df[OBJETIVO_FRAUDE_ORIGINAL].mean() * 100
        pct_despues = muestra[OBJETIVO_FRAUDE_ORIGINAL].mean() * 100
        self._registrar(
            "Muestreo estratificado",
            "El dataset completo es muy grande para ejecutarlo en cada push; la "
            "estratificación conserva la misma proporción de fraudes.",
            f"{filas:,} -> {len(muestra):,} filas; fraude {pct_antes:.3f} % -> "
            f"{pct_despues:.3f} % (semilla {self.random_state}).",
        )
        return muestra

    def traducir_y_limpiar(self, df: pd.DataFrame) -> pd.DataFrame:
        """Elimina columnas sin valor predictivo, traduce nombres y quita duplicados."""
        datos = df.copy()
        for columna in ("merchant", "city", "job", "zip"):
            if columna in datos.columns:
                self.cardinalidad_eliminadas[columna] = int(datos[columna].nunique())

        eliminar = [c for c in COLUMNAS_A_ELIMINAR if c in datos.columns]
        nulos_eliminados = int(datos[eliminar].isna().sum().sum())
        datos = datos.drop(columns=eliminar)
        detalle = ", ".join(
            f"{c} ({n:,} valores distintos)" for c, n in self.cardinalidad_eliminadas.items()
        )
        self._registrar(
            "Selección de variables (eliminar columnas)",
            "Se quitan datos personales, identificadores, columnas repetidas y "
            f"columnas con demasiadas categorías: {detalle}.",
            f"{len(eliminar)} columnas eliminadas ({', '.join(eliminar)}); "
            f"con ellas se van {nulos_eliminados:,} valores nulos.",
        )

        datos = datos.rename(columns=TRADUCCION)
        self._registrar(
            "Renombrar columnas al español",
            "Mantener el mismo idioma que en la EA1 y que el reporte sea fácil de leer.",
            f"{len([c for c in TRADUCCION if c in df.columns])} columnas renombradas.",
        )

        duplicados = int(datos.duplicated().sum())
        datos = datos.drop_duplicates().reset_index(drop=True)
        self._registrar(
            "Eliminar duplicados",
            "Una fila repetida pesaría doble al entrenar y podría quedar a la vez "
            "en entrenamiento y en prueba.",
            f"{duplicados:,} filas duplicadas eliminadas.",
        )
        return datos

    def enriquecer(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crea variables nuevas a partir de fechas, ubicación y monto."""
        datos = df.copy()

        fecha = pd.to_datetime(datos["fecha_hora_transaccion"], format="%Y-%m-%d %H:%M:%S")
        nacimiento = pd.to_datetime(datos["fecha_nacimiento"], format="%Y-%m-%d")
        self._registrar(
            "Corrección de tipos",
            "Las fechas llegan como texto; hay que convertirlas a fecha para poder calcular con ellas.",
            "'fecha_hora_transaccion' y 'fecha_nacimiento' pasan de texto a datetime.",
        )

        datos["hora"] = fecha.dt.hour
        datos["dia_semana"] = fecha.dt.dayofweek.map(dict(enumerate(DIAS)))
        datos["edad"] = ((fecha - nacimiento).dt.days / 365.25).astype(int)
        datos["distancia_km"] = self._haversine_km(
            datos["latitud"], datos["longitud"],
            datos["latitud_comerciante"], datos["longitud_comerciante"],
        )
        datos["log_poblacion"] = np.log1p(datos["poblacion_ciudad"])
        datos["log_monto"] = np.log1p(datos["monto"])
        self._registrar(
            "Enriquecimiento (variables nuevas)",
            "La hora y el día pueden mostrar patrones de compra y de fraude; la edad "
            "exacta mejora el cálculo por año de la EA1; la distancia resume cuatro "
            "coordenadas en una medida con sentido; el logaritmo reduce el sesgo de "
            "la población y del monto, que tienen unos pocos valores muy grandes.",
            "Nuevas: hora, dia_semana, edad, distancia_km, log_poblacion, log_monto.",
        )

        reemplazadas = [c for c in COLUMNAS_REEMPLAZADAS if c in datos.columns]
        datos = datos.drop(columns=reemplazadas)
        self._registrar(
            "Quitar columnas ya transformadas",
            "Sus versiones derivadas contienen la información útil; dejarlas repetiría datos.",
            f"{len(reemplazadas)} columnas eliminadas ({', '.join(reemplazadas)}).",
        )
        return datos

    def dividir(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Divide en entrenamiento y prueba, estratificado por fraude."""
        train, test = train_test_split(
            df,
            test_size=self.test_size,
            stratify=df["es_fraude"],
            random_state=self.random_state,
        )
        self.df_train = train.reset_index(drop=True)
        self.df_test = test.reset_index(drop=True)
        self._registrar(
            "Separación entrenamiento / prueba",
            "Evaluar con datos que el modelo no vio; se estratifica para que ambos "
            "conjuntos tengan la misma proporción de fraudes. Los dos modelos usan "
            "esta misma división.",
            f"entrenamiento={len(self.df_train):,} filas "
            f"(fraude {self.df_train['es_fraude'].mean() * 100:.3f} %); "
            f"prueba={len(self.df_test):,} filas "
            f"(fraude {self.df_test['es_fraude'].mean() * 100:.3f} %).",
        )
        return self.df_train, self.df_test

    @staticmethod
    def construir_transformador(
        numericas: list[str], categoricas: list[str]
    ) -> ColumnTransformer:
        """Escalado de numéricas + one-hot de categóricas (sin entrenar).

        ``drop="first"`` deja fuera una categoría de cada variable como
        referencia, para que los coeficientes de la regresión se puedan
        interpretar sin redundancia.

        ``min_frequency`` agrupa en "otros" las categorías con muy pocos casos
        en entrenamiento: con tan pocos ejemplos el modelo solo memorizaría
        esas filas y daría coeficientes exagerados.
        """
        return ColumnTransformer(
            [
                ("num", StandardScaler(), numericas),
                ("cat", OneHotEncoder(
                    drop="first",
                    handle_unknown="infrequent_if_exist",
                    min_frequency=MIN_FRECUENCIA,
                ), categoricas),
            ],
            sparse_threshold=0.0,  # matriz densa: el tamaño es manejable y simplifica los modelos
        )

    def ejecutar(self, df: pd.DataFrame) -> dict[str, Any]:
        """Ejecuta el preprocesamiento completo y devuelve los conjuntos."""
        self.registro = []
        self.estado_inicial = self._foto(df)
        datos = self.muestrear(df)
        datos = self.traducir_y_limpiar(datos)
        datos = self.enriquecer(datos)
        self.estado_final = self._foto(datos)
        train, test = self.dividir(datos)
        return {"df": datos, "train": train, "test": test}

    # ------------------------------------------------------------------ #
    # evidencia
    # ------------------------------------------------------------------ #
    def demostrar_transformador(self) -> str:
        """Ajusta el transformador SOLO con entrenamiento y lo aplica a prueba.

        Sirve como evidencia: las numéricas escaladas quedan con media 0 y
        desviación 1 en entrenamiento, pero no exactamente en prueba, porque
        el escalador nunca vio esos datos.
        """
        if self.df_train is None or self.df_test is None:
            raise RuntimeError("Primero ejecuta el preprocesamiento.")
        numericas = NUMERICAS + ["log_monto"]
        transformador = self.construir_transformador(numericas, CATEGORICAS)
        X_train = transformador.fit_transform(self.df_train)
        X_test = transformador.transform(self.df_test)

        escalador = transformador.named_transformers_["num"]
        tr = pd.DataFrame(escalador.transform(self.df_train[numericas]), columns=numericas)
        te = pd.DataFrame(escalador.transform(self.df_test[numericas]), columns=numericas)
        tabla = pd.DataFrame(
            {
                "media_train": tr.mean(),
                "desv_train": tr.std(),
                "media_test": te.mean(),
                "desv_test": te.std(),
            }
        ).round(4) + 0.0  # + 0.0 evita que se imprima "-0.0"
        codificador = transformador.named_transformers_["cat"]
        nombres_cat = transformador.named_transformers_["cat"].get_feature_names_out()
        columnas_cat = {
            var: sum(n.startswith(f"{var}_") for n in nombres_cat) for var in CATEGORICAS
        }
        agrupadas = {
            var: [str(v) for v in inf]
            for var, inf in zip(CATEGORICAS, codificador.infrequent_categories_)
            if inf is not None and len(inf)
        }
        return "\n".join(
            [
                "Transformador (ajustado solo con entrenamiento):",
                f"- StandardScaler sobre: {', '.join(numericas)}",
                f"- OneHotEncoder(drop='first', min_frequency={MIN_FRECUENCIA}) sobre: {', '.join(CATEGORICAS)}",
                f"- Columnas creadas por variable categórica: {columnas_cat}",
                "- Categorías agrupadas en 'otros' por tener menos de "
                f"{MIN_FRECUENCIA} casos en entrenamiento: {agrupadas if agrupadas else 'ninguna'}",
                f"- Matriz final: entrenamiento {X_train.shape}, prueba {X_test.shape}",
                "",
                "Numéricas escaladas (entrenamiento ~ media 0 y desv. 1; prueba solo se aproxima):",
                tabla.to_string(),
            ]
        )

    def generar_evidencia(self) -> str:
        """Texto con el antes, las operaciones y el después."""
        if self.estado_inicial is None or self.estado_final is None:
            raise RuntimeError("Primero ejecuta el preprocesamiento.")
        ini, fin = self.estado_inicial, self.estado_final
        comparacion = pd.DataFrame(
            {
                "antes": [ini["filas"], ini["columnas"], ini["nulos"], ini["duplicados"]],
                "despues": [fin["filas"], fin["columnas"], fin["nulos"], fin["duplicados"]],
            },
            index=["filas", "columnas", "valores nulos", "filas duplicadas"],
        )
        lineas = [
            "EA2 - COMPONENTE 1: PREPROCESAMIENTO",
            "=" * 70,
            "",
            "ANTES Y DESPUÉS",
            comparacion.to_string(),
            "",
            "Tipos antes:",
            *[f"- {c}: {t}" for c, t in ini["tipos"].items()],
            "",
            "Tipos después:",
            *[f"- {c}: {t}" for c, t in fin["tipos"].items()],
            "",
            "OPERACIONES APLICADAS (en orden)",
        ]
        for i, paso in enumerate(self.registro, start=1):
            lineas += [
                f"{i}. {paso['operacion']}",
                f"   Motivo: {paso['motivo']}",
                f"   Efecto: {paso['efecto']}",
            ]
        lineas += [
            "",
            self.demostrar_transformador(),
            "",
            "Muestra después del preprocesamiento (5 filas):",
            fin["muestra"],
            "",
        ]
        return "\n".join(lineas)

    @staticmethod
    def _haversine_km(lat1, lon1, lat2, lon2) -> pd.Series:
        """Distancia en km entre dos puntos sobre la superficie de la Tierra."""
        lat1, lon1, lat2, lon2 = (np.radians(v) for v in (lat1, lon1, lat2, lon2))
        a = (
            np.sin((lat2 - lat1) / 2) ** 2
            + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
        )
        return 2 * 6371.0 * np.arcsin(np.sqrt(a))
