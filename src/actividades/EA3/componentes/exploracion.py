"""Componente 0: revisar el DataFrame y ver si sirve para la pregunta.

Aquí no transformamos nada todavía. Solo dejamos la evidencia de cómo llegan
los datos (tamaño, columnas, tipos, una muestra) y sacamos unos números y una
gráfica que nos ayudan a justificar por qué este dataset sí responde la
pregunta de negocio.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")  # en GitHub Actions no hay pantalla, así que guardamos las gráficas como PNG
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from clases_po import Reportes

import pregunta as pn

COLOR_LEGITIMA = "#4C78A8"
COLOR_FRAUDE = "#E45756"


def _cifras_clave(df: pd.DataFrame) -> dict:
    """Los números que usamos para justificar el dataset."""
    fecha = pd.to_datetime(df["trans_date_trans_time"], format="%Y-%m-%d %H:%M:%S")
    fraude = df[pn.OBJETIVO] == 1
    return {
        "filas": len(df),
        "tarjetas": int(df["cc_num"].nunique()),
        "desde": fecha.min(),
        "hasta": fecha.max(),
        "fraudes": int(fraude.sum()),
        "tasa": float(fraude.mean()),
        "dinero_fraude": float(df.loc[fraude, "amt"].sum()),
        "dinero_total": float(df["amt"].sum()),
        "monto_mediano_fraude": float(df.loc[fraude, "amt"].median()),
        "monto_mediano_legitima": float(df.loc[~fraude, "amt"].median()),
        "fecha": fecha,
    }


def _por_mes(df: pd.DataFrame, fecha: pd.Series) -> pd.DataFrame:
    mes = fecha.dt.to_period("M")
    tabla = df.groupby(mes)[pn.OBJETIVO].agg(transacciones="size", fraudes="sum")
    tabla["tasa_%"] = (tabla["fraudes"] / tabla["transacciones"] * 100).round(3)
    tabla.index.name = "mes"
    return tabla


def _por_hora(df: pd.DataFrame, fecha: pd.Series) -> pd.DataFrame:
    tabla = df.groupby(fecha.dt.hour)[pn.OBJETIVO].agg(transacciones="size", fraudes="sum")
    tabla["tasa_%"] = (tabla["fraudes"] / tabla["transacciones"] * 100).round(3)
    tabla.index.name = "hora"
    return tabla


def graficar(df: pd.DataFrame, cifras: dict, ruta: Path) -> Path:
    """Tres gráficas rápidas: fraude por hora, montos y fraudes por mes."""
    fecha = cifras["fecha"]
    por_hora = _por_hora(df, fecha)
    por_mes = _por_mes(df, fecha)
    fraude = df[pn.OBJETIVO] == 1

    fig, ejes = plt.subplots(1, 3, figsize=(17, 4.5))

    ejes[0].bar(por_hora.index, por_hora["tasa_%"], color=COLOR_FRAUDE)
    ejes[0].axhline(cifras["tasa"] * 100, ls="--", color="gray", label="tasa general")
    ejes[0].set_title("¿A qué hora pasa el fraude?")
    ejes[0].set_xlabel("hora del día")
    ejes[0].set_ylabel("% de transacciones que son fraude")
    ejes[0].set_xticks(range(0, 24, 2))
    ejes[0].legend()

    bins = np.linspace(0, np.log10(df["amt"].max() + 1), 60)
    ejes[1].hist(np.log10(df.loc[~fraude, "amt"] + 1), bins=bins, density=True,
                 alpha=.6, color=COLOR_LEGITIMA, label="legítima")
    ejes[1].hist(np.log10(df.loc[fraude, "amt"] + 1), bins=bins, density=True,
                 alpha=.6, color=COLOR_FRAUDE, label="fraude")
    ticks = [1, 10, 100, 1000, 10000]
    ejes[1].set_xticks(np.log10(np.array(ticks) + 1), [f"${t:,}" for t in ticks])
    ejes[1].set_title("Monto de la compra (escala logarítmica)")
    ejes[1].set_xlabel("monto")
    ejes[1].set_ylabel("densidad")
    ejes[1].legend()

    etiquetas = por_mes.index.astype(str)
    ejes[2].plot(etiquetas, por_mes["fraudes"], "o-", color=COLOR_FRAUDE)
    ejes[2].set_title("Fraudes por mes")
    ejes[2].set_xlabel("mes")
    ejes[2].set_ylabel("fraudes")
    ejes[2].tick_params(axis="x", rotation=60)

    fig.suptitle(f"{pn.DATASET}: {cifras['filas']:,} transacciones, "
                 f"{cifras['fraudes']:,} fraudes ({cifras['tasa'] * 100:.2f} %)")
    fig.tight_layout()
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta, dpi=120)
    plt.close(fig)
    print(f"[0] Gráfica guardada en {ruta}")
    return ruta


def generar_evidencia(df: pd.DataFrame, cifras: dict) -> str:
    """Texto con la pregunta, la justificación del dataset y la radiografía del DataFrame."""
    fecha = cifras["fecha"]
    por_hora = _por_hora(df, fecha).sort_values("tasa_%", ascending=False)
    ahora = datetime.now(ZoneInfo("America/Bogota"))
    pct_dinero = cifras["dinero_fraude"] / cifras["dinero_total"] * 100

    lineas = [
        "EA3 - COMPONENTE 0: PREGUNTA DE NEGOCIO Y DATAFRAME",
        "=" * 70,
        f"Fecha de ejecución: {ahora:%Y-%m-%d %H:%M:%S} (hora de Bogotá)",
        "",
        "PREGUNTA DE NEGOCIO",
        pn.PREGUNTA,
        f"- Quién la necesita: {pn.BENEFICIARIO}",
        f"- Qué decisión apoya: {pn.DECISION}",
        f"- Tipo de aprendizaje: {pn.TIPO_APRENDIZAJE}",
        "",
        "DATASET",
        f"- Identificador: {pn.DATASET}",
        f"- Enlace: {pn.URL_DATASET}",
        f"- Split: {pn.SPLIT}",
        "- Contenido: transacciones de tarjeta de crédito simuladas de clientes en "
        "Estados Unidos, con la fecha y hora, el comercio y su categoría, el monto, "
        "datos del cliente, su ubicación y la del comercio, y la marca de fraude.",
        "",
        "POR QUÉ ESTE DATASET RESPONDE LA PREGUNTA",
        f"- Trae la respuesta histórica: la columna '{pn.OBJETIVO}' dice si cada "
        f"transacción fue fraude ({cifras['fraudes']:,} de {cifras['filas']:,}, "
        f"un {cifras['tasa'] * 100:.3f} %). Con eso podemos entrenar un modelo supervisado.",
        f"- Tiene el historial de cada tarjeta: {cifras['tarjetas']:,} tarjetas distintas "
        "con muchas compras cada una, así que podemos medir si una compra se sale de lo "
        "normal para ese cliente.",
        f"- Cubre un periodo largo ({cifras['desde']:%Y-%m-%d} a {cifras['hasta']:%Y-%m-%d}), "
        "lo que nos deja entrenar con meses antiguos y probar con los más recientes, "
        "como pasaría en el banco.",
        f"- El fraude es poco frecuente pero caro: es el {cifras['tasa'] * 100:.2f} % de las "
        f"transacciones y el {pct_dinero:.1f} % del dinero (${cifras['dinero_fraude']:,.0f}). "
        f"Una compra fraudulenta tiene un monto mediano de ${cifras['monto_mediano_fraude']:,.2f} "
        f"contra ${cifras['monto_mediano_legitima']:,.2f} de una legítima.",
        f"- La hora importa: las horas con más fraude son "
        + ", ".join(f"{h}h ({t:.2f} %)" for h, t in por_hora["tasa_%"].head(4).items())
        + ". Por eso tiene sentido que el equipo priorice alertas y no las revise todas igual.",
        "",
        Reportes().generar_reporte(df),
        "Muestra del contenido (5 filas):",
        df.head().to_string(),
        "",
        "Transacciones y fraudes por mes:",
        _por_mes(df, fecha).to_string(),
        "",
        "Tasa de fraude por hora del día:",
        _por_hora(df, fecha).to_string(),
        "",
    ]
    return "\n".join(lineas)


def ejecutar(df: pd.DataFrame, salidas: Path, graficas: Path) -> dict:
    cifras = _cifras_clave(df)
    texto = generar_evidencia(df, cifras)
    ruta = salidas / "00_dataframe.txt"
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(texto, encoding="utf-8")
    print(texto)
    print(f"[0] Evidencia guardada en {ruta}")
    graficar(df, cifras, graficas / "00_exploracion.png")
    return cifras
