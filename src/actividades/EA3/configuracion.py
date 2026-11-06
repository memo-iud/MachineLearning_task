"""Configuración del experimento de la EA3.

Dejamos aquí los valores que deciden cómo se arma el experimento, para que
estén en un solo lugar y queden documentados en el resumen.
"""

SEMILLA = 42

# División temporal: entrenamos con lo más antiguo y probamos con lo más reciente,
# como pasaría de verdad en el banco (el modelo se usa con transacciones del futuro).
INICIO_VALIDACION = "2019-12-01"  # dic-2019 a feb-2020: para elegir parámetros y umbral
INICIO_PRUEBA = "2020-03-01"      # mar-2020 a jun-2020: solo para la evaluación final

# En entrenamiento nos quedamos con todos los fraudes y solo con una parte de
# las transacciones legítimas (ver el preprocesamiento para la explicación).
FRACCION_LEGITIMAS_TRAIN = 0.10

# Tope para "horas desde la compra anterior" (30 días). La primera compra de
# cada tarjeta no tiene una anterior, así que también toma este valor.
TOPE_HORAS = 720
