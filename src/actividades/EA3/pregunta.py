"""Pregunta de negocio de la EA3.

La dejamos en un archivo aparte porque la usan varias partes del flujo
(la evidencia del DataFrame, el resumen .txt y luego los hallazgos), y así
si la ajustamos solo hay que cambiarla aquí.
"""

PREGUNTA = (
    "¿Qué transacciones con tarjeta tienen mayor riesgo de ser fraude, para que "
    "el equipo de prevención de fraude decida cuáles bloquear o revisar primero, "
    "si solo puede revisar un número limitado de alertas al día?"
)

BENEFICIARIO = (
    "El equipo de prevención de fraude del banco: los analistas que revisan las "
    "alertas y deciden si una transacción se bloquea o se deja pasar."
)

DECISION = (
    "Qué transacciones mandar a revisión o bloquear, y desde qué nivel de riesgo "
    "hacerlo, según cuántas alertas alcanza a revisar el equipo en un día."
)

TIPO_APRENDIZAJE = (
    "Supervisado (clasificación). Queremos predecir una variable que ya conocemos "
    "en los datos históricos: si la transacción fue fraude o no."
)

DATASET = "pointe77/credit-card-transaction"
URL_DATASET = f"https://huggingface.co/datasets/{DATASET}"
SPLIT = "train"
OBJETIVO = "is_fraud"
