"""Estimación local del puntaje de ARC-AGI-3 (RHAE).

Según la documentación pública del kit (``arc-agi`` 0.9.3–0.9.7): por nivel,
``min(1.15, humanas / acciones) ** 2``; nivel no resuelto = 0; el promedio del
juego pondera cada nivel por su índice (1, 2, 3, ...). Es una estimación: el
puntaje oficial lo calcula el servidor de la competencia.
"""

from __future__ import annotations

TOPE = 1.15


def por_nivel(humanas: int | None, acciones: int | None) -> float | None:
    if not humanas or humanas <= 0:
        return None
    if not acciones:
        return 0.0
    return min(TOPE, humanas / acciones) ** 2


def de_juego(acciones_por_nivel: list[int], base_humana: list[int] | None) -> float | None:
    """Promedio ponderado por índice de nivel. ``None`` si no hay base humana."""
    if not base_humana:
        return None
    total = peso = 0.0
    for i, h in enumerate(base_humana, start=1):
        a = acciones_por_nivel[i - 1] if i - 1 < len(acciones_por_nivel) else None
        p = por_nivel(h, a)
        if p is None:
            continue
        total += i * p
        peso += i
    return total / peso if peso else None
