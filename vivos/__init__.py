"""IA de Objetos Vivos: una colonia de reglas chicas, legibles y verificadas que
aprende jugando. Diseño y decisiones: ``docs/IA_OBJETOS_VIVOS.md``.

- ``percepcion``: cuadros 64x64 → objetos, fondo, relojes y efectos.
- ``reglas``: la colonia (ranuras, ciclo de vida, reparación, conjeturas).
- ``grafo``: estados visitados y acciones probadas por nivel.
- ``agente``: decide (plan a meta, exploración) y aprende de cada transición.
- ``arena``: bucle de juego sobre cualquier entorno ``reset``/``step``.
- ``juegos``: banco local hecho con ARCEngine (requiere ``requirements-arc.txt``).
- ``puntaje``: estimación local del puntaje de ARC-AGI-3.
- ``adaptador_agents``: plantilla para ``arcprize/ARC-AGI-3-Agents``.
"""

from .agente import AgenteVivo, Decision
from .arena import Resultado, jugar
from .reglas import Colonia

__all__ = ["AgenteVivo", "Colonia", "Decision", "Resultado", "jugar"]
