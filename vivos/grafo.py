"""Grafo de estados de un nivel: qué acciones se probaron en cada estado y a dónde llevaron."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Iterable

import numpy as np

from .percepcion import clave


@dataclass
class Nodo:
    id: int
    cuadro: np.ndarray
    aristas: dict[str, int] = field(default_factory=dict)  # acción -> id destino
    # Acción -> veces que terminó la partida desde este estado. Una vez puede ser
    # culpa de otra cosa (una barra de energía agotada); dos veces es la acción.
    fatales: dict[str, int] = field(default_factory=dict)

    def fatal(self, accion: str) -> bool:
        return self.fatales.get(accion, 0) >= 2


class Grafo:
    def __init__(self) -> None:
        self.nodos: dict[int, Nodo] = {}
        self._por_clave: dict[str, int] = {}
        self._mascara: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.nodos)

    def nodo(self, grid: np.ndarray, mascara: np.ndarray) -> Nodo:
        k = clave(grid, mascara)
        nid = self._por_clave.get(k)
        if nid is None:
            nid = max(self.nodos, default=-1) + 1
            self.nodos[nid] = Nodo(nid, grid.copy())
            self._por_clave[k] = nid
        return self.nodos[nid]

    def arista(self, origen: Nodo, accion: str, destino: Nodo | None, *, fatal: bool = False) -> None:
        if fatal:
            origen.fatales[accion] = origen.fatales.get(accion, 0) + 1
        elif destino is not None:
            origen.aristas[accion] = destino.id

    def probada(self, nodo: Nodo, accion: str) -> bool:
        return accion in nodo.aristas or accion in nodo.fatales

    def recalcular(self, mascara: np.ndarray) -> None:
        """La máscara de ruido cambió: se fusionan los nodos que ahora son iguales."""
        nueva: dict[str, int] = {}
        alias: dict[int, int] = {}
        for nid in sorted(self.nodos):
            k = clave(self.nodos[nid].cuadro, mascara)
            if k in nueva:
                alias[nid] = nueva[k]
            else:
                nueva[k] = nid
                alias[nid] = nid
        fusion: dict[int, Nodo] = {}
        for nid in sorted(self.nodos):
            viejo = self.nodos[nid]
            dest = fusion.setdefault(alias[nid], Nodo(alias[nid], self.nodos[alias[nid]].cuadro))
            for a, n in viejo.fatales.items():
                dest.fatales[a] = dest.fatales.get(a, 0) + n
            for a, d in viejo.aristas.items():
                dest.aristas.setdefault(a, alias[d])
        self.nodos = fusion
        self._por_clave = nueva

    def camino(self, origen: Nodo, meta: Callable[[Nodo], bool], limite: int = 20000) -> list[str] | None:
        """Acciones conocidas desde ``origen`` hasta el nodo más cercano que cumple ``meta``."""
        if meta(origen):
            return []
        previo: dict[int, tuple[int, str]] = {}
        cola = deque([origen.id])
        vistos = {origen.id}
        while cola and len(vistos) < limite:
            nid = cola.popleft()
            for a, d in sorted(self.nodos[nid].aristas.items()):
                if d in vistos:
                    continue
                vistos.add(d)
                previo[d] = (nid, a)
                if meta(self.nodos[d]):
                    pasos = []
                    while d != origen.id:
                        d, a2 = previo[d]
                        pasos.append(a2)
                    return pasos[::-1]
                cola.append(d)
        return None

    def destino(self, nodo: Nodo, accion: str) -> Nodo | None:
        d = nodo.aristas.get(accion)
        return self.nodos.get(d) if d is not None else None

    def pendientes(self, nodo: Nodo, acciones: Iterable[str]) -> list[str]:
        return [a for a in acciones if not self.probada(nodo, a)]
