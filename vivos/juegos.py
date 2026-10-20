"""Juegos de prueba hechos con ARCEngine, el motor oficial de ARC-AGI-3.

No son los juegos de la competencia (esos no se publican): son un banco local
para medir la IA sin red, con las mismas reglas de juego (64x64, 16 colores,
acciones 1-7 + RESET, por turnos). Cada uno ejercita una capacidad distinta:

* ``laberinto``: mover un avatar hasta la salida, con un reloj que cambia en
  cada acción (ruido que la percepción tiene que ignorar).
* ``interruptores``: solo clicks; prender todas las lámparas, con señuelos que
  no hacen nada.
* ``llave``: tocar la llave abre la puerta que tapa la salida.
* ``trampa``: tocar la lava pierde la partida; hay que aprender a evitarla.
* ``energia``: una barra que se gasta con cada acción (ruido que además mata):
  hay que explorar, morir y volver por lo ya aprendido.

Requiere ``arcengine`` (``requirements-arc.txt``).
"""

from __future__ import annotations

from typing import Callable

from arcengine import (
    ARCBaseGame,
    BlockingMode,
    Camera,
    FrameDataRaw,
    GameAction,
    InteractionMode,
    Level,
    Sprite,
)
from arcengine import ActionInput

# Colores (paleta ARC de 16).
FONDO, PARED, AVATAR, SALIDA, RELOJ = 0, 5, 8, 9, 11
APAGADA, PRENDIDA, SENUELO = 2, 4, 7
LLAVE, PUERTA, LAVA = 12, 10, 2

_MOVIMIENTOS = {
    GameAction.ACTION1: (0, -1),
    GameAction.ACTION2: (0, 1),
    GameAction.ACTION3: (-1, 0),
    GameAction.ACTION4: (1, 0),
}


def _celda(color: int, nombre: str, x: int, y: int, *, tangible: bool = True) -> Sprite:
    return Sprite(
        pixels=[[color]],
        name=nombre,
        x=x,
        y=y,
        blocking=BlockingMode.BOUNDING_BOX,
        interaction=InteractionMode.TANGIBLE if tangible else InteractionMode.INTANGIBLE,
    )


def _paredes(mapa: list[str]) -> list[Sprite]:
    """Un sprite por celda ``#`` del mapa."""
    return [
        _celda(PARED, "pared", x, y)
        for y, fila in enumerate(mapa)
        for x, c in enumerate(fila)
        if c == "#"
    ]


def _ubicar(mapa: list[str], letra: str) -> list[tuple[int, int]]:
    return [(x, y) for y, fila in enumerate(mapa) for x, c in enumerate(fila) if c == letra]


def _nivel_de_mapa(mapa: list[str], extra: Callable[[list[str]], list[Sprite]]) -> Level:
    sprites = _paredes(mapa) + extra(mapa)
    return Level(sprites=sprites, grid_size=(len(mapa[0]), len(mapa)))


class _JuegoDeAvatar(ARCBaseGame):
    """Base: el avatar se mueve con ACTION1-4 y choca con lo tangible."""

    def _avatar(self) -> Sprite:
        return self.current_level.get_sprites_by_name("avatar")[0]

    def _al_chocar(self, otros: list[Sprite]) -> None:  # pragma: no cover - abstracto
        raise NotImplementedError

    def _despues_de_mover(self) -> None:
        """Gancho para efectos de cada acción (el reloj)."""

    def step(self) -> None:
        delta = _MOVIMIENTOS.get(self.action.id)
        if delta is not None:
            choques = self.try_move_sprite(self._avatar(), *delta)
            if choques:
                self._al_chocar(choques)
        self._despues_de_mover()
        self.complete_action()


# --------------------------------------------------------------------------- laberinto

_LABERINTOS = [
    [
        "##########",
        "#A..#....#",
        "#.#.#.##.#",
        "#.#...#..#",
        "#.####.#.#",
        "#......#S#",
        "##########",
    ],
    [
        "############",
        "#A...#.....#",
        "#.##.#.###.#",
        "#.#..#...#.#",
        "#.#.####.#.#",
        "#.#......#.#",
        "#.######.#.#",
        "#........#S#",
        "############",
    ],
]


class Laberinto(_JuegoDeAvatar):
    """Llevar el avatar a la salida. Una barra de reloj avanza en cada acción."""

    def __init__(self) -> None:
        niveles = [
            _nivel_de_mapa(
                m,
                lambda m: [
                    _celda(AVATAR, "avatar", *_ubicar(m, "A")[0]),
                    _celda(SALIDA, "salida", *_ubicar(m, "S")[0]),
                    _celda(RELOJ, "reloj", 0, 0, tangible=False),
                ],
            )
            for m in _LABERINTOS
        ]
        super().__init__(
            game_id="laberinto",
            levels=niveles,
            camera=Camera(background=FONDO, letter_box=3, width=16, height=16),
            win_score=len(niveles),
            available_actions=[1, 2, 3, 4],
        )

    def _despues_de_mover(self) -> None:
        reloj = self.current_level.get_sprites_by_name("reloj")[0]
        ancho = self.current_level.grid_size[0]
        reloj.set_position((reloj.x + 1) % ancho, 0)
        reloj.set_layer(5)

    def _al_chocar(self, otros: list[Sprite]) -> None:
        if any(s.name == "salida" for s in otros):
            self.next_level()


# --------------------------------------------------------------------------- interruptores


def _bloque(color: int, nombre: str, x: int, y: int) -> Sprite:
    return Sprite(
        pixels=[[color, color], [color, color]],
        name=nombre,
        x=x,
        y=y,
        blocking=BlockingMode.BOUNDING_BOX,
        interaction=InteractionMode.TANGIBLE,
    )


class Interruptores(ARCBaseGame):
    """Prender todas las lámparas con clicks. Los señuelos no hacen nada."""

    _NIVELES = [
        {"lamparas": [(1, 1), (9, 5)], "senuelos": [(5, 9)]},
        {"lamparas": [(1, 1), (5, 5), (9, 9), (1, 9)], "senuelos": [(9, 1), (5, 1)]},
    ]

    def __init__(self) -> None:
        niveles = []
        for n, d in enumerate(self._NIVELES):
            sprites = [_bloque(APAGADA, f"lampara{i}", x, y) for i, (x, y) in enumerate(d["lamparas"])]
            sprites += [_bloque(SENUELO, f"senuelo{i}", x, y) for i, (x, y) in enumerate(d["senuelos"])]
            niveles.append(Level(sprites=sprites, grid_size=(12, 12), name=f"n{n}"))
        super().__init__(
            game_id="interruptores",
            levels=niveles,
            camera=Camera(background=FONDO, letter_box=3, width=12, height=12),
            win_score=len(niveles),
            available_actions=[6],
        )

    def step(self) -> None:
        if self.action.id == GameAction.ACTION6:
            celda = self.camera.display_to_grid(self.action.data.get("x", 0), self.action.data.get("y", 0))
            if celda is not None:
                sprite = self.current_level.get_sprite_at(*celda)
                if sprite is not None and sprite.name.startswith("lampara"):
                    sprite.color_remap(None, PRENDIDA)
                    sprite.set_name("prendida" + sprite.name)
            if not any(s.name.startswith("lampara") for s in self.current_level.get_sprites()):
                self.next_level()
        self.complete_action()


# --------------------------------------------------------------------------- llave


_MAPAS_LLAVE = [
    [
        "#########",
        "#A.....K#",
        "#.#####.#",
        "#.......#",
        "#####P###",
        "#...#.#.#",
        "#.#...S.#",
        "#########",
    ],
]


class Llave(_JuegoDeAvatar):
    """Tocar la llave hace desaparecer la puerta que encierra la salida."""

    def __init__(self) -> None:
        niveles = [
            _nivel_de_mapa(
                m,
                lambda m: [
                    _celda(AVATAR, "avatar", *_ubicar(m, "A")[0]),
                    _celda(SALIDA, "salida", *_ubicar(m, "S")[0]),
                    _celda(LLAVE, "llave", *_ubicar(m, "K")[0]),
                    _celda(PUERTA, "puerta", *_ubicar(m, "P")[0]),
                ],
            )
            for m in _MAPAS_LLAVE
        ]
        super().__init__(
            game_id="llave",
            levels=niveles,
            camera=Camera(background=FONDO, letter_box=3, width=16, height=16),
            win_score=len(niveles),
            available_actions=[1, 2, 3, 4],
        )

    def _al_chocar(self, otros: list[Sprite]) -> None:
        for s in otros:
            if s.name == "llave":
                self.current_level.remove_sprite(s)
                for p in self.current_level.get_sprites_by_name("puerta"):
                    self.current_level.remove_sprite(p)
            elif s.name == "salida":
                self.next_level()
                return


# --------------------------------------------------------------------------- trampa


_MAPAS_TRAMPA = [
    [
        "##########",
        "#A.L.....#",
        "#.#L####.#",
        "#.L....#.#",
        "#.#.#L...#",
        "#...#.L.S#",
        "##########",
    ],
]


class Trampa(_JuegoDeAvatar):
    """Tocar la lava pierde la partida (GAME_OVER); hay que llegar a la salida."""

    def __init__(self) -> None:
        niveles = [
            _nivel_de_mapa(
                m,
                lambda m: [
                    _celda(AVATAR, "avatar", *_ubicar(m, "A")[0]),
                    _celda(SALIDA, "salida", *_ubicar(m, "S")[0]),
                ]
                + [_celda(LAVA, "lava", x, y) for x, y in _ubicar(m, "L")],
            )
            for m in _MAPAS_TRAMPA
        ]
        super().__init__(
            game_id="trampa",
            levels=niveles,
            camera=Camera(background=FONDO, letter_box=3, width=16, height=16),
            win_score=len(niveles),
            available_actions=[1, 2, 3, 4],
        )

    def _al_chocar(self, otros: list[Sprite]) -> None:
        if any(s.name == "lava" for s in otros):
            self.lose()
        elif any(s.name == "salida" for s in otros):
            self.next_level()


# --------------------------------------------------------------------------- energía


_MAPAS_ENERGIA = [
    [
        "############",
        "#A.....#...#",
        "#.####.#.#.#",
        "#.#....#.#.#",
        "#.#.####.#.#",
        "#.#......#S#",
        "############",
        "            ",
        "            ",
        "            ",
    ],
]


class Energia(_JuegoDeAvatar):
    """Laberinto con barra de energía: cada acción gasta una celda de la barra de
    abajo; si se agota, GAME_OVER. Hay energía de sobra para el camino, no para
    explorar sin memoria."""

    ENERGIA = 30

    def __init__(self) -> None:
        niveles = []
        for m in _MAPAS_ENERGIA:
            alto = len(m)
            barra = [_celda(RELOJ, f"barra{i}", i % 12, alto - 3 + i // 12, tangible=False) for i in range(self.ENERGIA)]
            niveles.append(
                _nivel_de_mapa(
                    m,
                    lambda m, barra=barra: [
                        _celda(AVATAR, "avatar", *_ubicar(m, "A")[0]),
                        _celda(SALIDA, "salida", *_ubicar(m, "S")[0]),
                    ]
                    + barra,
                )
            )
        super().__init__(
            game_id="energia",
            levels=niveles,
            camera=Camera(background=FONDO, letter_box=3, width=16, height=16),
            win_score=len(niveles),
            available_actions=[1, 2, 3, 4],
        )

    def _despues_de_mover(self) -> None:
        if self._state.value != "NOT_FINISHED" or self._next_level:
            return
        barras = sorted(
            (s for s in self.current_level.get_sprites() if s.name.startswith("barra")),
            key=lambda s: s.x,
        )
        if barras:
            self.current_level.remove_sprite(barras[-1])
        if len(barras) <= 1:
            self.lose()

    def _al_chocar(self, otros: list[Sprite]) -> None:
        if any(s.name == "salida" for s in otros):
            self.next_level()


JUEGOS: dict[str, type[ARCBaseGame]] = {
    "laberinto": Laberinto,
    "interruptores": Interruptores,
    "llave": Llave,
    "trampa": Trampa,
    "energia": Energia,
}


class EntornoLocal:
    """La misma interfaz que ``arc_agi.EnvironmentWrapper`` (``reset`` / ``step``)
    sobre un juego ARCEngine en proceso, sin red ni archivos."""

    def __init__(self, juego: ARCBaseGame) -> None:
        self._juego = juego
        self.acciones_tomadas = 0

    @classmethod
    def crear(cls, nombre: str) -> "EntornoLocal":
        return cls(JUEGOS[nombre]())

    def reset(self) -> FrameDataRaw:
        return self.step(GameAction.RESET)

    def step(self, action: GameAction, data: dict | None = None, reasoning: dict | None = None) -> FrameDataRaw:
        self.acciones_tomadas += 1
        raw = self._juego.perform_action(ActionInput(id=action, data=data or {}), raw=True)
        if not isinstance(raw, FrameDataRaw):  # partida terminada: el motor devuelve FrameData sin cuadros
            out = FrameDataRaw()
            out.game_id = raw.game_id
            out.state = raw.state
            out.levels_completed = getattr(raw, "levels_completed", 0)
            out.win_levels = getattr(raw, "win_levels", 0)
            out.available_actions = raw.available_actions
            return out
        return raw
