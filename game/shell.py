"""The terminal REPL for the game (materials-engine-spec sec 11, M6).

Pure I/O: read a line, dispatch to :class:`game.session.Session`, print the returned lines.
ASCII only (the Windows console is cp1252). Run with ``python -m game``.
"""

from __future__ import annotations

from .session import Session

_BANNER = [
    "=" * 64,
    "  ALCHEMIST -- a deterministic materials discovery game",
    "  combine elements into alloys, measure what emerges, build machines.",
    "  type 'help' for commands, 'quit' to leave.",
    "=" * 64,
]


def _dispatch(session: Session, line: str) -> tuple[list[str], bool]:
    """Return (output_lines, should_quit) for one input line."""
    parts = line.split()
    if not parts:
        return [], False
    cmd, args = parts[0].lower(), parts[1:]
    if cmd in ("quit", "exit", "q"):
        return ["bye."], True
    if cmd in ("help", "h", "?"):
        return session.help(), False
    if cmd in ("inventory", "inv", "i"):
        return session.inventory(show_all=bool(args) and args[0] == "all"), False
    if cmd == "inspect":
        return (session.inspect(args[0]) if args else ["usage: inspect <handle>"]), False
    if cmd in ("discover", "combine", "mix"):
        return (session.discover(args[0], args[1]) if len(args) >= 2
                else ["usage: discover <handle1> <handle2>"]), False
    if cmd == "machines":
        return session.machines(), False
    if cmd == "build":
        return (session.build(args[0], *args[1:]) if args
                else ["usage: build <machine> <handle...>"]), False
    if cmd in ("goals", "goal", "g"):
        return session.goals(), False
    if cmd == "save":
        return (session.save(args[0]) if args else ["usage: save <file>"]), False
    if cmd == "load":
        return (session.load(args[0]) if args else ["usage: load <file>"]), False
    return [f"unknown command: {cmd!r} (type 'help')"], False


def run(session: Session | None = None) -> None:
    """Run the interactive loop until the player quits or EOF."""
    session = session or Session()
    for line in _BANNER:
        print(line)
    while True:
        try:
            raw = input("\nalchemist> ")
        except (EOFError, KeyboardInterrupt):
            print("\nbye.")
            return
        out, done = _dispatch(session, raw)
        for ln in out:
            print(ln)
        if done:
            return
