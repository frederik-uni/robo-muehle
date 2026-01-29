from piecewalker.ned2 import Ned2

# Note Docstrings are partially or fully written by ChatGPT,
# but I can't be asked to write them out myself and deleting them
# is also unwise, as they DO help you understand the functions imo.

_NED2: Ned2 | None = None

def get_ned2() -> Ned2:
    global _NED2
    if _NED2 is None:
        _NED2 = Ned2()
    return _NED2

def close_ned2() -> None:
    global _NED2
    if _NED2 is not None:
        _NED2.close()
        _NED2 = None
