# Note Docstrings are partially or fully written by ChatGPT,
# but I can't be asked to write them out myself and deleting them
# is also unwise, as they DO help you understand the functions imo.

"""
Reduced stress test: Only collect black chips from the board to the removed stack, to see if stacking works reliably.

Setup:
- Manually place BLACK chips on the board at the indices listed as (B).
- Ensure removed stack is empty before you start (removed_chips_count starts at 0 on init).

Graphic (B = place BLACK chip here):
  0(B) ---------------------- 1 --------------------- 2(B)
   |                        |                       |
   |                        |                       |
   |         3(B) --------- 4 ----------- 5(B)      |
   |         |              |             |         |
   |         |              |             |         |
   |         |      6 ----- 7 ----- 8     |         |
   |         |      |               |     |         |
   |         |      |               |     |         |
  9 ------  10(B) --11             12(B) --13 ----- 14
   |         |      |               |     |         |
   |         |      |               |     |         |
   |         |      15 ---- 16(B) --17     |         |
   |         |              |             |         |
   |         |              |             |         |
   |         18 ----------- 19 ---------- 20        |
   |                        |                       |
   |                        |                       |
  21 --------------------- 22(B) ------------------- 23(B)
"""

from piecewalker import ned2_provider

BLACK_START = [0, 2, 3, 5, 10, 12, 16, 22, 23]

if __name__ == "__main__":
    ned2 = ned2_provider.get_ned2()

    last_i = len(BLACK_START) - 1

    # Collect all black chips into removed stack.
    # back_to_idle=True on the last move only.
    for i, idx in enumerate(BLACK_START):
        if i == last_i:
            back_to_idle = True
        else:
            back_to_idle = False

        ned2.move(from_idx=idx, to_idx=None, back_to_idle=back_to_idle)

    ned2_provider.close_ned2()