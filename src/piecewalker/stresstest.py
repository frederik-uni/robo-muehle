# Note Docstrings are partially or fully written by ChatGPT,
# but I can't be asked to write them out myself and deleting them
# is also unwise, as they DO help you understand the functions imo.

"""
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

if __name__ == "__main__":
    ned2 = ned2_provider.get_ned2()

    # ---- 20 moves total ----
    ned2.move(from_idx=None, to_idx=6,  back_to_idle=False)
    ned2.move(from_idx=None, to_idx=7,  back_to_idle=False)
    ned2.move(from_idx=None, to_idx=8,  back_to_idle=False)
    ned2.move(from_idx=0,    to_idx=None, back_to_idle=False)

    ned2.move(from_idx=None, to_idx=18, back_to_idle=False)
    ned2.move(from_idx=None, to_idx=19, back_to_idle=False)
    ned2.move(from_idx=None, to_idx=20, back_to_idle=False)
    ned2.move(from_idx=2,    to_idx=None, back_to_idle=False)

    ned2.move(from_idx=None, to_idx=1,  back_to_idle=False)
    ned2.move(from_idx=None, to_idx=4,  back_to_idle=False)
    ned2.move(from_idx=3,    to_idx=None, back_to_idle=False)

    ned2.move(from_idx=None, to_idx=14, back_to_idle=False)

    ned2.move(from_idx=5,    to_idx=None, back_to_idle=False)
    ned2.move(from_idx=10,   to_idx=None, back_to_idle=False)
    ned2.move(from_idx=12,   to_idx=None, back_to_idle=False)
    ned2.move(from_idx=16,   to_idx=None, back_to_idle=False)
    ned2.move(from_idx=22,   to_idx=None, back_to_idle=False)
    ned2.move(from_idx=23,   to_idx=None, back_to_idle=False)

    ned2.move(from_idx=14,   to_idx=0,   back_to_idle=False)
    ned2.move(from_idx=0,    to_idx=23,  back_to_idle=True)

    ned2_provider.close_ned2()