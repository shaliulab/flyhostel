"""
Derive the value of physical constraints of proboscis extension bouts
using existing pose files listed in files.txt

Parameters:
    cone_rad: width of the angle in front of the head where the proboscis can be located, in radians
    max_ext_mm: maximum head-proboscis distance allowed, in mm
    max_tip_vel_mm_s: maximum velocity of the proboscis as it extends and protracts, in mm/s

Example:
{
    "cone_rad": 0.697681725025177,
    "max_ext_mm": 1.1090800762176514,
    "max_tip_vel_mm_s": 19.673828125,
}
"""
from .proboscis_candidates import load_and_derive_parameters

# ==========================================================================
if __name__ == "__main__":

    with open("files.txt", "r") as h:
        files = [l.strip() for l in h]

    load_and_derive_parameters(files)

