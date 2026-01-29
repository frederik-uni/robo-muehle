import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Runs the robo-muehle project.")

    # For calibration/__main__
    parser.add_argument(
        "--fixed_z",
        dest="fixed_z",
        action="store_true",
        help="If this flag is set, the z-value will not be based on the average of the calibration points. "
        'Instead, a hard-coded z-value will be used instead which has been defined in the "niryo_config.toml" file.',
    )
    parser.set_defaults(fixed_z=False)

    # For runtime/__main__
    parser.add_argument(
        "--calibration_file",
        default=None,
        type=Path,
        help="Path if a particular calibration file should be used. Defaults to the calibration file with the highest index.",
    )
    parser.add_argument(
        "--model-play",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--human-start",
        default=True,
        type=bool,
    )
    parser.add_argument(
        "--robot-only",
        default=False,
        type=bool,
    )

    return parser.parse_args()
