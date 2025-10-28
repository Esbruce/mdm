import logging
from functions import run_simulation


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    total_airtime, num_air_phases = run_simulation()
    print(f"Total airtime: {total_airtime:.3f} s")
    print(f"Number of air phases: {num_air_phases}")

    # TODO: plotting can be added here once trajectory storage is implemented

