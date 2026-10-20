"""``qvl-prepare-coco``: download COCO 2017 val and build stage-1 caption JSONLs."""

import argparse

from qwenvl_siglip.data.coco import prepare_coco


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="data/coco")
    ap.add_argument("--heldout", type=int, default=200, help="images held out for evaluation")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    prepare_coco(a.root, a.heldout, a.seed)


if __name__ == "__main__":
    main()
