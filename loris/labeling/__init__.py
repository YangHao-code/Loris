"""Latest-paper dynamic-Aux labeling, separate from the legacy PDF36 loop."""

SCHEMA = "loris_latest_paper_labeling_v1"


def main(argv=None):
    from .cli import main as entry

    return entry(argv)
