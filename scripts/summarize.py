"""Print the main-table success rates and pair-bootstrap intervals as JSON."""

import argparse
import json
from pathlib import Path

from analysis.main_results import summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('outputs/paper'))
    args = parser.parse_args()
    print(json.dumps(summarize(args.root.resolve()), indent=2))


if __name__ == '__main__':
    main()
