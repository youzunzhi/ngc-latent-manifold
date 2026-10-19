"""Print the main-table success rates and pair-bootstrap intervals as JSON."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'paper/scripts'))
import rerun


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('outputs/paper'))
    args = parser.parse_args()
    rerun.RESULTS = args.root.resolve() / 'results'
    table = rerun.main_ladder()
    rates = rerun.ladder_rate_intervals(table)
    records = []
    for task, _ in rerun.PAPER_TASKS:
        for offset, record in table[task].items():
            records.append(
                {
                    'task': task,
                    'goal_offset': offset,
                    **{
                        k: v
                        for k, v in record.items()
                        if k not in ('keys', 'first_pairs', 'second_pairs')
                    },
                    'l2_ci95': rates[task, 'first', offset],
                    'ngc_ci95': rates[task, 'second', offset],
                }
            )
    print(json.dumps(records, indent=2))


if __name__ == '__main__':
    main()
