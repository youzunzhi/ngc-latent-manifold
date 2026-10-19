"""Print the LaTeX rows of the paper tables and the numbers quoted in the
text, from the rerun results (``rerun.py``). The rows are pasted into
``main.tex`` and ``sections/appendix.tex`` by hand, so that every number in
the paper is traceable to this script and its source. Run from the
repository root::

    uv run --group paper python paper/scripts/tables.py
"""

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))
import rerun  # noqa: E402
from rerun import PAPER_TASKS  # noqa: E402

from experiments.plan import REPLICATE_TRAINING_SEEDS  # noqa: E402


def sign_text(value, digits):
    """Signed number, without a sign where it rounds to zero."""
    text = f'{value:+.{digits}f}'
    return text[1:] if float(text) == 0 else text


def signed(value, digits=1):
    return f'${sign_text(value, digits)}$'


# A record without an interval (every pair has the same value) prints its
# mean alone.
def interval(record, digits=1):
    if record['ci95'] is None:
        return signed(record['delta'], digits)
    lo, hi = record['ci95']
    return f'{signed(record["delta"], digits)} [{signed(lo, digits)}, {signed(hi, digits)}]'


def small_interval(record, digits=1):
    if record['ci95'] is None:
        return signed(record['delta'], digits)
    lo, hi = record['ci95']
    return f'{signed(record["delta"], digits)} {{\\scriptsize$[{sign_text(lo, digits)},{sign_text(hi, digits)}]$}}'


def success_rows(table):
    print(
        '% tab:success rows (task, offset, L2 [CI], NGC [CI], delta [CI], gained/lost); by pair seed in the comment'
    )
    intervals = rerun.ladder_rate_intervals(table)
    for task, name in PAPER_TASKS:
        for offset, r in table[task].items():
            cells = []
            for arm in ('first', 'second'):
                ci = intervals[task, arm, offset]
                cells.append(
                    f'{r[arm]:.1f}'
                    if ci is None
                    else f'{r[arm]:.1f} {{\\scriptsize[{ci[0]:.1f}, {ci[1]:.1f}]}}'
                )
            seeds = {
                k: '/'.join(f'{v:.0f}' for v in r['by_seed'][k])
                for k in ('l2', 'ngc')
            }
            print(
                f'{name} & {offset} & {cells[0]} & {cells[1]} & {interval(r)} & '
                f'{r["pairs_gained"]}/{r["pairs_lost"]} \\\\  % by seed: L2 {seeds["l2"]}, NGC {seeds["ngc"]}'
            )
        print('\\midrule')


def ablation_rows(table):
    print(
        '% tab:ablation rows (task, offset, L2, NGC_Euc, NGC, NGC_Euc - L2 [CI], NGC - NGC_Euc [CI])'
    )
    for task, name in PAPER_TASKS:
        for i, (offset, r) in enumerate(table[task].items()):
            a, b = r['euc_vs_l2'], r['ngc_vs_euc']
            assert abs(a['second'] - b['first']) < 1e-9
            label = name if i == 0 else ''
            print(
                f'{label} & {offset} & {a["first"]:.1f} & {a["second"]:.1f} & {b["second"]:.1f} & '
                f'{small_interval(a)} & {small_interval(b)} \\\\'
            )
        print('\\midrule')


def ckpt_rows(official, ours):
    print(
        '% tab:ckpt rows (task, offset, released L2, released NGC, delta [CI], delta ours)'
    )
    for task, name in PAPER_TASKS:
        for offset, r in official[task].items():
            print(
                f'{name} & {offset} & {r["first"]:.1f} & {r["second"]:.1f} & {interval(r)} & '
                f'{signed(ours[task][offset]["delta"])} \\\\'
            )
        print('\\midrule')


def seed_rows(replicates, ours):
    print(
        '% tab:seeds rows (task, offset, L2 / NGC and delta [CI] per training seed, delta ours)'
    )
    for task, name in PAPER_TASKS:
        for offset in ours[task]:
            cells = []
            for table in replicates:
                r = table[task][offset]
                cells.append(
                    f'{r["first"]:.1f} & {r["second"]:.1f} & {small_interval(r)}'
                )
            print(
                f'{name} & {offset} & '
                + ' & '.join(cells)
                + f' & {signed(ours[task][offset]["delta"])} \\\\'
            )
        print('\\midrule')


def comparability_rows(table):
    print(
        '% tab:comparability rows (task, ours L2 +- SD, NGC +- SD, delta [CI], released L2, NGC, delta [CI])'
    )
    for task, name in PAPER_TASKS:
        cells = []
        for kind in rerun.CHECKPOINTS:
            r = table[task][kind]
            for arm in ('l2', 'ngc'):
                runs = r['runs'][arm]
                cells.append(
                    f'${runs.mean():.1f}$ {{\\scriptsize$\\pm{runs.std(ddof=1):.1f}$}}'
                )
            cells.append(small_interval(r))
        print(f'{name} & ' + ' & '.join(cells) + ' & \\\\')


def oracle_rows(table):
    print(
        '% tab:oracle rows (task, offset, L2 learned, oracle, NGC learned, oracle, oracle NGC - L2 [CI], NGC oracle - learned [CI])'
    )
    for task, name in PAPER_TASKS:
        if task not in table:
            continue
        for offset, r in table[task].items():
            learned, oracle = r['learned'], r['oracle']
            print(
                f'{name} & {offset} & {learned["first"]:.0f} & {oracle["first"]:.0f}'
                f' & {learned["second"]:.0f} & {oracle["second"]:.0f}'
                f' & {small_interval(oracle, 0)} & {small_interval(r["ngc_change"], 0)} \\\\'
            )


def stacked(record):
    if record['ci95'] is None:
        delta = sign_text(record['delta'], 1)
        return f'\\shortstack{{{record["second"]:.1f}\\\\{{\\scriptsize${delta}$}}}}'
    lo, hi = record['ci95']
    delta, lo, hi = (sign_text(v, 1) for v in (record['delta'], lo, hi))
    return f'\\shortstack{{{record["second"]:.1f}\\\\{{\\scriptsize${delta}\\,[{lo},{hi}]$}}}}'


def graphsens_rows(table):
    print(
        '% tab:graphsens rows (task, offset, L2, default, resample 1, resample 2, union)'
    )
    for task, name in PAPER_TASKS:
        for offset, r in table[task].items():
            variants = [
                stacked(r['variants'][k])
                for k in (
                    *(f'nodes{s}' for s in rerun.REDRAW_NODE_SEEDS),
                    rerun.RULE_VARIANT,
                )
            ]
            print(
                f'{name} & {offset} & {r["l2"]:.1f} & {r["reference"]:.1f} & '
                + ' & '.join(variants)
                + ' \\\\'
            )
        print('\\midrule')


def graphk_rows(table):
    tasks = [task for task, _ in PAPER_TASKS]
    for block in (tasks[:2], tasks[2:]):
        columns = [(task, offset) for task in block for offset in table[task]]
        print(
            '% tab:graphk block: '
            + ', '.join(f'{task} {offset}' for task, offset in columns)
        )
        print(
            'L2 & '
            + ' & '.join(f'{table[t][o]["l2"]:.1f}' for t, o in columns)
            + ' \\\\'
        )
        print(
            'Default & '
            + ' & '.join(f'{table[t][o]["reference"]:.1f}' for t, o in columns)
            + ' \\\\'
        )
        print('\\midrule')
        for fraction, label in zip(rerun.GRAPH_FRACTIONS, ('18', '14', '12')):
            cells = []
            for t, o in columns:
                count = int(rerun.NODE_COUNT[t] * fraction)
                cells.append(stacked(table[t][o]['variants'][f'n{count}']))
            print(
                f'$\\tfrac{label}$ of the nodes & '
                + ' & '.join(cells)
                + ' \\\\'
            )
        print('\\midrule')
        for k in rerun.GRAPH_NEIGHBORS:
            cells = [
                stacked(table[t][o]['variants'][f'k{k}']) for t, o in columns
            ]
            print(f'$k={k}$ & ' + ' & '.join(cells) + ' \\\\')
        print('% bridges (inter-component edges) per arm and column:')
        for t, o in columns:
            print(
                f'%  {t} {o}: '
                + ', '.join(
                    f'{k} {len(v)}' for k, v in table[t][o]['bridges'].items()
                )
            )


def efficiency_rows(table):
    print(
        '% tab:efficiency rows (task, L2 replan s, NGC replan s (ratio), evaluation ratio, L2 ms, NGC ms); replans in the comment'
    )
    for task, name in PAPER_TASKS:
        r = table[task]
        print(
            f'{name} & {r["replan_s"]["l2"]:.2f} & {r["replan_s"]["ngc"]:.2f} {{\\scriptsize(${r["replan_ratio"]:.2f}\\times$)}} & '
            f'{r["solve_time_ratio"]:.2f} & {r["objective_ms"]["l2"]:.3f} & {r["objective_ms"]["ngc"]:.2f} \\\\'
            f'  % replans L2 {r["replans"]["l2"]}, NGC {r["replans"]["ngc"]}; {r["device"]}'
        )


def budget_text(table):
    print('% sec:efficiency and fig:budget numbers')
    for task, name in PAPER_TASKS:
        r = table[task]
        print(f'% {name} (offset {r["offset"]}):')
        for arm in ('l2', 'ngc'):
            moves = ', '.join(
                f'{m}x {interval(r["vs_default"][m][arm])}'
                for m in r['multipliers']
                if m != '1'
            )
            print(f'%   {arm} vs 1x: {moves}')
        print(
            '%   NGC 1x minus L2 at: '
            + ', '.join(
                f'{m}x {interval(r["ngc_default_vs_l2"][m])}'
                for m in r['multipliers']
            )
        )
        print(
            '%   NGC minus L2 at: '
            + ', '.join(
                f'{m}x {r["ngc_vs_l2"][m]["delta"]:+.1f}'
                for m in r['multipliers']
            )
        )


def strata_text():
    print(
        '% fig:success bottom row: pooled quintiles (median distance, L2, NGC, delta [CI])'
    )
    for task, name in PAPER_TASKS:
        s = rerun.strata(task)
        rows = ', '.join(
            f'{b["median"]:.3g}: {b["first"]:.0f}/{b["second"]:.0f} {b["delta"]:+.1f} {b["ci95"]}'
            for b in s['pooled']
        )
        print(f'% {name} ({s["unit"]}): {rows}')


def geometry_text():
    print(
        '% fig:costtruth: Spearman of each cost with the physical distance (study pair seed), and the union rule'
    )
    for task, name in PAPER_TASKS:
        g = rerun.geometry(task, 'project', rerun.STUDY_PAIR_SEED)
        u = rerun.geometry(task, 'project-union', rerun.STUDY_PAIR_SEED)
        print(
            f'% {name}: L2 {g["stats"]["l2"]["spearman"]:.2f}, NGC {g["stats"]["ngc"]["spearman"]:.2f} '
            f'(union {u["stats"]["ngc"]["spearman"]:.2f}); longest edge {g["edge_lengths"]["max"]:.3g} '
            f'(union {u["edge_lengths"]["max"]:.3g}) {g["unit"]}; bridges {len(g["bridges"])}; '
            f'participation ratio {g["concentration"]["participation_ratio"]:.0f}'
        )
        for seed in rerun.PAIR_SEEDS:
            gs = rerun.geometry(task, 'project', seed)
            print(
                f'%   seed {seed}: L2 {gs["stats"]["l2"]["spearman"]:.2f}, NGC {gs["stats"]["ngc"]["spearman"]:.2f}'
            )


def main():
    ours = rerun.main_ladder()
    success_rows(ours)
    print()
    ablation_rows(rerun.ablation())
    print()
    ckpt_rows(rerun.official_ladder(), ours)
    print()
    seed_rows(
        [rerun.replicate_ladder(s) for s in REPLICATE_TRAINING_SEEDS],
        ours,
    )
    print()
    comparability_rows(rerun.comparability())
    print()
    oracle_rows(rerun.oracle())
    print()
    graphsens_rows(rerun.rule_study())
    print()
    graphk_rows(rerun.graph_study())
    print()
    efficiency_rows(rerun.timing())
    print()
    budget_text(rerun.budget())
    print()
    strata_text()
    print()
    geometry_text()


if __name__ == '__main__':
    main()
