"""Adds a qualitative 'Purpose' column to generated/table_cumulative.tex for the main text."""
import os
GEN = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'paper', 'generated')
PURPOSE = [
    'graph fixed before training',
    'graph learned with prediction',
    'feature groups; scales with $d$',
    'regularization',
    'next model fits the errors',
    'weights adapt to $d,n,C$',
    'same predictions',
]
rows = [l.rstrip().rstrip('\\').rstrip() for l in open(os.path.join(GEN, 'table_cumulative.tex')) if '&' in l]
assert len(rows) == len(PURPOSE)
out = []
for r, p in zip(rows, PURPOSE):
    c = [x.strip() for x in r.split('&')]
    out.append(' & '.join(c + [p]) + ' \\\\')
open(os.path.join(GEN, 'table_cumulative_wide.tex'), 'w').write('\n'.join(out) + '\n')
