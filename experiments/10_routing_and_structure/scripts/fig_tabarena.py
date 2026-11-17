"""TabArena ELO bar chart for the main text, from generated/table_tabarena.tex."""
import os, re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
GEN = os.path.join(HERE, '..', '..', '..', 'paper', 'generated')
rows = []
for line in open(os.path.join(GEN, 'table_tabarena.tex')):
    parts = [p.strip() for p in line.replace('\\\\', '').split('&')]
    if len(parts) < 3:
        continue
    name = re.sub(r'\\textbf\{(.*)\}', r'\1', parts[1])
    pre = '$^\\dagger$' in name
    name = name.replace('$^\\dagger$', '').replace('(k=32)', '').strip()
    if 'auto' in name:
        continue  # auto-k is reported in the appendix table only
    rows.append((name, int(parts[2]), pre))
rows = rows[::-1]
fig, ax = plt.subplots(figsize=(3.4, 2.05))
for i, (n, e, pre) in enumerate(rows):
    if n.startswith('Cabin'):
        c = '#c0392b' if 'auto' not in n else '#e8a49c'
    else:
        c = '#9aa5b1' if pre else '#2c6fa8'
    ax.barh(i, e - 900, left=900, color=c, height=0.75)
    ax.text(e + 6, i, str(e), va='center', fontsize=6)
ax.set_yticks(range(len(rows)))
ax.set_yticklabels([n for n, _, _ in rows], fontsize=6.5)
for t, (n, _, _) in zip(ax.get_yticklabels(), rows):
    if n == 'Cabin':
        t.set_fontweight('bold')
ax.set_xlim(900, 1580)
ax.set_xlabel('ELO (default configurations)', fontsize=6.5)
ax.tick_params(axis='x', labelsize=6.5)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color='#9aa5b1', label='pre-trained in-context'),
                   Patch(color='#2c6fa8', label='trained from scratch')],
          fontsize=6, loc='lower right', frameon=False)
for s in ('top', 'right'):
    ax.spines[s].set_visible(False)
fig.tight_layout(pad=0.2)
fig.savefig(os.path.join(GEN, 'fig_tabarena.pdf'))
fig.savefig(os.path.join(GEN, 'fig_tabarena.png'), dpi=200)
