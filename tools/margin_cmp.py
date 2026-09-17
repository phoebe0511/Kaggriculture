import json, statistics as st, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

_a = sys.argv[1:3]
BASE, TREAT = (_a + ['qty-base', 'qty-on'])[:2] if len(_a) == 2 else ('qty-base', 'qty-on')

def load(tag):
    return [json.loads(l) for l in open('model/artifacts/%s/train.jsonl' % tag, encoding='utf-8')]

B, T = load(BASE), load(TREAT)
gb = [r for r in B if 'greedy_margin' in r]
gt = [r for r in T if 'greedy_margin' in r]
ib = [r['iter'] for r in gb]; it_ = [r['iter'] for r in gt]
common = sorted(set(ib) & set(it_))
gb = [r for r in gb if r['iter'] in common]
gt = [r for r in gt if r['iter'] in common]
print('評估點 n=%d  %s' % (len(common), common))
n = min(len(B), len(T))
if len(B) != len(T): print('[!] 輪數不同 %d vs %d，訓練側只比前 %d 輪' % (len(B), len(T), n))
B, T = B[:n], T[:n]

print("it  %11s  %11s         diff     b_win  t_win" % (BASE, TREAT))
diffs = []
for a, b in zip(gb, gt):
    d = b['greedy_margin'] - a['greedy_margin']
    diffs.append(d)
    print("%3d  %11.0f  %11.0f  %11.0f   %.2f   %.2f" % (
        a['iter'], a['greedy_margin'], b['greedy_margin'], d,
        a.get('greedy_win', float('nan')), b.get('greedy_win', float('nan'))))

mb = st.mean([r['greedy_margin'] for r in gb])
mt = st.mean([r['greedy_margin'] for r in gt])
md = st.mean(diffs); sd = st.stdev(diffs); se = sd / len(diffs) ** 0.5
print()
print("%d 點平均  %s %.0f   %s %.0f   diff %.0f" % (len(gb), BASE, mb, TREAT, mt, md))
print("配對差 SD %.0f  SE %.0f  t=%.3f  (n=%d, df=%d)" % (sd, se, md / se, len(diffs), len(diffs) - 1))

def slope(rows):
    xs = [r['iter'] for r in rows]; ys = [r['greedy_margin'] for r in rows]
    mx = st.mean(xs); my = st.mean(ys)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)

for tag, rows in ((BASE, gb), (TREAT, gt)):
    best = max(rows, key=lambda r: r['greedy_margin'])
    print("%-9s 起點 it%d %.0f | 終點 it%d %.0f | 最佳 %.0f(it%d) | 斜率 %.1f/輪 | 勝率 %.2f" % (
        tag, rows[0]['iter'], rows[0]['greedy_margin'], rows[-1]['iter'], rows[-1]['greedy_margin'],
        best['greedy_margin'], best['iter'], slope(rows),
        st.mean([r.get('greedy_win', 0) for r in rows])))

print()
print("--- 訓練側（%d 輪平均）---" % n)
keys = ['approx_kl', 'kl_last_epoch', 'clipfrac', 'policy', 'value', 'entropy',
        'explained_var', 'grad_norm', 'epochs_done']
print("%-16s %12s %12s %12s" % ("", BASE, TREAT, "diff"))
for k in keys:
    a = st.mean([r[k] for r in B if k in r]); b = st.mean([r[k] for r in T if k in r])
    print("%-16s %12.5f %12.5f %12.5f" % (k, a, b, b - a))
for k in ['cash_mean', 'opp_cash_mean']:
    a = st.mean([r[k] for r in B]); b = st.mean([r[k] for r in T])
    print("%-16s %12.0f %12.0f %12.0f" % (k, a, b, b - a))
a = st.mean([r['cash_mean'] - r['opp_cash_mean'] for r in B])
b = st.mean([r['cash_mean'] - r['opp_cash_mean'] for r in T])
print("%-16s %12.0f %12.0f %12.0f" % ("取樣差額", a, b, b - a))
print("%-16s %12.2f %12.2f" % ("win_rate", st.mean([r['win_rate'] for r in B]),
                               st.mean([r['win_rate'] for r in T])))
