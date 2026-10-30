import json, statistics as st

F = {'A': '/tmp/v10_smoke_before_diag.json',
     'B': '/tmp/v10_smoke_after_diag.json',
     'C': '/tmp/v10_smoke_hard_diag.json'}
D = {}
for k, p in F.items():
    try:
        D[k] = {r['sample_id']: r for r in json.load(open(p))['rows']}
    except Exception as e:
        print(f'{k} 读取失败: {e}')
ids = sorted(set(D.get('A', {})) & set(D.get('B', {})) & set(D.get('C', {})))
print(f'共同样本 {len(ids)} 个\n')

def pair(k, m):
    return [D[k][i][m] for i in ids]

for m in ['ring_noise_f', 'ring_lapvar_ratio', 'ring_sat_shift',
          'ring_val_shift', 'ring_delta', 'ring_noise_drop']:
    print(f'--- {m} ---')
    for k in ['A', 'B', 'C']:
        v = pair(k, m)
        print(f'  {k}: mean {st.mean(v):9.3f}  median {st.median(v):9.3f}  '
              f'p10 {sorted(v)[len(v)//10]:8.2f}  p90 {sorted(v)[len(v)*9//10]:9.2f}')
    # 配对胜率 B/C vs A
    for k in ['B', 'C']:
        wins = sum(1 for i in ids if D[k][i][m] < D['A'][i][m])
        print(f'  {k} < A（越低越好）: {wins}/{len(ids)} = {wins/len(ids):.0%}')
    print()

print('=== 逐样本 ring_lapvar_ratio（A / B / C）===')
print(f"{'id':<7}{'A':>11}{'B':>11}{'C':>11}   best")
for i in ids:
    a, b, c = (D[k][i]['ring_lapvar_ratio'] for k in 'ABC')
    best = min([('A', a), ('B', b), ('C', c)], key=lambda t: t[1])[0]
    print(f"{i:<7}{a:>11.1f}{b:>11.1f}{c:>11.1f}   {best}")

print('\n=== 逐样本 ring_delta（越低越好）===')
print(f"{'id':<7}{'A':>8}{'B':>8}{'C':>8}   best")
for i in ids:
    a, b, c = (D[k][i]['ring_delta'] for k in 'ABC')
    best = min([('A', a), ('B', b), ('C', c)], key=lambda t: t[1])[0]
    print(f"{i:<7}{a:>8.1f}{b:>8.1f}{c:>8.1f}   {best}")
