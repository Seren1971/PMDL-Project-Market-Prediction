import pandas as pd, numpy as np

tr = pd.read_csv('data/raw/train.csv')
cal = pd.read_csv('data/interim_calendar_estimate.csv')
print("календарь:", cal.shape, cal.columns.tolist())
df = tr.merge(cal, on='date_id', how='left')
dt = pd.to_datetime(df['date_est'])
df['dow'] = dt.dt.dayofweek; df['dom'] = dt.dt.day
df['month'] = dt.dt.month; df['year'] = dt.dt.year
df['ym'] = dt.dt.to_period('M')
# порядковый номер торгового дня в месяце с начала и с конца
df['tdom'] = df.groupby('ym').cumcount() + 1
df['tdom_end'] = df.groupby('ym').cumcount(ascending=False) + 1
# третья пятница месяца
df['nth_fri'] = df[df.dow == 4].groupby('ym').cumcount() + 1

D = [f'D{i}' for i in range(1, 10)]
for c in D:
    v = df[c].abs()
    sel = df[v == 1]
    if len(sel) == 0: print(f"\n{c}: пусто"); continue
    print(f"\n=== {c}  n={len(sel)} ({len(sel)/len(df)*100:.1f}%)  значения={sorted(df[c].unique())}")
    print("  день недели:", sel.dow.value_counts(normalize=True).round(2).sort_index().to_dict())
    print("  торг.день месяца с начала:", sel.tdom.value_counts().head(5).to_dict())
    print("  торг.день месяца с конца :", sel.tdom_end.value_counts().head(5).to_dict())
    print("  месяц:", sel.month.value_counts(normalize=True).round(2).sort_index().to_dict())
    print("  n-я пятница:", sel.nth_fri.value_counts(dropna=True).head(4).to_dict())
    yr = sel.year.value_counts().sort_index()
    print(f"  годы: {yr.index.min()}..{yr.index.max()}, дней/год медиана {yr.median():.0f}")
