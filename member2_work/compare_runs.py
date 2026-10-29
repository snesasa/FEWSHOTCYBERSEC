import os, pandas as pd

results_dir    = os.path.join(os.path.dirname(__file__), 'outputs', 'results')
checkpoints_dir = os.path.join(os.path.dirname(__file__), 'outputs', 'checkpoints')

print('=== COMPLETED TRAINING RUNS ===')
rows = []
for f in sorted(os.listdir(results_dir)):
    if f.startswith('training_log') and f.endswith('.csv'):
        df  = pd.read_csv(os.path.join(results_dir, f))
        tag = f.replace('training_log_','').replace('.csv','')
        best = df.loc[df['val_acc'].idxmax()]
        rows.append({'run': tag, 'best_val_acc': best['val_acc'], 'at_epoch': int(best['epoch'])})

for r in rows:
    print(f"  {r['run']:20s}  best_val_acc={r['best_val_acc']:.4f}  epoch={r['at_epoch']}")

print()
print('=== SAVED CHECKPOINTS ===')
for f in sorted(os.listdir(checkpoints_dir)):
    mb = os.path.getsize(os.path.join(checkpoints_dir, f)) / 1024 / 1024
    print(f'  {f}  ({mb:.1f} MB)')
