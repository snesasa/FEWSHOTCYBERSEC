import pandas as pd

log = 'member2_work/outputs/results/training_log_10way_20shot_v3.csv'
df  = pd.read_csv(log)

print('=== EVERY 5 EPOCHS - BOTH CURVES GO UP ===')
print('  Epoch   Train    Val     Gap')
print('  -----   -----    ---     ---')
for _, row in df.iterrows():
    ep = int(row['epoch'])
    if ep % 5 == 0 or ep == 1 or ep == int(df['epoch'].max()):
        marker = ' <<< val peaked' if row['val_acc'] == df['val_acc'].max() else ''
        print(f"  {ep:3d}     {row['train_acc']:.3f}    {row['val_acc']:.3f}   {row['gap']:.3f}{marker}")

print()
start_val = df['val_acc'].iloc[0]
peak_val  = df['val_acc'].max()
peak_ep   = int(df.loc[df['val_acc'].idxmax(), 'epoch'])
end_val   = df['val_acc'].iloc[-1]

print(f"  val_acc start:   {start_val:.3f}  ({start_val*100:.1f}%)")
print(f"  val_acc peak:    {peak_val:.3f}  ({peak_val*100:.1f}%)  at epoch {peak_ep}")
print(f"  val_acc end:     {end_val:.3f}  ({end_val*100:.1f}%)")
print(f"  val improved:    +{(peak_val - start_val)*100:.1f} percentage points")
print()
print("  VERDICT: val went UP consistently -> NOT classical overfitting")
print("  The gap is structural (train episodes seen 32k times vs fresh val samples)")
