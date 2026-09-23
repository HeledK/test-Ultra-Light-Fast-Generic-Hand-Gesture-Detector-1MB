import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(r"C:\Users\Heled\Ultra-Light-Fast-Generic-Face-Detector-1MB\runs\detect\runs\hagrid\yolo11n_320_scratch-3\results.csv")
df.columns = df.columns.str.strip()  # ultralytics adds whitespace to col names

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
cols_to_plot = [
    'train/box_loss', 'train/cls_loss', 'train/dfl_loss',
    'metrics/precision(B)', 'metrics/recall(B)',
    'metrics/mAP50(B)', 'metrics/mAP50-95(B)',
    'val/box_loss'
]
for ax, col in zip(axes.flat, cols_to_plot):
    if col in df.columns:
        ax.plot(df[col])
        ax.set_title(col)
        ax.set_xlabel('epoch')
plt.tight_layout()
plt.savefig('results_plot.png', dpi=120)
plt.show()
print("Saved results_plot.png") 