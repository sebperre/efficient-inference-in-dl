import pandas as pd

csv_path = "../imagenette/noisy_imagenette.csv"

df = pd.read_csv(csv_path)

if "noisy_labels_0" in df.columns:
    value_counts = df["noisy_labels_0"].value_counts()
    print("Distinct elements and their counts in noisy_labels_0:")
    print(value_counts)
else:
    print("Column noisy_labels_0 not found in the CSV.")