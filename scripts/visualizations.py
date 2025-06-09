import os
import pandas as pd
import numpy as np
import wfdb
import matplotlib.pyplot as plt
import seaborn as sns
import ast

# Settings
DATA_DIR = './../data/raw/physionet.org/files/ptb-xl/1.0.3/'
RECORD_DIR = os.path.join(DATA_DIR, 'records500')

# Load metadata
df = pd.read_csv(os.path.join(DATA_DIR, 'ptbxl_database.csv'))
scp_statements = pd.read_csv(os.path.join(DATA_DIR, 'scp_statements.csv'), index_col=0)

# Parse diagnostic labels
def extract_labels(scp_codes_str, threshold=0.0):
    codes = ast.literal_eval(scp_codes_str)
    labels = [label for label, score in codes.items() if score > threshold]
    return labels

df['labels'] = df['scp_codes'].apply(extract_labels)

# Filter for diagnostic statements
diagnostic_statements = scp_statements[scp_statements['diagnostic'] == 1].index.tolist()

df['diagnostic_labels'] = df['labels'].apply(lambda x: [label for label in x if label in diagnostic_statements])

# ----- Summary Info -----
print("Total ECG Records:", len(df))
print("Number of Unique Diagnostic Labels:", len(diagnostic_statements))

# ----- Top Labels Plot -----
from collections import Counter

all_labels = [label for sublist in df['diagnostic_labels'] for label in sublist]
label_counts = Counter(all_labels)
top_labels = label_counts.most_common(15)

labels, counts = zip(*top_labels)
plt.figure(figsize=(10, 5))
sns.barplot(x=counts, y=labels, palette="magma")
plt.title("Top 15 Diagnostic Labels in PTB-XL")
plt.xlabel("Count")
plt.ylabel("Label Code")
plt.tight_layout()
plt.show()

# ----- ECG Signal Plot -----
def plot_ecg(record_name, title='ECG Example'):
    record_path = os.path.join(DATA_DIR, record_name)
    signal, fields = wfdb.rdsamp(record_path)
    plt.figure(figsize=(15, 8))
    for i in range(12):  # 12-lead ECG
        plt.subplot(6, 2, i + 1)
        plt.plot(signal[:, i], linewidth=0.8)
        plt.title(fields['sig_name'][i], fontsize=8)
        plt.xticks([])
        plt.yticks([])
    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

# Show one sample ECG from a common class
sample_idx = df[df['diagnostic_labels'].apply(lambda x: 'NORM' in x)].index[0]
plot_ecg(df.loc[sample_idx, 'filename_hr'], title="Example ECG - Normal Rhythm (NORM)")
