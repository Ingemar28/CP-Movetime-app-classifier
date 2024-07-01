import os
import pickle
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox
from sklearn.preprocessing import LabelEncoder
from rf_function import extract_acc_fet

# Label encoder for predictions
all_class_names = ['Cycle', 'Sit', 'Stand / SUM', 'Walk']
label_encoder = LabelEncoder()
label_encoder.fit(all_class_names)

def load_and_segment_data_smooth(file_path, window_size, step_size, window_smoothing_size):
    data, timestamps, xyz_means = [], [], []
    feature_names = None

    df = pd.read_csv(file_path)
    df = df.assign(
        x=df['x'].rolling(window=window_smoothing_size, min_periods=1, center=True).mean(),
        y=df['y'].rolling(window=window_smoothing_size, min_periods=1, center=True).mean(),
        z=df['z'].rolling(window=window_smoothing_size, min_periods=1, center=True).mean()
    )

    for start in range(0, len(df) - window_size + 1, step_size):
        segment = df.iloc[start:start+window_size]
        segment_data = segment[['x', 'y', 'z']]
        features_series = extract_four_acc_fet(segment_data, 12.5)
        data.append(features_series.values)
        timestamps.append(segment.iloc[0]['timestamp'])  # Use the timestamp of the first row in the segment
        xyz_means.append(segment_data.mean().values)  # Calculate mean values for x, y, z
        
        if not feature_names:
            feature_names = features_series.index.to_list()

    return np.array(data), timestamps, xyz_means, feature_names

def main(input_file, output_dir, frequency):
    window_size = int(frequency * 8)  # Example: 8 seconds window size
    step_size = int(frequency * 4)    # Example: 4 seconds step size
    window_smoothing_size = 3

    data, timestamps, xyz_means, feature_names = load_and_segment_data_smooth(input_file, window_size, step_size, window_smoothing_size)

    with open('models/rf_fea7.pkl', 'rb') as file:
        random_forest_model = pickle.load(file)

    preds = random_forest_model.predict(data)
    preds_labels = label_encoder.inverse_transform(preds)  # Convert numeric predictions to string labels

    # Prepare the output DataFrame
    output_df = pd.DataFrame(data, columns=feature_names)
    output_df.insert(0, 'timestamp', timestamps)  # Insert timestamps as the first column
    output_df.insert(1, 'x_mean', [x[0] for x in xyz_means])
    output_df.insert(2, 'y_mean', [x[1] for x in xyz_means])
    output_df.insert(3, 'z_mean', [x[2] for x in xyz_means])
    output_df['prediction'] = preds_labels

    output_file = os.path.join(output_dir, 'output_with_predictions.csv')
    output_df.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")

def upload_file():
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if file_path:
        input_path.set(file_path)

def select_output_directory():
    directory = filedialog.askdirectory()
    if directory:
        output_path.set(directory)

def start_processing():
    input_file = input_path.get()
    output_dir = output_path.get()
    frequency = frequency_input.get()

    if not input_file or not output_dir or not frequency:
        messagebox.showerror("Error", "Please fill in all fields")
        return

    try:
        frequency = float(frequency)
    except ValueError:
        messagebox.showerror("Error", "Frequency must be a number")
        return

    main(input_file, output_dir, frequency)

# Set up the GUI
root = tk.Tk()
root.title("Activity Classification")

input_path = tk.StringVar()
output_path = tk.StringVar()

tk.Label(root, text="Input CSV File:").grid(row=0, column=0, padx=10, pady=10)
tk.Entry(root, textvariable=input_path, width=50).grid(row=0, column=1, padx=10, pady=10)
tk.Button(root, text="Browse", command=upload_file).grid(row=0, column=2, padx=10, pady=10)

tk.Label(root, text="Output Directory:").grid(row=1, column=0, padx=10, pady=10)
tk.Entry(root, textvariable=output_path, width=50).grid(row=1, column=1, padx=10, pady=10)
tk.Button(root, text="Browse", command=select_output_directory).grid(row=1, column=2, padx=10, pady=10)

tk.Label(root, text="Frequency (Hz):").grid(row=2, column=0, padx=10, pady=10)
frequency_input = tk.Entry(root)
frequency_input.grid(row=2, column=1, padx=10, pady=10)

tk.Button(root, text="Start", command=start_processing).grid(row=3, column=1, padx=10, pady=10)

root.mainloop()
