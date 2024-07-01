import os
import struct
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from rf_function import extract_acc_fet
import math

def read_binary_file(file_path):
    # check if input file is binary file
    if not file_path.lower().endswith('.bin'):
        raise ValueError(f"The file {file_path} is not a binary file. Please provide a .bin file.")

    # Read binary file and convert to DataFrame
    s = struct.Struct('>qhhh')
    sz = os.path.getsize(file_path)
    data = np.zeros([int(sz / 12), 4])

    with open(file_path, 'rb') as f:
        for i in range(data.shape[0]):
            d = f.read(12)
            if len(d) == 0:
                break
            data[i, :] = s.unpack(b'\0\0' + d)
    
    df = pd.DataFrame(data, columns=['datetime', 'x', 'y', 'z'])
    return df

def smoothing_data(df, window_smoothing_size):
    # Smooth the data
    df = df.assign(
        x=df['x'].rolling(window=window_smoothing_size, min_periods=1, center=True).mean(),
        y=df['y'].rolling(window=window_smoothing_size, min_periods=1, center=True).mean(),
        z=df['z'].rolling(window=window_smoothing_size, min_periods=1, center=True).mean()
    )
    return df

def window_features(df, window_size, step_size, frequency):
    data, timestamps, xyz_means = [], [], []
    feature_names = None

    for start in range(0, len(df) - window_size + 1, step_size):
        segment = df.iloc[start:start+window_size]
        segment_data = segment[['x', 'y', 'z']]
        features_series = extract_acc_fet(segment_data, Hz=frequency)
        data.append(features_series.values)
        timestamps.append(segment.iloc[0]['datetime'])  # Use the timestamp of the first row in the segment
        xyz_means.append(segment_data.mean().values)  # Calculate mean values for x, y, z
        
        if not feature_names:
            feature_names = features_series.index.to_list()

    return np.array(data), timestamps, xyz_means, feature_names

def main(input_file):
    frequency = 12.5
    window_size = int(frequency * 8)  # 8 second window
    step_size = int(frequency * 4)  # 4 second window (50% overlap)
    window_smoothing_size = 3  # this is determined by trained rf model

    # Label encoder for predictions, exact same as trained model
    all_class_names = ['Cycle', 'Sit', 'Stand / SUM', 'Walk']
    label_encoder = LabelEncoder()
    label_encoder.fit(all_class_names)

    # Read the binary file and convert to DataFrame
    # df = read_binary_file(input_file)

    ###### testing with csv file
    df = pd.read_csv(input_file)
    
    # Preprocess the data
    df = smoothing_data(df, window_smoothing_size)

    # Load and segment data
    data, timestamps, xyz_means, feature_names = window_features(df, window_size, step_size, frequency)

    with open('../models/rf_model.pkl', 'rb') as file:
        random_forest_model = pickle.load(file)

    preds = random_forest_model.predict(data)
    preds_labels = label_encoder.inverse_transform(preds)  # Convert numeric predictions to string labels

    # Prepare the output DataFrame
    output_df = pd.DataFrame(data, columns=feature_names)
    output_df.insert(0, 'datetime_window', timestamps)  # Insert timestamps as the first column
    output_df.insert(1, 'x_mean', [x[0] for x in xyz_means])
    output_df.insert(2, 'y_mean', [x[1] for x in xyz_means])
    output_df.insert(3, 'z_mean', [x[2] for x in xyz_means])
    output_df['prediction'] = preds_labels

    # Save the output DataFrame to CSV
    output_file = 'output_with_predictions.csv'
    output_df.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")


if __name__ == "__main__":
    # input_file = input("Please enter the path to your input binary file: ")

    ##### testing with csv file, replace with other csv file path
    input_file = input("Please enter the path to your input csv file: ")
    # input_file = '/Users/ingemar/Library/Mobile Documents/com~apple~CloudDocs/Desktop/UQ/acce_model/datasets/SENs/20230032-23_acc_24_11_2023_SENs_Raw_labeled.csv'
    main(input_file)
