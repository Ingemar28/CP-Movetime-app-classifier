import os
import struct
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from rf_function import extract_acc_fet
import math

def read_binary_file(file_path):
    # Check if input file is a binary file
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
    
    df['datetime'] = pd.to_datetime(df['datetime'], unit='ms').dt.tz_localize('UTC').dt.tz_convert('Australia/Perth')
    df['datetime'] = df['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S.%f').str[:-3]
    
    # Scale acc data
    df[['x', 'y', 'z']] = df[['x', 'y', 'z']] * 0.0078125

    # df = df.replace([np.inf, -np.inf], np.nan)
    # df = df.fillna(0)

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
    data, timestamps, xyz_means, xyz_std = [], [], [], []
    feature_names = None

    for start in range(0, len(df) - window_size + 1, step_size):
        segment = df.iloc[start:start+window_size]
        segment_data = segment[['x', 'y', 'z']]
        features_series = extract_acc_fet(segment_data, Hz=frequency)
        data.append(features_series.values)
        timestamps.append(segment.iloc[0]['datetime'])  # Use the timestamp of the first row in the segment
        xyz_means.append(segment_data.mean().values)  # Calculate mean values for x, y, z
        xyz_std.append(segment_data.std().values)  # Calculate mean values for x, y, z

        if not feature_names:
            feature_names = features_series.index.to_list()

    return np.array(data), timestamps, xyz_means, xyz_std, feature_names

def detect_non_wear_periods(df, sd_threshold, window_size, boutlength, short_wear_threshold, frequency):
    # Label windows with low standard deviation as non-wear (1)
    non_wear_windows = (df['x_std'] < sd_threshold) & (df['y_std'] < sd_threshold) & (df['z_std'] < sd_threshold)

    non_wear_windows = non_wear_windows.astype(int)

    # Perform run-length encoding to identify contiguous sequences
    rle_pairs = []
    prev_value = non_wear_windows.iloc[0]
    count = 0
    for value in non_wear_windows:
        if value == prev_value:
            count += 1
        else:
            rle_pairs.append((count, prev_value))
            prev_value = value
            count = 1
    rle_pairs.append((count, prev_value))  # Append the last run

    # Merge non-wear periods interrupted by short wear periods
    boutlength_windows = boutlength / 4  # Minimum number of winodws 
    merged_rle_pairs = []
    i = 0
    while i < len(rle_pairs):
        length, value = rle_pairs[i]
        if value == 0 and length < short_wear_threshold:
            # Merge with previous and next non-wear periods if wear period is shorter than threshold
            if i > 0 and i < len(rle_pairs) - 1:
                prev_length, prev_value = merged_rle_pairs[-1]
                next_length, next_value = rle_pairs[i + 1]
                # merge only when the wear period is surrunded with non-wear period that are long enough
                if prev_value == 1 and next_value == 1 and prev_length + next_length >= boutlength_windows:
                    merged_rle_pairs.pop()
                    merged_rle_pairs.append((prev_length + length + next_length, 1))

                    i += 2
                    continue
            merged_rle_pairs.append((length, value))
        else:
            merged_rle_pairs.append((length, value))
        i += 1

    # Initialize filtered_wear based on merged_rle_pairs
    filtered_wear = np.zeros(len(df), dtype=int)  # set all to 0 (wear)
    idx = 0
    for length, value in merged_rle_pairs:
        if value == 1 and length >= boutlength_windows:
            filtered_wear[idx:idx + length] = 1  # label non-wear as 1
        idx += length

    return filtered_wear

def main(input_file):
    frequency = 12.5
    window_size = int(frequency * 8)  # 8 second window (100 data)
    step_size = int(frequency * 4)  # 4 second window (50% overlap)
    window_smoothing_size = 3  # this is determined by trained rf model
    sd_threshold = 0.01  # Standard deviation threshold
    boutlength = 45 * 60  # Minimum non-wear period length in seconds
    short_wear_threshold = 3  # number of 4-second windows

    # Label encoder for predictions, exact same as trained model
    all_class_names = ['Cycle', 'Sit', 'Stand / SUM', 'Walk']
    label_encoder = LabelEncoder()
    label_encoder.fit(all_class_names)

    # Read the binary file and convert to DataFrame
    df = read_binary_file(input_file)
    
    # Preprocess the data
    df = smoothing_data(df, window_smoothing_size)

    # Segment data with features
    feature_data, timestamps, xyz_means, xyz_std, feature_names = window_features(df, window_size, step_size, frequency)

    # Predict the segemnted data
    with open('../models/rf_model.pkl', 'rb') as file:
        random_forest_model = pickle.load(file)

    preds = random_forest_model.predict(feature_data)
    preds_labels = label_encoder.inverse_transform(preds)  # Convert numeric predictions to string labels

    # Prepare the output DataFrame
    output_df = pd.DataFrame(feature_data, columns=feature_names)
    output_df.insert(0, 'datetime_window', timestamps)  # Insert timestamps as the first column
    output_df.insert(1, 'x_mean', [m[0] for m in xyz_means])
    output_df.insert(2, 'y_mean', [m[1] for m in xyz_means])
    output_df.insert(3, 'z_mean', [m[2] for m in xyz_means])
    output_df['prediction'] = preds_labels
    output_df.insert(output_df.shape[1], 'x_std', [std[0] for std in xyz_std])
    output_df.insert(output_df.shape[1], 'y_std', [std[1] for std in xyz_std])
    output_df.insert(output_df.shape[1], 'z_std', [std[2] for std in xyz_std])

    # print(f"before detecting {len(output_df)}")

    # Detect non-wear periods using std values
    non_wear_periods = detect_non_wear_periods(output_df, sd_threshold, window_size, boutlength, short_wear_threshold, frequency)
    output_df = output_df[non_wear_periods == 0]

    # print(f"after detecting {len(output_df)}")

    # Remove std columns before saving
    output_df.drop(columns=['x_std', 'y_std', 'z_std'], inplace=True)

    # Save the output DataFrame to CSV
    output_file = '../output_20230032-01_acc_27_11_2023.csv'
    output_df.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")


if __name__ == "__main__":
    # input_file = input("Please enter the path to your input binary file: ")
    input_file = '/Users/ingemar/Library/Mobile Documents/com~apple~CloudDocs/Desktop/UQ/acce_model/datasets/bin_files/export_20230032-01_acc_27_11_2023.bin'
    main(input_file)