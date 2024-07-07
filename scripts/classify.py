import os
import math
import struct
import pickle
import numpy as np
import pandas as pd
from scipy.stats import mode
from rf_function import extract_acc_fet
from sklearn.preprocessing import LabelEncoder

def get_params(frequency=12.5):
    # Function to calculate parameters based on frequency.
    params = {
        'frequency': frequency,
        'window_size': int(frequency * 8),  # 8 second window
        'step_size': int(frequency * 4),  # 4 second window (50% overlap)
        'window_smoothing_size': 3,  # Determined by the trained RF model
        'sd_threshold': 0.001,
        'boutlength_windows': int(55 * 60 / 4),  # minimum non-wear period length in windows 
        'short_wear_threshold': int(12 / 4)  # maximum wear period length surrounded by non-wear
    }
    return params

def read_binary_file(file_path):
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
    
    #### remove timezone at the end
    df['datetime'] = pd.to_datetime(df['datetime'], unit='ms').dt.tz_localize('UTC').dt.tz_convert('Australia/Brisbane')
    df['datetime'] = df['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S.%f').str[:-3]
    
    # Scale acc data
    df[['x', 'y', 'z']] = df[['x', 'y', 'z']] * 0.0078125

    return df

def smoothing_data(df, window_smoothing_size):
    # Smooth the raw acc data
    df = df.assign(
        x=df['x'].rolling(window=window_smoothing_size, min_periods=1, center=True).mean(),
        y=df['y'].rolling(window=window_smoothing_size, min_periods=1, center=True).mean(),
        z=df['z'].rolling(window=window_smoothing_size, min_periods=1, center=True).mean()
    )
    return df

def window_features(df, window_size, step_size, frequency):
    # Segment data into windows and extract features.
    data, timestamps, xyz_means, xyz_std = [], [], [], []
    feature_names = None

    for start in range(0, len(df) - window_size + 1, step_size):
        segment = df.iloc[start:start+window_size]
        segment_data = segment[['x', 'y', 'z']]
        features_series = extract_acc_fet(segment_data, Hz=frequency)
        data.append(features_series.values)
        timestamps.append(segment.iloc[0]['datetime'])  # Use the timestamp of the first row in the segment
        xyz_means.append(np.round(segment_data.mean().values, 3))  # Calculate mean values for x, y, z
        xyz_std.append(np.round(segment_data.std().values, 3))  # Calculate std values for x, y, z

        if not feature_names:
            feature_names = features_series.index.to_list()

    return np.array(data), timestamps, xyz_means, xyz_std, feature_names

def lag_lead_smoothing(predictions, lag, lead):
    # Smooth predictions using a lag-lead approach
    smoothed_predictions = []
    for i in range(len(predictions)):
        start = max(0, i - lag)
        end = min(len(predictions), i + lead + 1)
        window_predictions = predictions[start:end]
        smoothed_predictions.append(mode(window_predictions, keepdims=True).mode[0])
    return np.array(smoothed_predictions)

def detect_non_wear_periods(df, sd_threshold, window_size, boutlength_windows, short_wear_threshold, frequency):
    # Detect non-wear periods in the data.

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
                if prev_value == 1 and next_value == 1 and prev_length + next_length >= boutlength_windows / 2:
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

def process_file(input_file_path, output_file_path, random_forest_model, label_encoder, params):
    """
    Process a single binary file and save the results to a CSV file.
    
    Parameters:
        input_file_path (str): Path to the input binary file.
        output_file_path (str): Path to save the output CSV file.
        random_forest_model (RandomForestClassifier): Trained Random Forest model.
        label_encoder (LabelEncoder): Label encoder for predictions.
        params (dict): Dictionary containing parameters.
    """
    print(f"Processing file: {input_file_path}")

    # Read the binary file and convert to DataFrame
    df = read_binary_file(input_file_path)
    
    # Smooth the raw data
    df = smoothing_data(df, params['window_smoothing_size'])

    # Segment data with features
    feature_data, timestamps, xyz_means, xyz_std, feature_names = window_features(df, params['window_size'], params['step_size'], params['frequency'])

    # Predict the segmented data
    preds = random_forest_model.predict(feature_data)
    preds = lag_lead_smoothing(preds, lag=3, lead=3)
    preds_labels = label_encoder.inverse_transform(preds)

    # Prepare the output DataFrame
    output_df = pd.DataFrame(np.round(feature_data, 3), columns=feature_names)
    output_df.insert(0, 'datetime_window', timestamps)
    output_df.insert(1, 'x_mean', [m[0] for m in xyz_means])
    output_df.insert(2, 'y_mean', [m[1] for m in xyz_means])
    output_df.insert(3, 'z_mean', [m[2] for m in xyz_means])
    output_df['prediction'] = preds_labels
    output_df.insert(output_df.shape[1], 'x_std', [std[0] for std in xyz_std])
    output_df.insert(output_df.shape[1], 'y_std', [std[1] for std in xyz_std])
    output_df.insert(output_df.shape[1], 'z_std', [std[2] for std in xyz_std])

    # Detect non-wear periods using std values
    non_wear_periods = detect_non_wear_periods(output_df, params['sd_threshold'], params['window_size'], params['boutlength_windows'], params['short_wear_threshold'], params['frequency'])
    output_df.loc[non_wear_periods == 1, 'prediction'] = 'non-wear'

    # print(f"After detecting non-wear periods: {len(output_df[output_df['prediction'] == 'non-wear'])}")

    # Add column for upright vs non-upright
    upright_activities = ['Stand / SUM', 'Walk']
    output_df['body_position'] = output_df['prediction'].apply(lambda x: 'upright' if x in upright_activities else 'non-upright')

    # Add column to indicate changes between upright and non-upright
    output_df['change'] = 'no change'
    body_position_shifted = output_df['body_position'].shift(-1)
    output_df.loc[(output_df['body_position'] == 'non-upright') & (body_position_shifted == 'upright'), 'change'] = 'stand up'
    output_df.loc[(output_df['body_position'] == 'upright') & (body_position_shifted == 'non-upright'), 'change'] = 'sit down'

    # Remove std columns before saving
    output_df.drop(columns=['x_std', 'y_std', 'z_std'], inplace=True)

    # Save the output DataFrame to CSV
    output_df.to_csv(output_file_path, index=False)
    print(f"Predictions saved to {output_file_path}")

def main():
    input_folder = input("Please enter the path to your input folder containing .bin files: ")
    output_folder = input("Please enter the path to your output folder: ")

    if not os.path.exists(input_folder):
        print(f"Input folder '{input_folder}' does not exist. Please provide a valid path.")
        return

    if not os.path.exists(output_folder):
        print(f"Output folder '{output_folder}' does not exist. Please provide a valid path.")
        return

    # Get parameters
    params = get_params()

    all_class_names = ['Cycle', 'Sit', 'Stand / SUM', 'Walk']
    label_encoder = LabelEncoder()
    label_encoder.fit(all_class_names)

    with open('models/rf_model.pkl', 'rb') as file:
        random_forest_model = pickle.load(file)

    for file_name in os.listdir(input_folder):
        if file_name.lower().endswith('.bin'):
            input_file_path = os.path.join(input_folder, file_name)
            output_file_path = os.path.join(output_folder, f'processed_{file_name}.csv')
            process_file(input_file_path, output_file_path, random_forest_model, label_encoder, params)

if __name__ == "__main__":
    main()