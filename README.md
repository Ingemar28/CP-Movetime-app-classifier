# CP-Movetime-app-classifier

This repository contains the code for processing and classifying accelerometer data from binary files. The script processes raw accelerometer data, extracts features, and classifies the data using a trained Random Forest model. It also detects non-wear periods and labels changes between upright and non-upright body positions.

## Setup

1. Clone the repository:

    ```bash
    git clone https://github.com/Ingemar28/CP-Movetime-app-classifier.git
    cd CP-Movetime-app-classifier
    ```

2. Install the required Python packages:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Prepare your input data:
    - Place your binary `.bin` files in an input folder.
    - Ensure that the folder paths are correct when prompted by the script.

2. Run the script:

    ```bash
    cd scripts/
    python classify.py
    ```

3. The script will prompt you to enter the paths to your input and output folders:

    ```text
    Please enter the path to your input folder containing .bin files: <path_to_input_folder>
    Please enter the path to your output folder: <path_to_output_folder>
    ```

4. The script will process each `.bin` file in the input folder, extract features, and classify the data. The results will be saved as CSV files in the output folder.

## Detailed Steps in the Script

1. **Read Binary Files**:
    - The `read_binary_file` function reads binary files and converts them to a pandas DataFrame.

2. **Smooth the Data**:
    - The `smoothing_data` function smooths the accelerometer data using a rolling mean.

3. **Segment Data and Extract Features**:
    - The `window_features` function segments the data into windows and extracts features using the `extract_acc_fet` function.

4. **Predict Using the Trained Model**:
    - The script uses a pre-trained Random Forest model to predict the activity for each window of data.

5. **Detect Non-Wear Periods**:
    - The `detect_non_wear_periods` function identifies non-wear periods based on the standard deviation of the accelerometer data.

6. **Label Changes Between Upright and Non-Upright Positions**:
    - The script labels changes in body position (e.g., standing up, sitting down) based on the predicted activities.

7. **Save the Results**:
    - The results are saved to CSV files in the specified output folder.