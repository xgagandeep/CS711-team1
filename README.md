
# Predictive Maintenance Model

This repository contains a predictive maintenance solution developed as a team project. The project utilizes machine learning techniques to predict the Remaining Useful Life (RUL) of Jet Engines. The model is built using a neural network to process sensor data and operational settings from jet engines, enabling predictive maintenance decisions.

## Project Structure

- **model.py**: A Streamlit app for user interaction and RUL prediction.
- **train_FD001.txt**: Training data file used for model development.
- **test_FD001.txt**: Testing data file used for model evaluation.
- **RUL_FD001.txt**: Actual Remaining Useful Life data for the test set.
- **CS711_Team1.ipynb**: Jupyter notebook containing the exploratory data analysis, preprocessing, and model training pipeline.

## Prerequisites

Before running the project, ensure that you have the following dependencies installed:

- Python 3.8+
- TensorFlow 2.0+
- Keras
- Pandas
- Scikit-learn
- Streamlit

To install the required dependencies, run:

```bash
pip install -r requirements.txt
```

## How to Run

### 1. Clone the Repository:

```bash
git clone https://github.com/xgagandeep/Predictive-Maintenance-Model.git
```

### 2. Run the Streamlit App:

```bash
streamlit run model.py
```

The Streamlit app provides an interactive interface where users can input machinery operational settings and sensor readings. The model will then output the predicted Remaining Useful Life (RUL) for the given inputs.

### 3. Input and Interact with the App:

- Enter operational settings (`op_setting1`, `op_setting2`, `op_setting3`).
- Enter sensor readings (`s1` to `s21`).
- The model will compute and display the predicted RUL on the Streamlit interface.

The app provides a user-friendly way to interact with the predictive maintenance model.

## Dataset

The dataset used for training and testing comes from NASA’s Prognostics Center of Excellence. The data consists of multiple time-series sensor readings and operational settings for machinery components.

### Data Fields:

- `unit`: Unique identifier for each machine.
- `cycles`: Number of cycles the machine has undergone.
- `op_setting1`, `op_setting2`, `op_setting3`: Operational settings of the machine.
- `s1` to `s21`: Sensor readings from the machine.
- `rul`: Remaining Useful Life (calculated in preprocessing).

## Model Architecture

We developed an Artificial Neural Network (ANN) using Keras. The architecture is as follows:

- **Input Layer**: 16 features (sensor readings and operational settings).
- **Hidden Layers**:
  - Dense layer with 34 neurons and ReLU activation.
  - Two Dense layers with 17 neurons and ReLU activation.
- **Output Layer**: 1 neuron predicting the RUL.

## Preprocessing Steps

1. **Data Cleaning**: Removed columns with NaN values and low variance.
2. **Feature Engineering**:
   - Calculated the Remaining Useful Life (RUL) by finding the difference between the maximum cycles for each unit and the current cycles.
3. **Normalization**: Applied Min-Max scaling to the operational settings and sensor readings.
4. **Dimensionality Reduction**: Removed highly correlated features (threshold = 0.9).

## Performance Metrics

- **Loss**: Mean Squared Error (MSE) was used to evaluate the model's performance during training.
- **Accuracy**: Tracked but not the main evaluation metric.

## Usage

Once the model is deployed, users can input operational settings and sensor readings, and the model will predict how many cycles the machine has left before it fails (RUL).

## Future Improvements

- Improve model accuracy with more advanced architectures such as LSTM or GRU.
- Explore additional preprocessing steps to enhance data quality.
- Deploy the app as a web service for real-time predictive maintenance.


## Acknowledgments

This project was developed as part of the **CS711** course at the University, with contributions from team members.
