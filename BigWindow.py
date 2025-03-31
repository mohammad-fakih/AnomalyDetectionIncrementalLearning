import pandas as pd
import joblib
import numpy as np
def get_data(n_splits=10):
    # Load the NumPy array from the file
    data = joblib.load('_highlycorrfeaturesremovedandscaleddown.pkl')
    remained_cols = joblib.load('_remainedcolumns.pkl')
 
    # Edge case for n_splits = 1
    if n_splits == 1:
        return data
    # Calculate split size
    split_size = len(data) // n_splits

    # Split the data
    splits = [data.iloc[i * split_size: (i + 1) * split_size] for i in range(n_splits)]

    # Handle any remaining rows (if len(data) % n_splits != 0)
    if len(data) % n_splits != 0:
        splits[-1] = pd.concat([splits[-1], data.iloc[n_splits * split_size:]])
    return splits
def split_data(data,  n_splits=10):
    # Load the NumPy array from the file
    #data
    # Edge case for n_splits = 1
    if n_splits == 1:
        return data
    # Calculate split size
    split_size = len(data) // n_splits

    # Split the data
    splits = [data.iloc[i * split_size: (i + 1) * split_size] for i in range(n_splits)]

    # Handle any remaining rows (if len(data) % n_splits != 0)
    if len(data) % n_splits != 0:
        splits[-1] = pd.concat([splits[-1], data.iloc[n_splits * split_size:]])
    return splits
## Splitting the data splits to windows 
def generate_datasets_for_training2(data, window_size=300, stride=100):
    _l = len(data)  # Length of the DataFrame
    print(f'then length of the data is {_l}')
    Xs = []
    
    # Loop through the data with the specified stride
    for i in range(0, _l - window_size + 1, stride):
        # Use iloc to slice rows from the DataFrame
        Xs.append(data.iloc[i:i + window_size].values)
    # Convert the list of windows into a numpy array
    Xs = np.array(Xs)
    
    return (Xs.shape[0], Xs)

## Splitting the data splits to windows 
def generate_datasets_for_labels(labels, window_size=300, stride=300):
    _l = len(labels)  # Length of the DataFrame

    Xs = []
    
    # Loop through the data with the specified stride
    for i in range(0, _l - window_size + 1, stride):
        # Use iloc to slice rows from the DataFrame
        Xs.append(labels[i:i + window_size])
   
    return (Xs)



def dataEngineeringfor_anomalies(mixed_data, scaler, to_drop, actuators_NAMES):
    # Load data 
    mixed_data.set_index('Timestamp', inplace=True)

    # Remove the last column
    mixed_data = mixed_data.iloc[:, :-1]

    # Drop highly correlated features
    mixed_data.drop(columns=to_drop, inplace=True)

    # Filter actuator names that are still in the dataset
    actuators_NAMES = [col for col in actuators_NAMES if col in mixed_data.columns]

    # Separate sensors and actuators
    sensors = mixed_data.drop(columns=actuators_NAMES)
    sens_cols = sensors.columns
    actuators = mixed_data[actuators_NAMES]

    # Fit and transform the sensors data using the scaler
    scaler.fit(sensors)
    sensors = scaler.transform(sensors)

    # Convert normalized data back to a DataFrame
    sensors = pd.DataFrame(sensors, columns=sens_cols)

    # Create one-hot encoded dummies for actuators
    actuators_dummies = actuators.copy()
    for actuator in actuators_NAMES:
        actuators_dummies[actuator] = pd.Categorical(actuators_dummies[actuator], categories=[0, 1, 2])
        actuators_dummies = pd.get_dummies(actuators_dummies, columns=[actuator], dtype=int)

    # Ensure index consistency
    sensors.index = actuators_dummies.index

    # Concatenate sensors and actuators
    allData = pd.concat([sensors, actuators_dummies], axis=1)

    return allData

def dataEngineeringfor_anomalies2(mixed_data, scaler, to_drop):
    # Ensure 'Timestamp' column exists before setting it as index
    if 'Timestamp' in mixed_data.columns:
        mixed_data.set_index('Timestamp', inplace=True)

    # Remove the last column
    mixed_data = mixed_data.iloc[:, :-1]

    # Drop only the existing columns to avoid errors
    mixed_data.drop(columns=[col for col in to_drop if col in mixed_data], inplace=True)

    # Fit and transform the data using the scaler
    mixed_data_scaled = scaler.fit_transform(mixed_data)

    # Convert back to a DataFrame to maintain structure
    mixed_data = pd.DataFrame(mixed_data_scaled, columns=mixed_data.columns, index=mixed_data.index)

    return mixed_data

def Process_mixed(pathtodata, scaler, to_drop, actuators_NAMES):
    data=pd.read_csv(pathtodata)
    labels=labelsHelper(data)
    return dataEngineeringfor_anomalies(data, scaler, to_drop, actuators_NAMES) , labels

def Process_mixed2(pathtodata, scaler, to_drop):
    data=pd.read_csv(pathtodata)
    labels=labelsHelper(data)
    return dataEngineeringfor_anomalies2(data, scaler, to_drop ) , labels


def labelsHelper (df): 
    labels= df['Normal/Attack']
    #now let us consider the a ttack and the attack the same (let us replace them by 1)
    #and we will replace the normal with a zero
    labels= labels.replace({'Normal': 0, 'Attack': 1, 'A ttack':1})
    return labels 

def get_all_normal_data():
    return joblib.load('_highlycorrfeaturesremovedandscaleddown.pkl')