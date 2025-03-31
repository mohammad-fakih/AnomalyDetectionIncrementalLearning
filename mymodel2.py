# Function to build the LSTM model with batch normalization

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import seaborn as sns
from keras.models import Sequential
from keras import metrics
import keras
import tensorflow as tf
import os
#from scikeras.wrappers import KerasRegressor
#from sklearn.model_selection import GridSearchCV
from keras.layers import LSTM, RepeatVector, TimeDistributed, Dense, Dropout, BatchNormalization, LayerNormalization, Bidirectional
import time
from memory_profiler import memory_usage
import psutil
from tensorflow.keras.callbacks import EarlyStopping, Callback
from tensorflow.keras.optimizers import Adam
import GPUtil
import matplotlib.pyplot as plt
import pynvml
from tensorflow.keras.regularizers import l2
from psutil import virtual_memory
def build_model_var_window(windowsize, nboffeatures=69):
    model = Sequential()
    # Encoder part
    model.add(LSTM(40, activation='tanh', input_shape=(windowsize, nboffeatures), return_sequences=True, name='encoder_1'))
    model.add(BatchNormalization())
    model.add(LSTM(20, activation='tanh', return_sequences=True, name='encoder_2'))
    model.add(BatchNormalization())
    model.add(LSTM(10, activation='tanh', return_sequences=False, name='encoder_3'))
    model.add(BatchNormalization())
    model.add(RepeatVector(windowsize, name='encoder_decoder_bridge'))
    # Decoder part
    model.add(LSTM(10, activation='tanh', return_sequences=True, name='decoder_1'))
    model.add(BatchNormalization())
    model.add(LSTM(20, activation='tanh', return_sequences=True, name='decoder_2'))
    model.add(BatchNormalization())
    model.add(LSTM(40, activation='tanh', return_sequences=True, name='decoder_3'))
    model.add(BatchNormalization())
    model.add(TimeDistributed(Dense(nboffeatures, name='output_layer')))
    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

# def build_model_2_var_window(windowsize):
#     model = Sequential()
#     # Encoder part
#     model.add(LSTM(40, activation='tanh', input_shape=(windowsize, 69), return_sequences=True, name='encoder_1'))
#     model.add(LayerNormalization())
#     model.add(LSTM(20, activation='tanh', return_sequences=True, name='encoder_2'))
#     model.add(LayerNormalization())
#     model.add(LSTM(10, activation='tanh', return_sequences=False, name='encoder_3'))
#     model.add(LayerNormalization())
#     model.add(RepeatVector(windowsize, name='encoder_decoder_bridge'))
#     # Decoder part
#     model.add(LSTM(10, activation='tanh', return_sequences=True, name='decoder_1'))
#     model.add(LayerNormalization())
#     model.add(LSTM(20, activation='tanh', return_sequences=True, name='decoder_2'))
#     model.add(LayerNormalization())
#     model.add(LSTM(40, activation='tanh', return_sequences=True, name='decoder_3'))
#     model.add(LayerNormalization())
#     model.add(TimeDistributed(Dense(69, name='output_layer')))
#     # Compile the model
#     model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
#     return model




def build_improved_model(windowsize ,nboffeatures=69):
    model = Sequential()
    
    # Encoder part with Bidirectional LSTM and Dropout
    model.add(Bidirectional(LSTM(nboffeatures, activation='tanh', input_shape=(windowsize, nboffeatures), return_sequences=True, kernel_regularizer=l2(0.01)), name='encoder_1'))
    model.add(Dropout(0.2))
    model.add(LayerNormalization())
    
    model.add(Bidirectional(LSTM(32, activation='tanh', return_sequences=True, kernel_regularizer=l2(0.01)), name='encoder_2'))
    model.add(Dropout(0.2))
    model.add(LayerNormalization())
    
    model.add(LSTM(16, activation='tanh', return_sequences=False, kernel_regularizer=l2(0.01), name='encoder_3'))
    model.add(Dropout(0.2))
    model.add(LayerNormalization())
    
    model.add(RepeatVector(windowsize, name='encoder_decoder_bridge'))
    
    # Decoder part with LSTM and Attention Mechanism (optional)
    model.add(LSTM(16, activation='tanh', return_sequences=True, name='decoder_1'))
    model.add(Dropout(0.2))
    model.add(LayerNormalization())
    
    model.add(LSTM(32, activation='tanh', return_sequences=True, name='decoder_2'))
    model.add(Dropout(0.2))
    model.add(LayerNormalization())
    
    model.add(LSTM(64, activation='tanh', return_sequences=True, name='decoder_3'))
    model.add(Dropout(0.2))
    model.add(LayerNormalization())
    
    model.add(TimeDistributed(Dense(nboffeatures, name='output_layer')))
    model.build(input_shape=(None, windowsize, nboffeatures))
    # Compile the model with learning rate scheduler
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    
    return model

# Function to monitor GPU usage

def get_gpu_stats():
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # Assuming the first GPU (adjust if necessary)
    memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)

    gpu_stats = {
        "gpu_memory_used": memory_info.used / (1024 * 1024),  # Convert bytes to MiB
        "gpu_load": utilization.gpu  # GPU utilization percentage
    }
    pynvml.nvmlShutdown()
    return gpu_stats

def get_total_memory_stats():
    memory = psutil.virtual_memory()
    return {
        "total_memory": memory.total / (1024 * 1024),  # Convert bytes to MiB
        "used_memory": memory.used / (1024 * 1024),    # Convert bytes to MiB
        "available_memory": memory.available / (1024 * 1024)  # Convert bytes to MiB
    }

# Function to train the model on a given split and return history
def train_model_on_split(split, model, early_stop, epochs=100, batch_size=16):
    history = model.fit(
        x=split,
        y=split,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=False,
        validation_split=0.2,
        callbacks=[early_stop]
    )
    return history

# Function to train the model on multiple splits and monitor GPU usage
# def train_model_on_splits(splits, model, splitsindexes , modelsavedname, epochs=100, batch_size=16): 
#     # Early stopping callback
#     early_stop = tf.keras.callbacks.EarlyStopping(
#         monitor='val_loss', min_delta=1e-2, patience=5, verbose=0, mode='auto', restore_best_weights=True
#     )
    
#     # Metrics storage
#     train_losses = []
#     val_losses = []
#     times = []
#     memory_usages = []
#     system_memory_usages = []
#     gpu_memory_usages = []
#     gpu_loads = []


#     # Modify the training loop to include system-wide memory stats
#     for i in splitsindexes:
#         split = splits[i]
#         print(f"Training on split {i + 1}/{len(splitsindexes)}")

#         # Measure time and memory before training
#         start_time = time.time()
#         initial_process_memory = psutil.Process().memory_info().rss / (1024 * 1024)  # Process-specific memory in MiB
#         initial_system_memory = virtual_memory().used / (1024 * 1024)  # System-wide memory in MiB
#         initial_gpu_stats = get_gpu_stats()

#         # Train the model
#         history = train_model_on_split(split, model, early_stop, epochs, batch_size)

#         # Measure time and memory after training
#         end_time = time.time()
#         final_process_memory = psutil.Process().memory_info().rss / (1024 * 1024)
#         final_system_memory = virtual_memory().used / (1024 * 1024)
#         final_gpu_stats = get_gpu_stats()

#         # Calculate elapsed time and memory usage
#         elapsed_time = end_time - start_time
#         process_memory_usage = final_process_memory - initial_process_memory
#         system_memory_usage = final_system_memory - initial_system_memory
#         gpu_memory_used = final_gpu_stats["gpu_memory_used"] - initial_gpu_stats["gpu_memory_used"]
#         gpu_load = final_gpu_stats["gpu_load"]

#         # Debugging
#         # print(f"Initial GPU memory: {initial_gpu_stats['gpu_memory_used']} MiB")
#         # print(f"Final GPU memory: {final_gpu_stats['gpu_memory_used']} MiB")
#         # print(f"Initial Process memory: {initial_process_memory} MiB")
#         # print(f"Final Process memory: {final_process_memory} MiB")
#         # print(f"Initial System-wide memory: {initial_system_memory} MiB")
#         # print(f"Final System-wide memory: {final_system_memory} MiB")

#         # Store metrics
#         times.append(elapsed_time)
#         system_memory_usages
#         memory_usages.append(process_memory_usage)
#         gpu_memory_usages.append(gpu_memory_used)
#         gpu_loads.append(gpu_load)
#         # Store the losses for this split
#         train_losses.extend(history.history['loss'])
#         val_losses.extend(history.history['val_loss'])

#     # Save the model weights
#     model.save_weights(f'{modelsavedname}.h5')

#     # Print the time, memory, and GPU consumption
#     for i, (time_consumed, memory_used, gpu_memory, gpu_load) in enumerate(
#         zip(times, memory_usages, gpu_memory_usages, gpu_loads)
#     ):
#         print(f"Split {i + 1}:")
#         print(f"  Time consumed = {time_consumed:.2f} seconds")
#         print(f"  CPU Memory used = {memory_used:.2f} MiB")
#         print(f"  GPU Memory used = {gpu_memory:.2f} MiB")
#         print(f"  GPU Load = {gpu_load:.2f}%")

#     # Optionally, plot the losses
#     plt.figure(figsize=(10, 6))
#     plt.plot(train_losses, label='Training Loss')
#     plt.plot(val_losses, label='Validation Loss')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.legend()
#     plt.title('Training and Validation Loss')
#     plt.grid()
#     plt.show()

#     return model, train_losses, val_losses

#this is optimized according to deepseek
import gc
def train_model_on_splits(splits, model, splitsindexes, modelsavedname, epochs=100, batch_size=16):
    """Optimized training function with better memory management and performance tracking."""
    
    # Ensure the directory exists
    save_dir = "modelParametersSwat"
    os.makedirs(save_dir, exist_ok=True)

    # Configure callbacks and monitoring
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        min_delta=1e-2, 
        patience=5, 
        verbose=1,  # Changed to 1 for better feedback
        mode='auto', 
        restore_best_weights=True
    )
    
    # Pre-allocate metrics storage
    metrics = {
        'train_losses': [],
        'val_losses': [],
        'times': [],
        'memory_usages': [],
        'system_memory_usages': [],
        'gpu_memory_usages': [],
        'gpu_loads': []
    }

    # Enable memory growth to prevent OOM errors
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    for i in splitsindexes:
        split = splits[i]
        print(f"\nTraining on split {i + 1}/{len(splitsindexes)}")
        print("="*50)

        # Setup monitoring
        start_time = time.perf_counter()  # More precise timing
        process = psutil.Process()
        
        # Get initial stats
        initial_stats = {
            'process_memory': process.memory_info().rss / (1024 * 1024),
            'system_memory': virtual_memory().used / (1024 * 1024),
            'gpu_stats': get_gpu_stats()
        }

        # Train with cleanup
        try:
            history = train_model_on_split(split, model, early_stop, epochs, batch_size)
            
            # Get final stats
            final_stats = {
                'process_memory': process.memory_info().rss / (1024 * 1024),
                'system_memory': virtual_memory().used / (1024 * 1024),
                'gpu_stats': get_gpu_stats(),
                'time': time.perf_counter() - start_time
            }

            # Calculate deltas
            metrics['times'].append(final_stats['time'])
            metrics['memory_usages'].append(final_stats['process_memory'] - initial_stats['process_memory'])
            metrics['system_memory_usages'].append(final_stats['system_memory'] - initial_stats['system_memory'])
            metrics['gpu_memory_usages'].append(
                final_stats['gpu_stats']["gpu_memory_used"] - initial_stats['gpu_stats']["gpu_memory_used"]
            )
            metrics['gpu_loads'].append(final_stats['gpu_stats']["gpu_load"])
            
            # Store losses
            metrics['train_losses'].extend(history.history['loss'])
            metrics['val_losses'].extend(history.history['val_loss'])

            # Save model parameters after each split
            model_save_path = os.path.join(save_dir, f'model_split_{i + 1}.h5')
            model.save_weights(model_save_path)
            print(f"Model parameters saved to {model_save_path}")
            
            # Clear session to free memory
            tf.keras.backend.clear_session()
            gc.collect()
            
        except Exception as e:
            print(f"Error during training on split {i}: {str(e)}")
            tf.keras.backend.clear_session()
            gc.collect()
            continue

        # Immediate feedback per split
        print(f"\nSplit {i + 1} completed in {final_stats['time']:.2f}s")
        print(f"Memory delta: {metrics['memory_usages'][-1]:.2f} MiB")
        print(f"GPU memory delta: {metrics['gpu_memory_usages'][-1]:.2f} MiB")
        print(f"GPU load: {metrics['gpu_loads'][-1]:.2f}%")

    # Save final model weights
    model.save_weights(f'{modelsavedname}.h5')
    print(f"\nFinal model weights saved to {modelsavedname}.h5")

    # Visualization
    plot_training_metrics(metrics)
    
    return model, metrics['train_losses'], metrics['val_losses']


def plot_training_metrics(metrics):
    """Helper function to plot training metrics."""
    plt.figure(figsize=(12, 8))
    
    # Loss plot
    plt.subplot(2, 2, 1)
    plt.plot(metrics['train_losses'], label='Training Loss')
    plt.plot(metrics['val_losses'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.grid()
    
    # Memory plot
    plt.subplot(2, 2, 2)
    plt.plot(metrics['memory_usages'], 'r-', label='Process Memory')
    plt.plot(metrics['system_memory_usages'], 'b-', label='System Memory')
    plt.xlabel('Split')
    plt.ylabel('Memory Usage (MiB)')
    plt.legend()
    plt.title('Memory Consumption')
    plt.grid()
    
    # GPU plot
    plt.subplot(2, 2, 3)
    plt.plot(metrics['gpu_memory_usages'], 'g-', label='GPU Memory')
    plt.xlabel('Split')
    plt.ylabel('GPU Memory (MiB)')
    plt.legend()
    plt.title('GPU Memory Usage')
    plt.grid()
    
    plt.subplot(2, 2, 4)
    plt.plot(metrics['gpu_loads'], 'm-', label='GPU Load')
    plt.xlabel('Split')
    plt.ylabel('GPU Utilization (%)')
    plt.legend()
    plt.title('GPU Load')
    plt.grid()
    
    plt.tight_layout()
    plt.show()

from sklearn.metrics import mean_absolute_error, mean_squared_error

def evaluate_model(model, evaluation_split, batch_size=16, sample_index=0):
    # Evaluate the model
    loss = model.evaluate(evaluation_split, evaluation_split, batch_size=batch_size)
    print(f"Evaluation Loss (MSE): {loss}")

    # Generate predictions
    predictions = model.predict(evaluation_split)

    # Flatten arrays for metric computation
    y_true = evaluation_split.reshape(-1, evaluation_split.shape[-1])
    y_pred = predictions.reshape(-1, predictions.shape[-1])

    # Compute additional metrics
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)


    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Mean Squared Error (MSE): {mse}")

    # Select a range of time steps instead of a single index
    sample_range = range(200, 300)  # Pick a reasonable subset of time steps
    true_values = evaluation_split[sample_range, 0]  # First feature over time
    predicted_values = predictions[sample_range, 0]  # First feature over time

    # Plot true vs predicted sequences for a feature (e.g., first feature)
    plt.figure(figsize=(12, 6))
    plt.plot(true_values[:, 0], label='True', marker='o')
    plt.plot(predicted_values[:, 0], label='Predicted', marker='x')
    plt.legend()
    plt.title(f'True vs Predicted Values (Sample {sample_index})')
    plt.xlabel('Time Step')
    plt.ylabel('Feature Value')
    plt.grid()
    plt.show()

    return mae, mse, loss


def evaluate_model_windowwise(model, evaluation_split, batch_size=16, window_size=10, stride=1): 
    # Evaluate the model
    loss = model.evaluate(evaluation_split, evaluation_split, batch_size=batch_size)
    print(f"Evaluation Loss (MSE): {loss}")

    # Generate predictions
    predictions = model.predict(evaluation_split)

    # Number of windows
    num_windows = evaluation_split.shape[0] 
    window_errors = np.zeros(num_windows)

    # Compute windowwise reconstruction error
    for i in range(num_windows):
        window_true = evaluation_split[i]  # Extract true window
        window_pred = predictions[i]  # Extract predicted window
        
        # Compute MSE for the entire window
        window_errors[i] = np.mean((window_true - window_pred) ** 2)

    # Compute overall metrics based on windowwise errors
    mae = np.mean(np.sqrt(window_errors))  # Use RMSE for better interpretability
    mse = np.mean(window_errors)  # Average window MSE

    print(f"Windowwise Mean Absolute Error (MAE): {mae}")
    print(f"Windowwise Mean Squared Error (MSE): {mse}")
 
    # Plot bar chart of window errors
    plt.figure(figsize=(12, 6))
    plt.bar(range(num_windows), window_errors, color='blue', alpha=0.7)
    plt.xlabel('Window Index')
    plt.ylabel('Reconstruction Error (MSE)')
    plt.title('Bar Chart of Window-wise Reconstruction Errors')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()


    return mae, mse, loss

def evaluate_model_with_pointwise_loss(
    model, evaluation_split, batch_size=16, sample_index=0
):
    """
    Evaluates the model on the provided dataset and computes pointwise reconstruction losses.
    
    Parameters:
        model: The trained model.
        evaluation_split: 3D numpy array of shape (windows, time_steps, features).
        batch_size: Batch size for evaluation.
        sample_index: Index of a sample to visualize true vs predicted values.
        
    Returns:
        Tuple (mae, mse, loss, pointwise_reconstruction_error, max_reconstruction_error, percentile_95_reconstruction_error)
    """
    # Evaluate the model
    loss = model.evaluate(evaluation_split, evaluation_split, batch_size=batch_size)
    print(f"Evaluation Loss (MSE): {loss}")

    # Generate predictions
    predictions = model.predict(evaluation_split)

    # Compute flattened arrays for metric computation
    y_true = evaluation_split.reshape(-1, evaluation_split.shape[-1])
    y_pred = predictions.reshape(-1, predictions.shape[-1])

    # Compute additional metrics
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Mean Squared Error (MSE): {mse}")

    # Compute pointwise reconstruction error (per time step and feature)
    reconstruction_errors = np.mean((evaluation_split - predictions) ** 2, axis=-1)  # Mean over features

    # Calculate the maximum reconstruction error
    max_reconstruction_error = np.max(reconstruction_errors)
    print(f"Maximum Reconstruction Error: {max_reconstruction_error}")

    # Calculate the 95th percentile reconstruction error
    percentile_90_reconstruction_error = np.percentile(reconstruction_errors, 90)
    print(f"95th Percentile Reconstruction Error: {percentile_90_reconstruction_error}")
    # Calculate the 95th percentile reconstruction error
    percentile_95_reconstruction_error = np.percentile(reconstruction_errors, 95)
    print(f"95th Percentile Reconstruction Error: {percentile_95_reconstruction_error}")

    # Flatten the reconstruction errors
    flattened_errors = reconstruction_errors.flatten()

    # Visualization of pointwise reconstruction errors as a bar chart
    plt.figure(figsize=(20, 8))
    plt.bar(range(len(flattened_errors)), flattened_errors, color='blue', alpha=0.7)
    plt.title("Pointwise Reconstruction Error for Each Data Point")
    plt.axhline(y=percentile_95_reconstruction_error, color='red', linestyle='--', label='95th Percentile Threshold')
    plt.xlabel("Data Point Index")
    plt.ylabel("Reconstruction Error")
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

    # Visualization of True vs Predicted values for a sample
    true_values = evaluation_split[sample_index]  # True sequence
    predicted_values = predictions[sample_index]  # Predicted sequence

    plt.figure(figsize=(12, 6))
    plt.plot(true_values[:, 0], label='True', marker='o')  # First feature
    plt.plot(predicted_values[:, 0], label='Predicted', marker='x')  # First feature
    plt.legend()
    plt.title(f"True vs Predicted Values (Sample {sample_index})")
    plt.xlabel("Time Step")
    plt.ylabel("Feature Value")
    plt.grid()
    plt.show()

    return mae, mse, loss, reconstruction_errors, max_reconstruction_error,percentile_90_reconstruction_error, percentile_95_reconstruction_error

import numpy as np


def findanomalies(model, evaluation_split, batch_size=16, threshold=0.1): 

    with tf.device('/GPU:0'):  # Explicitly place on the GPU
        predictions = model.predict(evaluation_split, batch_size=batch_size)

    # Compute pointwise reconstruction error (per time step and feature)
    reconstruction_errors = np.mean((evaluation_split - predictions) ** 2, axis=-1)  # Mean over features

    # Initialize anomalies_flagged as a 2D array (same shape as evaluation_split, without features)
    anomalies_flagged = np.zeros(reconstruction_errors.shape)

    # Flag anomalies based on the threshold for each time step
    anomalies_flagged[reconstruction_errors > threshold] = 1

    return reconstruction_errors, anomalies_flagged

def findanomalies_pointwise(model, evaluation_split, batch_size=16, threshold=0.1): 
    """
    Detects anomalies based on reconstruction errors for windows and pointwise data.

    Parameters:
        model: The trained model (e.g., autoencoder).
        evaluation_split: 3D numpy array (samples, time_steps, features) to evaluate.
        batch_size: Batch size for prediction.
        threshold: Threshold for anomaly detection.

    Returns:
        Tuple containing:
        - Pointwise reconstruction errors (same shape as evaluation_split).
        - Window-level reconstruction errors (average over features for each time step).
        - Binary anomaly flags (same shape as window-level reconstruction errors).
    """
    with tf.device('/GPU:0'):  # Explicitly place computation on GPU
        predictions = model.predict(evaluation_split, batch_size=batch_size)

    # Compute pointwise reconstruction error (per time step and feature)
    pointwise_reconstruction_errors = (evaluation_split - predictions) ** 2

    # Compute window-level reconstruction error (mean over features)
    window_reconstruction_errors = np.mean(pointwise_reconstruction_errors, axis=-1)

    # Initialize anomalies_flagged (binary flags for windows)
    anomalies_flagged = np.zeros(window_reconstruction_errors.shape)

    # Flag anomalies based on the threshold for each time step
    anomalies_flagged[window_reconstruction_errors > threshold] = 1

    return pointwise_reconstruction_errors, window_reconstruction_errors, anomalies_flagged

def find_anomalies_windowwise(model, evaluation_split, batch_size=16, threshold=0.1):
 
    with tf.device('/GPU:0'):  
        predictions = model.predict(evaluation_split.astype(np.float32), batch_size=batch_size)

    window_errors = np.mean((evaluation_split - predictions) ** 2, axis=(1, 2))  # Vectorized MSE
    anomalies_flagged = (window_errors > threshold).astype(int)  # Vectorized anomaly flagging
    
    # Clear GPU memory to prevent OOM issues
    tf.keras.backend.clear_session()
    gc.collect()
    return window_errors, anomalies_flagged
def findanomalies_decision_tree(model, evaluation_split, batch_size=16, threshold=0.1):
    """
    Detects anomalies based on a two-step decision tree:
    1. Check if the window-level reconstruction error exceeds the threshold.
    2. If yes, check if at least half of the points in the window also exceed the threshold.
       If so, classify as anomalous; otherwise, classify as normal.

    Parameters:
        model: The trained model (e.g., autoencoder).
        evaluation_split: 3D numpy array (samples, time_steps, features) to evaluate.
        batch_size: Batch size for prediction.
        threshold: Threshold for anomaly detection.

    Returns:
        Tuple containing:
        - featurewise reconstruction errors (same shape as evaluation_split).
        - dataPt-level reconstruction errors (average over features for each time step).
        - Binary anomaly flags for windows (same shape as window-level reconstruction errors).
    """
    with tf.device('/GPU:0'):  # Explicitly place computation on GPU
        predictions = model.predict(evaluation_split, batch_size=batch_size)

    # Compute featurewise reconstruction error (per time step and feature)
    featuretwise_reconstruction_errors = (evaluation_split - predictions) ** 2
    #the shape o fthe above tensor is (22495, 20, 69)
    print(f'the shape of the featurewise reconstrucrion errors is {featuretwise_reconstruction_errors.shape}')
    print(f'this is a sample{featuretwise_reconstruction_errors}')

    # Compute window wise reconstruction error (mean over features)
    datapointreconstruction_errors = np.mean(featuretwise_reconstruction_errors, axis=-1)
    #the shape of the abover tensor is (22495, 20)
    print(f'the shape of the datapoint reconstrucrion errors is {datapointreconstruction_errors.shape}')
    print(f'this is a sample{datapointreconstruction_errors}')

    ##[[23, 12,2,43], [543, 432, 432], [432, 432, 54, 654] ].... 
    # Initialize anomalies_flagged (binary flags for windows)
    #anomalies_flagged = np.zeros(datapointreconstruction_errors.shape)
    anomalies_flagged= array = np.zeros((22495, 20))
    # Apply decision tree logic
    for i in range(len(datapointreconstruction_errors)):
        #iterating through each window 
        window_errors = datapointreconstruction_errors[i]  
        # window_errors = [4,3,2,4,2,3] 1 window
        print(f'the {i}th window is {window_errors}')
        num_points_exceeding_threshold = np.sum(
                    window_errors > threshold
                )
        total_points=len(window_errors)
        exceeded_half= (num_points_exceeding_threshold >= (total_points / 2))
        # iterating on each data points in the window 
        for j in range(len(window_errors)):

            if window_errors[j] > threshold:  # First check: Is the window reconstruction error above the threshold?
                # Check proportion of pointwise errors exceeding the threshold within the window
                
                # If at least half of the points exceed the threshold, classify as anomalous
                if exceeded_half:
                    anomalies_flagged[i,j] = 1  # Mark the window as anomalous
                else:
                    anomalies_flagged[i, j]=0
            else:
                anomalies_flagged[i,j]=0
    # anomalies_flagged=anomalies_flagged.reshape(-1)

    return featuretwise_reconstruction_errors, datapointreconstruction_errors, anomalies_flagged




import numpy as np

def find_best_threshold(reconstruction_errors: np.ndarray, contamination: float) -> float:

    if not (0 < contamination < 1):
        raise ValueError("Contamination must be between 0 and 1")
    
    # Sort the reconstruction error values
    sorted_errors = np.sort(reconstruction_errors)
    
    # Find the index corresponding to the contamination percentile
    index = int((1 - contamination) * len(sorted_errors))
    
    # Select the threshold
    threshold = sorted_errors[index]
    
    return threshold



from sklearn.metrics import precision_score, recall_score, f1_score,   accuracy_score

def evaluate_performance(true_labels: np.ndarray, predictions: np.ndarray):

    # Calculate TP, FP, TN, FN
    TP = np.sum((true_labels == 1) & (predictions == 1))
    FP = np.sum((true_labels == 0) & (predictions == 1))
    TN = np.sum((true_labels == 0) & (predictions == 0))
    FN = np.sum((true_labels == 1) & (predictions == 0))
    
    # Calculate accuracy, precision, recall, F1 and F2 scores
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions)
    recall = recall_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions)
    # f2 = f2_score(true_labels, predictions)
    
    # Return results in a dictionary
    return {
        "TP": TP,
        "FP": FP,
        "TN": TN,
        "FN": FN,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    }

  