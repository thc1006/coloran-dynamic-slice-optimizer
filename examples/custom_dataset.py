# examples/custom_dataset.py
"""
An example demonstrating how to adapt the pipeline for a custom dataset.
"""
import pandas as pd
import numpy as np

# --- Step 1: Create a Custom Data Loader ---
# Let's assume your custom data is in a single CSV file.

class CustomDataLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_raw_data(self):
        print(f"Loading custom data from {self.file_path}...")
        # Your data loading logic here
        df = pd.read_csv(self.file_path)
        # **Crucially, you must ensure the column names match what the processor expects,
        # or you must create a custom processor.**
        # Example: Renaming columns
        df.rename(columns={'ue_count': 'num_ues', 'slice_name': 'slice_id'}, inplace=True)
        return df

# --- Step 2: (Optional) Create a Custom Data Processor ---
# If your feature engineering is different, you'd create a new class
# similar to MemoryOptimizedProcessor. For this example, we assume the
# raw data columns are sufficient or have been renamed to match.

from src.data.data_processor import MemoryOptimizedProcessor

# --- Step 3: Run the pipeline with your custom components ---

def run_with_custom_data(data_path):
    print("## Running pipeline with custom dataset ##")
    
    # Create dummy data for the example
    if not os.path.exists(data_path):
        print("Creating dummy custom dataset...")
        dummy_df = pd.DataFrame({
            'Timestamp': pd.to_datetime(np.arange(100), unit='s'),
            'sched_policy': ['custom_policy'] * 100,
            'training_config': ['custom_config'] * 100,
            'slice_id': np.random.randint(0, 3, 100),
            'sum_requested_prbs': np.random.randint(5, 20, 100),
            'sum_granted_prbs': np.random.randint(4, 19, 100),
            'tx_brate downlink [Mbps]': np.random.uniform(10, 80, 100),
            'tx_errors downlink (%)': np.random.uniform(0, 5, 100)
        })
        dummy_df.to_csv(data_path, index=False)

    # 1. Use your custom loader
    custom_loader = CustomDataLoader(data_path)
    raw_data = custom_loader.load_raw_data()
    
    # 2. Use the existing processor (or your custom one)
    # We need to provide some dummy slice configs
    slice_configs = {'custom_config': [10, 5, 2]}
    processor = MemoryOptimizedProcessor(slice_configs)
    processed_data = processor.process_data(raw_data)
    
    if processed_data is not None:
        print("\nCustom data processed successfully!")
        print(processed_data.head())
        # From here, you would proceed with training, simulation, etc.
        # e.g., pass `processed_data` to `A100OptimizedTrainer`
    else:
        print("Failed to process custom data.")

if __name__ == '__main__':
    import os
    custom_data_file = 'my_custom_network_data.csv'
    run_with_custom_data(custom_data_file)

