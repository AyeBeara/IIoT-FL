import kagglehub
import pandas as pd
import os

def setup_and_split():
    # Load the dataset the first time

    path = kagglehub.dataset_download("canozensoy/industrial-iot-dataset-synthetic")
    data = path + "/factory_sensor_simulator_2040.csv"

    df = pd.read_csv(data)

    # Split the dataframe into 33 partitions based on the "Machine_Type" column
    partitions = {}
    for machine_type in df["Machine_Type"].unique():
        partitions[machine_type] = df[df["Machine_Type"] == machine_type]

    # Save each partition into a separate directory in ./data/{machine_type}
    for machine_type, partition_df in partitions.items():
        os.makedirs(f"./data/{machine_type}", exist_ok=True)
        partition_df.to_csv(f"./data/{machine_type}/data.csv", index=False)

if __name__ == "__main__":
    # Only run the setup if the data directory doesn't exist
    if not os.path.exists("./data"):
        setup_and_split()