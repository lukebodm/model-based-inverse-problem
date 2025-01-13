def save_training_data(sensors, Ez, eps, mu, width, length):
    """
    Saves Ez values at specific sensor locations to a file, appending new time-step data.
    """
    # Get Ez values at the sensor locations
    sensor_values = [Ez[x, y] for x, y in sensors]

    # Create the directory if it doesn't exist
    directory = './training_data/'
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Create a filename based on the parameters
    filename = f"{directory}eps{int(eps)}_mu{int(mu)}_w{int(width)}_l{int(length)}.csv"

    # If the file doesn't exist, create it with a header
    if not os.path.exists(filename):
        with open(filename, 'w') as f:
            header = "eps,mu,width,length," + ",".join([f"time_series_{i}" for i in range(len(sensors))]) + "\n"
            f.write(header)

    # Append new data
    with open(filename, 'a') as f:
        data = f"{eps},{mu},{width},{length}," + ",".join(map(str, sensor_values)) + "\n"
        f.write(data)

