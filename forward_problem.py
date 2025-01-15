import numpy as np
import argparse
import os


def create_training_data_structures(eps, mu, width, length, n_timesteps, n_sensors):
    # pytorch tensor of input values
    input_data = np.array([eps, mu, width, length])
    # pytorch tensor of output values
    output_data = np.zeros((n_sensors*3, n_timesteps+1))
    return input_data, output_data


def update_sensor_data(sensors, output_data, Ez, Hx, Hy, timestep):
    # Extract sensor values for Ez, Hx, and Hy and stack them into a single array
    sensor_values = np.array([[Ez[x, y], Hx[x, y], Hy[x, y]] for x, y in sensors])

    # Flatten the array to match the dimensions of output_data
    flattened_sensor_values = sensor_values.flatten()

    # Update output_data at the specified timestep for each sensor
    output_data[:, timestep] = flattened_sensor_values
    return output_data


def save_training_data(input_data, output_data, t0_datacut):
    """
    Saves Ez, Hx, and Hy values at specific sensor locations
    to a numpy file for all timesteps
    """
    # Extract values from the input tensor
    eps, mu, width, length = input_data.tolist()

    # Format float parameters (eps, mu) to preserve 3 decimal places
    def format_param(param):
        return int(round(param * 1000))  # Scale floats to 3 decimal places

    eps_str = format_param(eps)
    mu_str = format_param(mu)
    width_str = int(width)
    length_str = int(length)

    # Create the filename using the parameters
    directory = './training_data/'
    os.makedirs(directory, exist_ok=True)
    filename = f"{directory}eps{eps_str}_mu{mu_str}_w{width_str}_l{length_str}.npz"

    # Remove the first t0_datacut points in the time series from output_data
    trimmed_output_data = output_data[:, t0_datacut:]

    # Flatten the trimmed output_data before saving
    flattened_output_data = trimmed_output_data.flatten()

    # Save input and output arrays to the file
    np.savez(filename, input_data=input_data, output_data=flattened_output_data)
    return


def forward_problem(eps_r, mu_r, length, width, plot=False, save=True):
    """
    2D FDTD electromagnetic wave simulation
    Simulates Transverse Magnetic (TM) mode field components with
    PML boundary conditions
    """

    # physics parameters
    c = 3e8                     # [m/s] speed of light
    mu = np.pi*4e-7             # [H/m] vacuum permeability
    epsilon = 1 / (mu * c**2)   # [F/m]

    # wave source parameters
    frequency = 5e9  # [Hz]
    wavelength = c / frequency  # [m]
    sources = [
        (31, 190),   # 1st source
        (200, 31),   # 2nd source
        (163, 329),  # 3rd source
        (329, 120)   # 4th source
    ]

    # simulation parameters
    t0_datacut = 150

    # Define the perimeter of the square (using x = 31, 329 and y = 31, 329)
    x_min, x_max = 31, 329
    y_min, y_max = 31, 329

    # Function to compute Euclidean distance between two points
    def distance(p1, p2):
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    # Function to generate the sensors along the perimeter and filter them
    def generate_sensors(x_min, x_max, y_min, y_max, num_sensors, sources, min_distance=20):
        sensors = set()  # Using a set to avoid duplicate sensors

        # Divide the perimeter into segments
        # Horizontal lines: y = y_min and y = y_max
        for y in [y_min, y_max]:
            step = (x_max - x_min) / (num_sensors // 2)  # Half the sensors on each horizontal side
            for i in range(num_sensors // 2):
                x = x_min + i * step
                x_int = round(x)  # Round to nearest integer
                sensor = (x_int, y)
                # Filter sensor if within min_distance of any source
                if not any(distance(sensor, source) <= min_distance for source in sources):
                    sensors.add(sensor)

        # Vertical lines: x = x_min and x = x_max
        for x in [x_min, x_max]:
            step = (y_max - y_min) / (num_sensors // 2)  # Half the sensors on each vertical side
            for i in range(num_sensors // 2):
                y = y_min + i * step
                y_int = round(y)  # Round to nearest integer
                sensor = (x, y_int)
                # Filter sensor if within min_distance of any source
                if not any(distance(sensor, source) <= min_distance for source in sources):
                    sensors.add(sensor)

        return list(sensors)  # Convert set back to list for final output

    # Generate the sensors (you can adjust num_sensors to increase or decrease the number of sensors)
    num_sensors = 40
    sensors = generate_sensors(x_min, x_max, y_min, y_max, num_sensors, sources)
    n_sensors = len(sensors)

    # interior grid parameters (not PML)
    domain_nx = 300                     # number of points in x direction
    domain_ny = 300                     # number of points in the y direction
    size_x = 0.5                 # total domin size in x direction [m]
    size_y = 0.5                 # total domin size in x direction [m]
    dx = size_x / domain_nx             # grid spacing in x direction [m]
    dy = size_y / domain_ny             # grid spacing in y direction [m]

    # PML boundary parameters
    pml_thickness = 30
    R_0 = 1e-8  # PML refelction Coefficient
    pml_order = 2

    # overall grid parameters (interior + pml)
    nx = domain_nx + 2*pml_thickness  # number of CELLS in x direction
    ny = domain_ny + 2*pml_thickness  # number of CELLS in y direciton
    nxm1 = nx - 1
    nym1 = ny - 1
    nxp1 = nx + 1
    nyp1 = ny + 1

    # imbedded rectangle parameters
    length = length
    width = width
    material_epsilon = eps_r
    material_mu = mu_r

    # time stepping parameters
    courant_factor = 0.9
    dt = courant_factor * 1/(c*np.sqrt(1/(dx**2) + 1/(dy**2)))   # based on CFL
    n_timesteps = 1300
    t_final = dt*n_timesteps     # final time
    t = 0                        # starting time

    # log information to the consol
    # print("\n PROBLEM INFORMATION ")
    # print(" -------------------")
    # print(f"number of grid points in x direction = {nx}")
    # print(f"number of grid points in y direction = {ny}")
    # print(f"domain width  = {size_x}[m]")
    # print(f"domain height = {size_y}[m]")
    # print(f"dx = {dx}[m] \ndy = {dy}[m] \ndt = {dt}[s]")
    # print(f"wavelength = {wavelength}[m] \nfrequency = {frequency}[Hz]")
    # print(f"epsilon = {epsilon}[?] \nmu = {mu}[?]")
    # print(f"c = {c}[m/s]\n")

    # Initialize Fields for entire domain (even PML)
    Ez = np.zeros((nxp1, nyp1))
    Hx = np.zeros((nxp1, ny))
    Hy = np.zeros((nx, nyp1))

    # Initialize relative material parameters
    eps_rz = np.ones((nxp1, nyp1))
    mu_rx = np.ones((nxp1, ny))
    mu_ry = np.ones((nx, nyp1))
    sigma_ez = np.zeros((nxp1, nyp1))
    sigma_mx = np.zeros((nxp1, ny))
    sigma_my = np.zeros((nx, nyp1))

    # Calculate material parameters
    center = nx/2
    lower_w = int(center - width/2)
    upper_w = int(center + width/2)+1
    lower_l = int(center - length/2)
    upper_l = int(center + length/2)+1
    eps_rz[lower_l:upper_l, lower_w:upper_w] = material_epsilon
    mu_ry[lower_l:upper_l, lower_w:upper_w] = material_mu
    mu_rx[lower_l:upper_l, lower_w:upper_w] = material_mu

    # Create update coefficients
    # for updating Ez
    Ceze = (2 * eps_rz * epsilon - dt * sigma_ez) / (2 * eps_rz * epsilon + dt * sigma_ez)
    Cezhy = (2 * dt / dx) / (2 * eps_rz * epsilon + dt * sigma_ez)
    Cezhx = -(2 * dt / dy) / (2 * eps_rz * epsilon + dt * sigma_ez)
    # for updating Hx
    Chxh = (2 * mu_rx * mu - dt * sigma_mx) / (2 * mu_rx * mu + dt * sigma_mx)
    Chxez = -(2 * dt / dy) / (2 * mu_rx * mu + dt * sigma_mx)
    # for updating Hy
    Chyh = (2 * mu_ry * mu - dt * sigma_my) / (2 * mu_ry * mu + dt * sigma_my)
    Chyez = (2 * dt / dx) / (2 * mu_ry * mu + dt * sigma_my)

    # Initialize PML fields
    Ezx_xn = np.zeros((pml_thickness, nym1))
    Ezy_xn = np.zeros((pml_thickness, nym1 - 2*(pml_thickness)))
    Ezx_xp = np.zeros((pml_thickness, nym1))
    Ezy_xp = np.zeros((pml_thickness, nym1 - 2*(pml_thickness)))
    Ezx_yn = np.zeros((nxm1 - 2*(pml_thickness), pml_thickness))
    Ezy_yn = np.zeros((nxm1, pml_thickness))
    Ezx_yp = np.zeros((nxm1 - 2*(pml_thickness), pml_thickness))
    Ezy_yp = np.zeros((nxm1, pml_thickness))

    # Initialize PML Coefficients
    # for the xn region
    # sigma_pex_xn stores the values of the sigma values in the PML cells
    sigma_pex_xn = np.zeros((pml_thickness, nym1))
    sigma_pmx_xn = np.zeros((pml_thickness, nym1))
    # For the xp region (PML layer in the positive x direction)
    sigma_pex_xp = np.zeros((pml_thickness, nym1))
    sigma_pmx_xp = np.zeros((pml_thickness, nym1))
    # For the yn region (PML layer in the negative y direction)
    sigma_pey_yn = np.zeros((nxm1, pml_thickness))
    sigma_pmy_yn = np.zeros((nxm1, pml_thickness))
    # For the yp region (PML layer in the positive y direction)
    sigma_pey_yp = np.zeros((nxm1, pml_thickness))
    sigma_pmy_yp = np.zeros((nxm1, pml_thickness))

    # max sigma value
    sigma_max = -(pml_order + 1) * epsilon * c * np.log(R_0) / (2 * dx * pml_thickness)
    # calculated rho and sigma values (distance from the interior boundary
    # of the PML
    rho_e = (np.arange(pml_thickness, 0, -1) - 0.25) / pml_thickness
    rho_m = (np.arange(pml_thickness, 0, -1) - 0.75) / pml_thickness
    for i in range(pml_thickness):
        # for xn
        sigma_pex_xn[i, :] = sigma_max * rho_e[i] ** pml_order
        sigma_pmx_xn[i, :] = (mu / epsilon) * sigma_max * rho_m[i] ** pml_order
        # for yn
        sigma_pey_yn[:, i] = sigma_max * rho_e[i] ** pml_order
        sigma_pmy_yn[:, i] = (mu / epsilon) * sigma_max * rho_m[i] ** pml_order

    rho_e = (np.arange(1, pml_thickness+1) - 0.25) / pml_thickness
    rho_m = (np.arange(1, pml_thickness+1) - 0.75) / pml_thickness
    for i in range(pml_thickness):
        # calcualte the actual values of sigma_p
        # for xp
        sigma_pex_xp[i, :] = sigma_max * rho_e[i] ** pml_order
        sigma_pmx_xp[i, :] = (mu / epsilon) * sigma_max * rho_m[i] ** pml_order
        # for yp
        sigma_pey_yp[:, i] = sigma_max * rho_e[i] ** pml_order
        sigma_pmy_yp[:, i] = (mu / epsilon) * sigma_max * rho_m[i] ** pml_order

    # Update coefficients for PML layer
    # Coefficients updating Hx
    Chxh_yn = (2 * mu - dt * sigma_pmy_yn) / (2 * mu + dt * sigma_pmy_yn)
    Chxez_yn = -(2 * dt / dy) / (2 * mu + dt * sigma_pmy_yn)
    Chxh_yp = (2 * mu - dt * sigma_pmy_yp) / (2 * mu + dt * sigma_pmy_yp)
    Chxez_yp = -(2 * dt / dy) / (2 * mu + dt * sigma_pmy_yp)
    # Coefficients updating Hy
    Chyh_xn = (2 * mu - dt * sigma_pmx_xn) / (2 * mu + dt * sigma_pmx_xn)
    Chyez_xn = (2 * dt / dx) / (2 * mu + dt * sigma_pmx_xn)
    Chyh_xp = (2 * mu - dt * sigma_pmx_xp) / (2 * mu + dt * sigma_pmx_xp)
    Chyez_xp = (2 * dt / dx) / (2 * mu + dt * sigma_pmx_xp)
    # Coefficients updating Ezx
    Cezxe_xp = (2 * epsilon - dt * sigma_pex_xp) / (2 * epsilon + dt * sigma_pex_xp)
    Cezxhy_xp = (2 * dt / dx) / (2 * epsilon + dt * sigma_pex_xp)
    Cezxe_yn = 1
    Cezxhy_yn = dt / (dx * epsilon)
    Cezxe_yp = 1
    Cezxhy_yp = dt / (dx * epsilon)
    Cezxe_xn = (2 * epsilon - dt * sigma_pex_xn) / (2 * epsilon + dt * sigma_pex_xn)
    Cezxhy_xn = (2 * dt / dx) / (2 * epsilon + dt * sigma_pex_xn)
    # Coefficients updating Ezy
    Cezye_xn = 1
    Cezyhx_xn = -dt / (dy * epsilon)
    Cezye_xp = 1
    Cezyhx_xp = -dt / (dy * epsilon)
    Cezye_yp = (2 * epsilon - dt * sigma_pey_yp) / (2 * epsilon + dt * sigma_pey_yp)
    Cezyhx_yp = -(2 * dt / dy) / (2 * epsilon + dt * sigma_pey_yp)
    Cezye_yn = (2 * epsilon - dt * sigma_pey_yn) / (2 * epsilon + dt * sigma_pey_yn)
    Cezyhx_yn = -(2 * dt / dy) / (2 * epsilon + dt * sigma_pey_yn)

    # Initialize files for storing simulation data
    input_data, output_data = create_training_data_structures(eps_r, mu_r, width, length, n_timesteps, n_sensors)

    timestep = 0
    while t < t_final:
        t += dt
        ps = pml_thickness
        pe = -pml_thickness

        # update Hx
        Hx[:, ps:pe] = Chxh[:, ps:pe] * Hx[:, ps:pe] + \
            Chxez[:, ps:pe] * (Ez[:, ps+1:pe] - Ez[:, ps:pe-1])

        # update Hy
        Hy[ps:pe, :] = Chyh[ps:pe, :] * Hy[ps:pe, :] + \
            Chyez[ps:pe, :] * (Ez[ps+1:pe, :] - Ez[ps:pe-1, :])

        # update H in PML layers
        # for xn
        Hy[:ps, 1:-1] = \
            Chyh_xn * Hy[:ps, 1:-1] + \
            Chyez_xn * (Ez[1:ps+1, 1:-1] - Ez[:ps, 1:-1])
        # for xp
        Hy[pe:, 1:-1] = \
            Chyh_xp * Hy[pe:, 1:-1] + \
            Chyez_xp * (Ez[pe:, 1:-1] - Ez[pe-1:-1, 1:-1])
        # for yn
        Hx[1:-1, :ps] = \
            Chxh_yn * Hx[1:-1, :ps] + \
            Chxez_yn * (Ez[1:-1, 1:ps+1] - Ez[1:-1, :ps])
        # for yp
        Hx[1:-1, pe:] = \
            Chxh_yp * Hx[1:-1, pe:] + \
            Chxez_yp * (Ez[1:-1, pe:] - Ez[1:-1, pe-1:-1])

        # update_Ez()
        Ez[ps+1:pe-1, ps+1:pe-1] = (
            Ceze[ps+1:pe-1, ps+1:pe-1] * Ez[ps+1:pe-1, ps+1:pe-1] +
            Cezhy[ps+1:pe-1, ps+1:pe-1] *
            (Hy[ps+1:pe, ps+1:pe-1] - Hy[ps:pe-1, ps+1:pe-1]) +
            Cezhx[ps+1:pe-1, ps+1:pe-1] *
            (Hx[ps+1:pe-1, ps+1:pe] - Hx[ps+1:pe-1, ps:pe-1])
        )

        # update impressed current J_z
        for psx, psy in sources:
            Cezj = -(2 * dt) / (2 * eps_rz[psx, psy] * epsilon + dt * sigma_ez[psx, psy])
            Ez[psx, psy] = Ez[psx, psy] + Cezj * np.sin(2 * np.pi * frequency * t)

        # update_Ez_PML layers
        # For xn PML region
        Ezx_xn = Cezxe_xn * Ezx_xn + Cezxhy_xn * (Hy[1:ps+1, 1:-1] - Hy[:ps, 1:-1])
        Ezy_xn = Cezye_xn * Ezy_xn + Cezyhx_xn * (Hx[1:ps+1, ps+1:pe] - Hx[1:ps+1, ps:pe-1])
        # For xp PML region
        Ezx_xp = Cezxe_xp * Ezx_xp + Cezxhy_xp * (Hy[pe:, 1:-1] - Hy[pe-1:-1, 1:-1])
        Ezy_xp = Cezye_xp * Ezy_xp + Cezyhx_xp * (Hx[pe-1:-1, ps+1:pe] - Hx[pe-1:-1, ps:pe-1])
        # For yn PML region
        Ezx_yn = Cezxe_yn * Ezx_yn + Cezxhy_yn * (Hy[ps+1:pe, 1:ps+1] - Hy[ps:pe-1, 1:ps+1])  
        Ezy_yn = Cezye_yn * Ezy_yn + Cezyhx_yn * (Hx[1:-1, 1:ps+1] - Hx[1:-1, :ps])
        # For yp PML region
        Ezx_yp = Cezxe_yp * Ezx_yp + Cezxhy_yp * (Hy[ps+1:pe, pe-1:-1] - Hy[ps:pe-1, pe-1:-1])
        Ezy_yp = Cezye_yp * Ezy_yp + Cezyhx_yp * (Hx[1:-1, pe:] - Hx[1:-1, pe-1:-1])
        # Update the Ez field at the corresponding PML regions
        Ez[1:ps+1, 1:ps+1] = Ezx_xn[:, :ps] + Ezy_yn[:ps, :]  # lower L corner
        Ez[1:ps+1, pe-1:-1] = Ezx_xn[:, pe:] + Ezy_yp[:ps, :]  # upper L corner
        Ez[pe-1:-1, pe-1:-1] = Ezx_xp[:, pe:] + Ezy_yp[pe:, :]  # upper R corner
        Ez[pe-1:-1, 1:ps+1] = Ezx_xp[:, :ps] + Ezy_yn[pe:, :]  # lower R corner
        Ez[ps+1:pe-1, 1:ps+1] = Ezx_yn + Ezy_yn[ps:pe, :]  # bottom
        Ez[ps+1:pe-1, pe-1:-1] = Ezx_yp + Ezy_yp[ps:pe, :]   # top
        Ez[1:ps+1, ps+1:pe-1] = Ezx_xn[:, ps:pe] + Ezy_xn  # left
        Ez[pe-1:-1, ps+1:pe-1] = Ezx_xp[:, ps:pe] + Ezy_xp   # right

        # Plot simulation
        if plot:
            # Ensure the output directory exists
            output_dir = "plotting_matrices/"
            os.makedirs(output_dir, exist_ok=True)
            timestep_str = f"{timestep:05d}"  # Format timestep as a zero-padded 5-digit number
            np.save(os.path.join(output_dir, f"Ez_timestep_{timestep_str}.npy"), Ez)
            np.save(os.path.join(output_dir, f"Hx_timestep_{timestep_str}.npy"), Hx)
            np.save(os.path.join(output_dir, f"Hy_timestep_{timestep_str}.npy"), Hy)

        # update sensors
        output_data = update_sensor_data(sensors, output_data, Ez, Hx, Hy, timestep)

        # increment timestep
        timestep += 1

    if save:
        # Save data for training gaussian process
        save_training_data(input_data, output_data, t0_datacut)
    else:
        # Remove the first t0_datacut points in the time series from output_data
        trimmed_output_data = output_data[:, t0_datacut:]

        # Flatten the trimmed output_data before saving
        flattened_output_data = trimmed_output_data.flatten()

        return flattened_output_data

    return


if __name__ == "__main__":
    # parse commandline arguments
    parser = argparse.ArgumentParser("2D FDTD forward problem")
    parser.add_argument('eps', type=float, default=10, help='relative material epsilon')
    parser.add_argument('mu', type=float, default=1, help='relative material epsilon')
    parser.add_argument('length', type=float, help='length of inclusion')
    parser.add_argument('width', type=float, help='width of inclusion')
    parser.add_argument('plot', type=bool, help='whether or not to plot the simulation')
    args = parser.parse_args()
    # run forward simulation
    forward_problem(args.eps, args.mu, args.length, args.width, args.plot)
