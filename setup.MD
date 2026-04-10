This is a great idea. Having a dedicated "cheat sheet" will save you a ton of time every time you sit down to work on your thesis.

Here is your complete, step-by-step master guide to connecting your local PyFluent environment to the TU Ilmenau HPC cluster. Bookmark this or save it to a text file!

### Phase 1: Start the Fluent Server on the Cluster

First, we need to launch the Ansys solver in the background on the HPC and get the secret connection credentials.

1. **Log in to the cluster** from your local terminal:
```bash
ssh jote9827@cslogin.tu-ilmenau.de

```


*(Note the name of the compute node you land on, e.g., `makalu48` or `makalu47`)*.
2. **Load the Ansys module** (Make sure to use a version your PyFluent supports, like 2024 R2):
```bash
module purge
module load ansys/v2024r2

```


3. **Start Fluent in Server Mode** in the background:
```bash
fluent 2ddp -t4 -gu -sifile=serverinfo.txt > fluent_server.txt &

```


*(Flags used: `2ddp` = 2D Double Precision, `-t4` = 4 CPU cores, `-gu` = hidden graphics, `-sifile` = write the password to a file).*
4. **Get your credentials:** Wait about 10 seconds, then read the generated info file:
```bash
cat serverinfo.txt

```


*You will see a line like: `141.24.193.47:33283:6cmg****`. Write down the **IP Address**, the **Port**, and the **Password**.*

---

### Phase 2: Create the SSH Tunnel

Now we need to securely bridge your laptop to that specific compute node.

1. Open a **brand new, separate terminal window** on your local Linux PC.
2. **Start the tunnel** using the exact IP and Port you just got from `serverinfo.txt`. Run this command, replacing `YOUR_PORT` and `YOUR_IP` with those numbers:
```bash
ssh -L 0.0.0.0:YOUR_PORT:YOUR_IP:YOUR_PORT jote9827@cslogin.tu-ilmenau.de

```


*(Example: `ssh -L 0.0.0.0:33283:141.24.193.47:33283 jote9827...`)*
3. Enter your university password when prompted, and **leave this terminal open and minimized**. The tunnel is now active!

---

### Phase 3: Connect via Python

Finally, run your PyFluent script on your local PC.

Here is the exact boilerplate code you need. Just update the `port` and `password` every time you start a new session:

```python
import ansys.fluent.core as pyfluent

print("Connecting to TU Ilmenau Cluster...")

# Connect to the local end of the SSH tunnel
solver = pyfluent.connect_to_fluent(
    ip="0.0.0.0",                 # Magic IP to bypass Windows security checks
    port=33283,                   # <-- UPDATE THIS to your current port
    password="your_password",     # <-- UPDATE THIS to your current password
    allow_remote_host=True,       # Permission to connect over the network
    insecure_mode=True            # Bypasses PyFluent's internal TLS (SSH handles security)
)

print("Successfully connected to remote Fluent!")

# ---- YOUR AUTOMATION CODE GOES HERE ----
# Example: Load your case file
# solver.file.read_case(file_name="/scratch/jote9827/Rio_Project_Static_28c/Rio_Static.cas.h5")

# Close the connection when the script finishes
solver.exit()

```

Run your Python script (`python fluent.py`), and you are fully connected!

---

Have a great time logging off and taking a break! When you come back to tackle this next, would you like me to help you write the Python code to extract specific data (like drag or lift forces) directly from your `Rio_Static.dat.h5` file?