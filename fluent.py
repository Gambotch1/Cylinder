import ansys.fluent.core as pyfluent

print("Attempting to connect to TU Ilmenau Cluster...")

solver = pyfluent.connect_to_fluent(
    ip="0.0.0.0",   
    port=33283,           # <-- UPDATE THIS
    password="ohmn98iv",  # <-- UPDATE THIS
    allow_remote_host=True,
    insecure_mode=True    
)

print("Successfully connected to remote Fluent!")

version = solver.scheme.eval("(cx-version)")
print(f"Fluent Version running on cluster: {version}")

solver.exit()