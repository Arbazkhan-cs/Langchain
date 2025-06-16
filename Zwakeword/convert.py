import re

# Read the C++ file
with open("model.cc", "r") as f:
    lines = f.readlines()

# Find the array definition and extract values
data = []
inside_array = False

for line in lines:
    # Start processing after detecting the array definition
    if "unsigned char" in line and "=" in line:
        inside_array = True
        continue  # Skip the definition line

    if inside_array:
        # Stop if the array ends
        if "};" in line:
            break
        
        # Extract hexadecimal values
        hex_values = re.findall(r'0x[0-9A-Fa-f]+', line)
        data.extend(int(h, 16) for h in hex_values)

# Convert byte array to binary file
with open("extracted_model.tflite", "wb") as f:
    f.write(bytearray(data))

print("Model extracted successfully as extracted_model.tflite")
