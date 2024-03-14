import re
import subprocess

# Use grep to extract only the relevant lines
grep_output = subprocess.check_output(['grep', '-E', r'\[Profile\] guid\(.*latency\(', '/home/ubuntu/FlexFlow/inference/output/1_machine-1_gpu-1_batchsize-sequence_specinfer.out'])
lines = grep_output.decode('utf-8').splitlines()

# Define the regular expression pattern
pattern = r'\[Profile\] guid\((\d+)\) .*latency\((\d+\.\d+)\)'

# Iterate through the lines
sum = 0
for line in lines:
    match = re.search(pattern, line)
    if match:
        guid = match.group(1)
        latency = match.group(2)
        print(f"GUID: {guid}, Latency: {latency}")
        sum += float(latency)
print(sum)
print(sum/(128*359))