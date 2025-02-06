import pandas as pd

def convert_to_libsvm_and_extract(input_file, output_file, subset_sizes):
    # Load the dataset
    df = pd.read_csv(input_file)
    
    # Convert labels to be in the range [0, num_class)
    df.iloc[:, -1] = df.iloc[:, -1] - 1
    
    # Open the main output file for the full dataset in LIBSVM format
    with open(output_file, 'w') as full_file:
        # Initialize files for subsets
        subset_files = {size: open(f'{output_file.rsplit(".", 1)[0]}_{size}.txt', 'w') for size in subset_sizes}
        subset_counts = {size: 0 for size in subset_sizes}
        
        for index, row in df.iterrows():
            # Get the class label (target)
            label = int(row[-1])
            # Initialize the libsvm formatted line with the label
            libsvm_line = [str(label)]
            # Iterate through features and format them
            for i in range(len(row) - 1):
                # Skip zero features for LIBSVM format efficiency
                if row[i] != 0:
                    libsvm_line.append(f"{i+1}:{row[i]}")
            # Join the features
            formatted_line = " ".join(libsvm_line) + "\n"
            
            # Write the line to the full output file
            full_file.write(formatted_line)
            
            # Write to subset files if applicable
            for size in subset_sizes:
                if subset_counts[size] < size:
                    subset_files[size].write(formatted_line)
                    subset_counts[size] += 1
        
        # Close all subset files
        for size in subset_sizes:
            subset_files[size].close()

# Define the input and output file paths
input_file = 'data/covtype/covtype.data'
output_file = 'data/covtype/covtype.libsvm'

# Define the subset sizes
subset_sizes = [1000, 10000, 100000]

# Convert the dataset and extract subsets
convert_to_libsvm_and_extract(input_file, output_file, subset_sizes)
