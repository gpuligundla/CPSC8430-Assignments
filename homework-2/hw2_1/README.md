The Model is trained using GPUs in the palmetto. The source files are in /scratch DIR

# For Training
run seq_to_seq.ipynb 

# For Testing
run ./hw2_seq2seq.sh [testing feature file directory] [output txt file path]

ex: ./hw2_seq2seq.sh /scratch/gpuligu/MLDS_hw2_1_data/testing_data/feat output.txt
