import subprocess
import cPickle as pickle
import sys

# Directories
files_dir = '/Genomics/grid/users/dtodd/SPIDER2'


# Load genes to aa sequence mapping for each domain
with open(files_dir+"/sequence_dict.pik", 'rb') as handle:
    sequence_dict = pickle.load(handle)

# Get indices as command line args
lo = int(sys.argv[1])
hi = int(sys.argv[2])

# Get secondary structure characteristics for each sequence
for g in range(lo,hi):
    gene = sequence_dict.keys()[g]
    for prot in sequence_dict[gene]:
        # Save sequence to file
        filename = files_dir+"/seq/"+gene+'-'+prot+'.seq'
        with open(filename,'w') as f:
            f.write(sequence_dict[gene][prot].replace('*','').replace('X',''))
        f.close()

        # Run SPIDER2
        subprocess.call([files_dir+'/misc/run_local.sh',filename])

    # Help pinpoint errors
    print(gene)

# Print message to check for random crashes
print("SUCCESS!!!")
