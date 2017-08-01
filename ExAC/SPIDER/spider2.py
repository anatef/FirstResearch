import subprocess
import cPickle as pickle
import sys

# Directories
files_dir = '/Genomics/grid/users/dtodd/SPIDER2'

# Load genes to aa sequence mapping for each domain
with open(files_dir+"/gene_dict.pik", 'rb') as handle:
    gene_dict = pickle.load(handle)

# Get indices as command line args
domain_index = int(sys.argv[1])
domain = gene_dict.keys()[domain_index]
lo = int(sys.argv[2])
hi = int(sys.argv[3])

# Get secondary structure characteristics for each sequence
for g in range(lo,hi):
    gene = gene_dict[domain].keys()[g]

    # Save sequence to file
    filename = files_dir+"/seq/"+domain+"_"+gene.replace('.','-')+'.seq'
    with open(filename,'w') as f:
        f.write(gene_dict[domain][gene])
    f.close

    # Run SPIDER2
    subprocess.call([files_dir+'/misc/run_local.sh',filename])

# Print message to check for random crashes
print("SUCCESS!!!")
