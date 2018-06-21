## SPIDER2 Secondary Structure

SPIDER2 with HSE can be downloaded from http://sparks-lab.org/index.php/Main/Downloads.
Follow the instructions in the README to setup.

### Run SPIDER2 program

Move a dictionary of the form gene_id -> prot_id -> sequence called sequence_dict to the directory users/dtodd/SPIDER2/ on gen-comp1. Make the call:

    python run_spider.py

to submit all necessary jobs. This automatically generates the relevant seq files and partitions the
request into jobs that take ~1 day. Use sftp or your file transfer method of choice to move the output
files off the cluster.

### check_output

Optional

Checks out files for crashes and provides an option to rerun a few failed jobs without creating a new
dictionary

### make_domain_dicts

Given a dictionary that contains the genes associated with each domain, the sequence dictionary used to run SPIDER, and the SPIDER output files, creates the predicted secondary structure domain dictionaries
