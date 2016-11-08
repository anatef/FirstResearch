## Identification of functionally-important residues using population data

The pipeline files should be run according to the numbered directories of code files.

### 1) Process ExAC population data
1. Download the ExAC non-TCGA dataset from: ftp://ftp.broadinstitute.org/pub/ExAC_release/release0.3/subsets/ExAC.r0.3.nonTCGA.sites.vep.vcf.gz

2. Run the script "ExAC_parser.ipynb" to process the database data and create a processed file for each chromosome.
The processed files are saved in the folder: "parsed/".

Note: the script "loading_chrom.ipynb" can be used to view a parsed chromosome file. But it's not a part of the pipeline.


### 2) Parsing Pfam
1. Download the Pfam-A.hmm data from: ftp://ftp.ebi.ac.uk/pub/databases/Pfam/releases/Pfam30.0/Pfam-A.hmm.gz

2. Run the script "domain_GA.ipynb" to create a table of all the domains gathering thresholds.
The table is saved to: "domains_GA.csv"

3. Run the script "domains_conserved_states.ipynb" to create a dictionary of the HMM emission probabilities, per domain.
The dictionary is pickled and saved to: "domains_hmm_prob_dict.pik"

Note: the script "pfam_hmm_profile.ipynb" can be used to parse one domain .hmm file.


### 3) Process HMMER results
1. Save HMMER results to a table called: "from_shilpa/allhmmresbyprot-v30.tsv"

2. Run the script "get_domain_hmm.ipynb" to filter the relevant domains HMM-matches.
The script uses #2 "domains_GA.csv" and "domains_hmm_prob_dict.pik" to do the filtering.
The parsed table for all the domains is saved to: "allhmm_parsed/".
The domain files are saved in the folder: "hmm_domains/".

3. Save the exons frameshifts results in the folder: "from_shilpa/exons_seqs/"

4. Run the script "exons_frameshifts.ipynb" to create a dictionary of the exons frameshifts, per protein ENSEMBL ID.
The dictionary, and a readable .csv of this mapping is saved to: "domains_frameshifts/".


### 4) Parse Uniprot
1. Run the script "canonical_protein.ipynb" to find the canonical proteins in each domain family of proteins.
The scripts uses #3 "hmm_domains/" files as input.
The mapping of gene-ID to canonic protein-ID is pickled and saved to: "domains_canonic_prot/pfam_v30/".


### 5) HMM-alterations alignment
1. Run the script "alteration_to_hmm_state.ipynb" with your domain of choice.
The script uses #1 "parsed/", #3 "hmm_domains/" and  #4 "domains_canonic_prot/pfam_v30/" as input.
The script map all the domain genomic locations to the pfam HMM match states.
Then, a count of all genomic alterations is saved in the form of Minor Allele Frequency and a detailed dictionary of amino-acids frequencies per genomic position.
The output is a pickled dictionary of the domain HMM states. Each state points to a list of the aligned genomic locations. Each such genomic location is represented by a dictionary of all the relevant data.
This pickled dictionary is saved to: "domains_states_dicts/pfam_v30/".


### 6) ExAC coverage infromation
1. Download ExAC coverage data from: ftp://ftp.broadinstitute.org/pub/ExAC_release/release0.3/coverage and save to: "coverage_raw/"

2. Run the script "coverage_data.ipynb" in order to add coverage data to any of the states dictionaries from #5.
The output is saved to: "coverage_states_dicts/"


### 7) HMM-states data filtering
1. Run the script "states_filter.ipynb" in order to filter the data according to specific parameters, such as: coverage.
the output is saved to: "filtered_dicts/".


### 8) States analysis
Scripts for analyzing the HMM states data in various ways, such as: EMD.




