## Pipeline for identification of important residues based on population data


### Process ExAC population data
1. Download the ExAC database to a file called: "ExAC.r0.3.nonTCGA.sites.vep.vcf.gz"

2. Run the script "ExAC_parser.ipynb" to process the database data anc create a processed file for each chromosome.
The processed files are saved in the folder: "parsed/".

Note: the script "loading_chrom.ipynb" can be used to view a parsed chromosome file. But it's not a part of the pipeline.

### Process HMMER results
1. Download the HMM results to a file called: "allhmmresbyprot-new.tsv"

2. Run the script "get_domain_hmm.ipynb" to filter the relevant domain HMM-matches.
The domain is saved in the folder: "hmm_domains/".

3. Run the script "canonical_protein.ipynb" to find the protein ids of the canonical proteins.
The output is saved to: "hmm_domains/".

### Align population data to match states

1. Run the script "alteration_to_hmm_state.ipynb" to align the ExAC chromosomes data to the HMM-match states.
The output is saved in the folder: "hmm_domains/".

2. Run the script "states_distribution.ipynb" to analyze the MAF of each HMM-state.



