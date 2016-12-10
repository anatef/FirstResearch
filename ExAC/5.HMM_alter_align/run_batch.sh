#! /bin/bash
#
# Run the domains in batch, according to the list index.
#
#


for index in $(seq 0 1 9)
do
	cat alteration_to_hmm_state-run.ipynb | idx=$index runipy --stdout > domains_states_dicts/pfam-v30/reports/idx$index.ipynb
done
