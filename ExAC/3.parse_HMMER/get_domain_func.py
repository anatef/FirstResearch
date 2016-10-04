import pandas as pd
import unicodedata


def process_hmmer_results(allhmm):
    """
    Get hmmer results in Shilpa's format as pandas dataframe.
    Parse and organize the dataframe.
    """
    #Disable false positive Pandas warning for 'SettingWithCopyWarning'
    pd.options.mode.chained_assignment = None

    #Splitting the Pfam id and domain name column
    allhmm['pfam_id'], allhmm['domain_name'] = zip(*allhmm['HMM_Name'].apply(lambda x: x.split('_', 1)))
    del allhmm['HMM_Name']
    #Get the columns to the original order
    cols = allhmm.columns.tolist()
    cols = cols[0:1] + cols[-2:] + cols[1:-2]
    allhmm = allhmm[cols]

    #Splitting the Description
    allhmm['prot_id'], allhmm['ensembl_id'], allhmm['transcript_id'], allhmm['chromosome_id'], allhmm['length'], allhmm['refseq_hmm_start_end'] = zip(*allhmm['Description'].apply(lambda x: x.split(' ', 5)))
    del allhmm['Description']

    #Splitting the last column to deal with missing refseq ids
    allhmm['refseq'] = allhmm['refseq_hmm_start_end'].apply(lambda x: x[x.find("refseq:")+7:x.find("HMMStart")])
    allhmm["hmm_start"] = allhmm['refseq_hmm_start_end'].apply(lambda x: x[x.find("HMMStart")+9:x.find(";")])
    allhmm["hmm_end"] = allhmm['refseq_hmm_start_end'].apply(lambda x: x[x.find("HMMEnd")+7:-1])
    del allhmm['refseq_hmm_start_end']

    #Extracting the numbers alone from the description columns
    allhmm['prot_id'] = allhmm['prot_id'].apply(lambda x: x[x.find(':')+1:])
    allhmm['ensembl_id'] = allhmm['ensembl_id'].apply(lambda x: x[x.find(':')+1:])
    allhmm['transcript_id'] = allhmm['transcript_id'].apply(lambda x: x[x.find(':')+1:])
    allhmm['chromosome_id'] = allhmm['chromosome_id'].apply(lambda x: x[x.find(':')+1:])
    allhmm['length'] = allhmm['length'].apply(lambda x: x[x.find(':')+1:])

    #Extract only the hugo symbol (without the .number) and add to a new column
    allhmm["Hugo_symbol"] = allhmm['#TargetID'].apply(lambda x: x.split('.')[0])
    #Get the columns to the original order
    cols = allhmm.columns.tolist()
    cols = cols[0:1] + cols[-1:] + cols[1:-1]
    allhmm = allhmm[cols]

    #Seperate chromosome number to a different column
    allhmm["chrom_num"] = allhmm["chromosome_id"].apply(lambda x: x[x.find(":")+1:x.find(":", x.find(":")+1)])
    #Get the columns to the original order
    cols = allhmm.columns.tolist()
    cols = cols[:15] + cols[-1:] + cols[15:-1]
    allhmm = allhmm[cols]
    
    return allhmm
    