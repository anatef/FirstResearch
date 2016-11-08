import pandas as pd
import unicodedata
from collections import defaultdict

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

    #Parsing the description field
    description_headers = ["prot", "pep", "chromosome", "gene", "transcript", "gene_biotype", "transcript_biotype", "hgncID",
                           "hugoSymbol", "refseq", "entrez", "length", "HMMStart", "HMMEnd"]
    descriptions_parsed = defaultdict(list)
    desc_list = allhmm["Description"].tolist()
    
    #Iterating over the Description field and parsing the data
    for i in range(len(desc_list)):
        curr_desc = desc_list[i].split(' ')
        desc_idx = 0
        #Iterating over the headers list and looking for the headers in the parsed data
        for headers_idx in range(len(description_headers)):
            val = curr_desc[desc_idx]
            if (val.find(description_headers[headers_idx]) == 0):
                if (description_headers[headers_idx] in ("HMMStart", "HMMEnd")):
                     parsed = val[val.find("=")+1:-1] 
                else:
                    parsed = val[val.find(":")+1:]
                descriptions_parsed[description_headers[headers_idx]].append(parsed)
                desc_idx += 1
            else:
                #If the headers isn't found, appending empty string instead
                descriptions_parsed[description_headers[headers_idx]].append("")

    #Adding the parsed description to the data frame
    del allhmm["Description"]
    for col_name in description_headers:
        allhmm[col_name] = descriptions_parsed[col_name]
    
    #Seperate chromosome number to a different column
    allhmm["chrom_num"] = allhmm["chromosome"].apply(lambda x: x[x.find(":")+1:x.find(":", x.find(":")+1)])
    #Get the columns to the original order
    cols = allhmm.columns.tolist()
    cols = cols[0:13]+cols[-1:]+cols[13:-1]
    allhmm = allhmm[cols]
    
    return allhmm
    