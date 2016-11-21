#!/usr/bin/python

###############################################################################
# processTCGAMut.py                                                           #
# Author:  Dario Ghersi, updated by Pawel Przytycki and Shilpa Nadimpalli     #
# Version: 20130503                                                           #
# Goal:    process a TCGA MAF file and output the FASTA sequence of the gene  #
#          (protein format) and the coordinate of the mutations in            #
#          the protein                                                        #
# Usage:   processTCGAMut.py TCGA_MAF NCBI_HSREF NCBI_GBS NP_SEQS ...         #
#                            OUT_SEQS OUT_MUT                                 #
#                                                                             #
# Note:    the script can be made more efficient in the mapping of mutations  #
#          part                                                               #
###############################################################################

import glob
import os
import re
import sys
import collections
import gzip

###############################################################################
# CONSTANTS                                                                   #
###############################################################################

DBPATH = '/home/anat/Research/ExAC/Ensembl_build/'
PATH = '/home/snadimpa/workspace/cancertf/data/'

COMPLEMENT = {"A":"T", "T":"A", "C":"G", "G":"C", "N":"N",
              "a":"t", "t":"a", "c":"g", "g":"c", "n":"n"}

GENCODE = {
    'ATA':'I',    # Isoleucine
    'ATC':'I',    # Isoleucine
    'ATT':'I',    # Isoleucine
    'ATG':'M',    # Methionine
    'ACA':'T',    # Threonine
    'ACC':'T',    # Threonine
    'ACG':'T',    # Threonine
    'ACT':'T',    # Threonine
    'AAC':'N',    # Asparagine
    'AAT':'N',    # Asparagine
    'AAA':'K',    # Lysine
    'AAG':'K',    # Lysine
    'AGC':'S',    # Serine
    'AGT':'S',    # Serine
    'AGA':'R',    # Arginine
    'AGG':'R',    # Arginine
    'CTA':'L',    # Leucine
    'CTC':'L',    # Leucine
    'CTG':'L',    # Leucine
    'CTT':'L',    # Leucine
    'CCA':'P',    # Proline
    'CCC':'P',    # Proline
    'CCG':'P',    # Proline
    'CCT':'P',    # Proline
    'CAC':'H',    # Histidine
    'CAT':'H',    # Histidine
    'CAA':'Q',    # Glutamine
    'CAG':'Q',    # Glutamine
    'CGA':'R',    # Arginine
    'CGC':'R',    # Arginine
    'CGG':'R',    # Arginine
    'CGT':'R',    # Arginine
    'GTA':'V',    # Valine
    'GTC':'V',    # Valine
    'GTG':'V',    # Valine
    'GTT':'V',    # Valine
    'GCA':'A',    # Alanine
    'GCC':'A',    # Alanine
    'GCG':'A',    # Alanine
    'GCT':'A',    # Alanine
    'GAC':'D',    # Aspartic Acid
    'GAT':'D',    # Aspartic Acid
    'GAA':'E',    # Glutamic Acid
    'GAG':'E',    # Glutamic Acid
    'GGA':'G',    # Glycine
    'GGC':'G',    # Glycine
    'GGG':'G',    # Glycine
    'GGT':'G',    # Glycine
    'TCA':'S',    # Serine
    'TCC':'S',    # Serine
    'TCG':'S',    # Serine
    'TCT':'S',    # Serine
    'TTC':'F',    # Phenylalanine
    'TTT':'F',    # Phenylalanine
    'TTA':'L',    # Leucine
    'TTG':'L',    # Leucine
    'TAC':'Y',    # Tyrosine
    'TAT':'Y',    # Tyrosine
    'TAA':'_',    # Stop
    'TAG':'_',    # Stop
    'TGC':'C',    # Cysteine
    'TGT':'C',    # Cysteine
    'TGA':'_',    # Stop
    'TGG':'W',    # Tryptophan
}

MAX_SEQ_LINE = 70

NONDECIMAL = re.compile(r'[^\d]+')

VALID_CHROMOSOMES = map(str,range(1,23))+['X','Y','MT']

MAX_MISS_RATE = .05

###############################################################################
# FUNCTIONS                                                                   #
###############################################################################

def _getSeqs(filename, includelist=[]):
  """Given a fasta filename, return a dictionary of ID -> seq"""

  if not os.path.isfile(filename): 
    sys.stderr.write('Could not open '+filename+'\n')
    return

  FILE = gzip.open(filename) if filename.endswith('gz') else open(filename)

  allseqs = {}

  currID = ''
  currseq = []

  for l in FILE:
    if l.startswith('>'):
      if len(currseq)>0 and (len(includelist)<1 or currID in includelist):
        allseqs[currID] = ''.join(currseq)
      currID = l.strip().split()[0][1:]
      currseq = []
    elif len(includelist)<1 or currID in includelist:
      currseq.append(l.strip())
  if len(currseq)>0 and (len(includelist)<1 or currID in includelist):
    allseqs[currID] = ''.join(currseq)

  return allseqs


###############################################################################

def getComplement(seq):
  """
  get the reverse complement of a DNA string
  """

  reverseComplement = []

  ## reverse the string
  seq = seq[::-1]

  ## get the complement
  for base in seq: 
    reverseComplement.append(COMPLEMENT[base])

  return ''.join(reverseComplement)
  
###############################################################################

def getSeqExons(seq, cds):
  """
  stitch together the exons -- if we have NEGATIVE indices, we need to SKIP!
  """

  # check if complementation is needed
  complement = 'complement' in cds
  if complement:
    cds = cds.replace("complement(", "")

  # get the exons
  cds = cds.replace("join(", "").replace(")", "").split(",")
  #if complement:
  #  cds.reverse()
    
  rnaSeq = []
  dnaSeq = collections.OrderedDict()
    
    
  for exon in cds:
    if '..' in exon:
      beg, end = exon.split("..")
      beg = int(NONDECIMAL.sub('', beg))
      end = int(NONDECIMAL.sub('', end))
    else:
      beg = end = int(NONDECIMAL.sub('', exon))
    
    try:
      dnasubseq = seq[beg-1:end]
    except:
      dnasubseq = ''
    
    if exon.startswith('-'): # e.g., -3..-3, meaning we need 3 nucleotides here!
      rnaSeq.append('N'*beg)
      dnaSeq[str(beg)+':'+str(end)] = 'N'*beg
    
    elif complement:
      rnaSeq.append(getComplement(dnasubseq))
      dnaSeq[str(end)+':'+str(beg)] = dnasubseq[::-1]
    else:
      rnaSeq.append(dnasubseq)
      dnaSeq[str(beg)+':'+str(end)] = dnasubseq
  
  rnaSeq = ''.join(rnaSeq)
  protSeq = translateRNA(rnaSeq.upper())

  return dnaSeq, rnaSeq, protSeq


###############################################################################

def translateRNA(rnaSeq):
  """
  translate the cDNA sequence to protein using GENCODE; unknown codons 
  or accidental stop codons are translated as 'X'
  """

  protSeq = []
  for i in range(0, len(rnaSeq), 3):
    codon = rnaSeq[i:i+3]
    if len(codon) == 3:
      if 'N' in codon:
        protSeq.append('X')
      elif GENCODE[codon]=="_" and i<len(rnaSeq)-5: # stop codon somewhere in the middle..
        protSeq.append('X')
      else:
        protSeq.append(GENCODE[codon])
  return ''.join(protSeq)

###############################################################################

def transcribeDNA(dnaSeq):
  """
  return one cDNA sequence from the set of exons (in order)
  """

  rnaSeq = []
  for exonRange, exon in dnaSeq.iteritems():
    rnaSeq.append(exon)
  return ''.join(rnaSeq)

###############################################################################

def matchProts(protSeq1, protSeq2, maxMissRate=0.05):
  """
  determine if two protein sequences match well enough. 
  """

  misses = 0.0
  if len(protSeq1) != len(protSeq2):
    return False
  for i in xrange(len(protSeq1)):
    if protSeq1[i] != protSeq2[i]:
      misses += 1
  return misses/len(protSeq1) <= maxMissRate

###############################################################################

def alignToRNA(x, y):
  """
  align sequence X to sequence Y
  """
    
  # alignment MATRIX
  M = [[0]*(len(y)+1) for i in xrange(len(x)+1)]

  for i in xrange(1, len(x)+1):
    diff = 0 if i==len(x) else 1
    
    for j in xrange(1, len(y)+1):
      M[i][j] = max(M[i-1][j-1] + score_bp(x[i-1], y[j-1]), # allow for a mismatch
                    M[i-1][j] - 1, # incorporate a gap 
                    M[i][j-1] - diff,
                    0) # never go below 0
            
  opt = 0
  xLoc = 0
  yLoc = 0
  j = len(y)
    
  for i in range(len(x)+1):
    if M[i][j]>opt:
      opt = M[i][j]
      xLoc = i
      yLoc = j

  dnaSeq = []
  while M[xLoc][yLoc]!=0:
    diff = 0 if xLoc == len(x) else 1

    if M[xLoc][yLoc] == M[xLoc-1][yLoc] - 1:
      dnaSeq.insert(0,'[' + x[xLoc-1] + ']')
      xLoc -= 1
    elif M[xLoc][yLoc] == M[xLoc][yLoc-1] - diff:
      dnaSeq.insert(0, '[' + y[yLoc-1].lower() + ']')
      yLoc -= 1
    elif M[xLoc][yLoc] == M[xLoc-1][yLoc-1] -1:
      dnaSeq.insert(0, x[xLoc-1].lower())
      yLoc -= 1
      xLoc -= 1
    else:
      dnaSeq.insert(0, x[xLoc-1])
      yLoc -= 1
      xLoc -= 1

  while yLoc!=0 and xLoc!=0:
    dnaSeq.insert(0, x[xLoc-1].lower())
    yLoc -= 1
    xLoc -= 1
  while yLoc!=0:
    dnaSeq.insert(0, '[' + y[yLoc-1].lower() + ']')
    yLoc -= 1
  while xLoc!=0:
    dnaSeq.insert(0, '[' + x[xLoc-1] + ']')
    xLoc -= 1

  return ''.join(dnaSeq), opt # return the alignment and the score

############################################################################### 

def score_bp(x, y):
  """
  internal score function, matches score 2 and mismatches/gaps score -1
  """

  return 2 if x==y else -1

###############################################################################	

def trimRnaLoc(dnaSeq, s):
    
  newDnaSeq = collections.OrderedDict()
  for exonRange, exon in dnaSeq.iteritems():
    beg = int(exonRange.split(":")[0])
    end = int(exonRange.split(":")[1])
    if beg>end:
      diff = -1
    else:
      diff = 1

    if s<len(exon)-1 and s>0:
      newDnaSeq[str(beg)+':'+str(beg+(s-1)*diff)] = exon[:s]
      newDnaSeq[str(beg+(s+1)*diff)+':'+str(end)] = exon[s+1:]
    elif s==0:
      newDnaSeq[str(beg+diff)+':'+str(end)] = exon[1:]
    elif s==len(exon)-1:
      newDnaSeq[str(beg)+':'+str(end-diff)] = exon[:-1]
    else:
      newDnaSeq[exonRange] = exon

    s = s - len(exon)

  return newDnaSeq

############################################################################### 

def trimDnaLoc(dnaSeq, s):
    
  newDnaSeq = collections.OrderedDict()
  for exonRange, exon in dnaSeq.iteritems():
    beg = int(exonRange.split(":")[0])
    end = int(exonRange.split(":")[1])
    if beg>end:
      diff = -1
    else:
      diff = 1

    if s>min(beg,end) and s<max(beg,end):
      newDnaSeq[str(beg)+':'+str(s-diff)] = exon[:abs(beg-s)]
      newDnaSeq[str(s+diff)+':'+str(end)] = exon[abs(beg-s)+1:]
    elif s==beg:
      newDnaSeq[str(s+diff)+':'+str(end)] = exon[1:]
    elif s==end:
      newDnaSeq[str(beg)+':'+str(s-diff)] = exon[:-1]
    else:
      newDnaSeq[exonRange] = exon

  return newDnaSeq


###############################################################################

def buildCDS(dnaSeq, bestAlDNA):

  complement = False
  for exonRange, exon in dnaSeq.iteritems():
    beg = int(exonRange.split(":")[0])
    end = int(exonRange.split(":")[1])
    if beg>end:
      complement = True
      break

  if complement: # reverse the inputted sequence
    compAlDNA = []
    for base in bestAlDNA:
      if base in COMPLEMENT:
        compAlDNA.append(COMPLEMENT[base])
      else:
        compAlDNA.append(base)
    bestAlDNA = ''.join(compAlDNA)

  newDnaSeq = collections.OrderedDict()
  i = 0
  key = -1
  for exonRange, exon in dnaSeq.iteritems():
    beg = int(exonRange.split(":")[0])
    end = int(exonRange.split(":")[1])
    if beg>end:
      diff = -1
    else:
      diff = 1

    currentExon = ""

    while i<len(bestAlDNA) and len(currentExon) <= abs(end-beg):

      while i<len(bestAlDNA) and bestAlDNA[i] == '[':

        if currentExon != "":
          newDnaSeq[str(beg)+':'+str(beg+(len(currentExon)-1)*diff)] = currentExon
          beg = beg+len(currentExon)*diff
          currentExon = ""

        if bestAlDNA[i+1].islower(): # what is inside the []? 
          if newDnaSeq.has_key(str(key)+':'+str(key)):
            newDnaSeq[str(key)+':'+str(key)] = newDnaSeq[str(key)+':'+str(key)] + bestAlDNA[i+1].upper()
          else:
            newDnaSeq[str(key)+':'+str(key)] = bestAlDNA[i+1].upper()
        else:
          if currentExon != "":
            newDnaSeq[str(beg)+':'+str(beg+(len(currentExon)-1)*diff)] = currentExon
            currentExon = ""
          beg = beg+(len(currentExon)+1)*diff
        i = i + 3

      key -= 1
      
      if i<len(bestAlDNA): # we are still building up this exon!
        currentExon += bestAlDNA[i].upper()
        i+=1
        
    # exited the while loop, is there a new exon to write out!?
    if currentExon != "":
      newDnaSeq[str(beg)+':'+str(beg+(len(currentExon)-1)*diff)] = currentExon

  # we never actually reached the end of the sequence, so make this change...??
  if i<len(bestAlDNA):
    newDnaSeq[str(key)+':'+str(key)] = bestAlDNA[i:].upper().replace('[','').replace(']','')
    
  return newDnaSeq

###############################################################################

def returnloc(dnaSeq):
  exons = sorted([(int(a.split(':')[0]), int(a.split(':')[1])) for a in dnaSeq.keys()])

  complement = False
  for start,end in exons: 
    if start>end: 
      complement = True
      break
  if complement: 
    exons = exons[::-1] # reverse the list, so we go LARGEST to SMALLEST index
    allexons = ','.join([str(a[1])+'..'+str(a[0]) for a in exons])
    if ',' in allexons: allexons = 'join('+allexons+')'
    allexons = 'complement('+allexons+')'
  else:
    allexons = ','.join([str(a[0])+'..'+str(a[1]) for a in exons])
    if ',' in allexons: allexons = 'join('+allexons+')'
  return allexons

###############################################################################

def _getChromosomeSeq(chromID, seqfile=''):
  """
  return the sequence for a particular chromosome
  """

  if seqfile=='': 
    seqfile=DBPATH+'ensembl/dna/Homo_sapiens.GRCh37.75/Homo_sapiens.GRCh37.75.dna_sm.chromosome.'+chromID+'.fa.gz'
  
  if not os.path.isfile(seqfile):
    sys.stderr.write('Could not find '+seqfile+'\n')
    sys.exit(1)
  
  x = gzip.open(seqfile) if seqfile.endswith('.gz') else open(seqfile)
  x.next() # skip the starting FASTA header line
  seq = []
  for l in x: seq.append(l.strip())
  return ''.join(seq)


############################################################################### 

def storeChromosome(chromID):
  """
  store the sequence of the chromosome and the CDS data for all genes
  """

  ## get the sequence for the appropriate chromosome
  seq = _getChromosomeSeq(chromID)

  cds = set()
  ## get the GENE LOCATIONS for all proteins in that appropriate chromosome. 
  for l in gzip.open(DBPATH+'tcga/Homo_sapiens.GRCh37.75.pep.all.hugosymbols.fa.gz'): 
    if l.startswith('>'):
      hugoProt     = l.strip().split()[0][1:] # e.g., A1BG.001
      ensemblProt  = l[l.find('prot:')+5:].strip().split()[0] # e.g., ENSP00000263100
      ensemblGene  = l[l.find('gene:')+5:].strip().split()[0] # e.g., ENSG00000121410
      ensemblCdna  = l[l.find('transcript:')+11:].strip().split()[0] # e.g., ENST00000263100
      refseqID     = l[l.find('refseq:')+7:].strip().split()[0] # e.g., NP_570602
      chromosome   = l[l.find('chromosome:GRCh37.75:')+21:].strip().split()[0].split(':')[0] # e.g., 19
      # e.g., complement(join(58864770..58864658,58864658..58864693))
      instructions = l[l.find('chromosome:GRCh37.75:')+21:].strip().split()[0].split(':')[1]
      
      if chromosome==chromID:
        cds.add((hugoProt,ensemblProt,ensemblGene,ensemblCdna,refseqID,instructions))

  #return seq, cds, mRNAs # store chromosome sequences, all coding instructions, and list of transcripts per gene...?
  return seq, sorted(list(cds))
  

###############################################################################

def storeNPSeqs(npSeqsFileName, getseqID=lambda x:x.split('|')[3].split('.')[0]):
  """
  store the protein sequences
  """
  getseqID = lambda x:x.strip().split()[0][1:]  

  npSeqs = {}
  protID = ""
  seq = []
  for line in open(npSeqsFileName):
    if line.startswith('>'):
      if len(seq) > 0: # store sequence
        npSeqs[protID] = ''.join(seq)
      seq = []
      protID = getseqID(line)
    else:
      seq.append(line.strip())

  npSeqs[protID] = ''.join(seq)

  return npSeqs


###############################################################################
def writeSeq(build, chromID, seq, geneID, mapping, currprotSeqs, currcdnaSeqs, exonDir=DBPATH+'hgnc/'):
#def writeSeq(seq, cdsInfo, mRNAs, npSeqs, rnaSeqs, geneID, dnaSeqDir, rnaSeqDir, protSeqDir, chromosome):
  """
  mapping = ensemblProtID -> (ensemblCdnaID, loc, linedescription)
  write the protein sequence of the gene to file and return the isoforms that
  have been written successfully
  """

  # ensemblProtID -> (ensemblGeneID, ensemblCdnaID, refseqID, loc) 
    
  PROGRESS = open("missesAndFixes.txt", "a")
  
  ## make sure the file doesn't already exist
  i = 0
  for ensemblProtID in sorted(mapping.keys()):
    # if there is no available protein sequence, move on
    
    if ensemblProtID in currprotSeqs: 

      # sorted list of exon locs and seqs, full cdna seq, full prot seq
      dnaSeq, rnaSeq, protSeq = getSeqExons(seq, mapping[ensemblProtID][1])
    
      #trim stop codon
      protSeq = protSeq[:-1]
    
      bestRNA = mapping[ensemblProtID][0] # best cDNA ID
    
      # make sure the transcripts match
      fix = False
      if not matchProts(protSeq, currprotSeqs[ensemblProtID], MAX_MISS_RATE):

        bestOpt    = 0  # best alignment score        
        bestRNASeq = "" # best cDNA sequence
        bestAlDNA  = "" # 
        
        # check and see if we're actually only considering some sort of subsequence...??
        for ensemblCdnaID, cdnaSeq in currcdnaSeqs.items():
          for s in xrange(len(cdnaSeq)-(3*len(currprotSeqs[ensemblProtID]))+1):
            
            # get the protein sequence from translating this cDNA sequence
            thisprot = translateRNA(cdnaSeq[s:s+3*len(currprotSeqs[ensemblProtID])].upper())
            if thisprot==currprotSeqs[ensemblProtID]: 
                
              # obviously something was wrong with what we actually pulled from the chromosome file
              # according to the CDS exon positions, so figure out how to fix that
              # usually we're off by up to 2 basepairs (frameshift), which is why we +3 at the end
              alDNA, thisOpt = alignToRNA(rnaSeq, cdnaSeq[s:s+3*len(currprotSeqs[ensemblProtID])+3])
              if thisOpt > bestOpt:
                bestAlDNA = alDNA
                bestOpt = thisOpt
                bestRNA = ensemblCdnaID
                bestRNASeq = cdnaSeq[s:s+3*len(currprotSeqs[ensemblProtID])+3]
                
        if bestOpt>2*(1-MAX_MISS_RATE)*len(rnaSeq):
          dnaSeq = buildCDS(dnaSeq, bestAlDNA) # new ordered list of fixed exons
          rnaSeq = transcribeDNA(dnaSeq)
          protSeq = translateRNA(rnaSeq.upper())[:-1]
          fix = True

        PROGRESS.write('\t'.join([geneID,bestRNA,str(bestOpt),str(fix)])+'\n'+bestAlDNA+'\n')
            
      if matchProts(protSeq, currprotSeqs[ensemblProtID], MAX_MISS_RATE) or fix: # if it already matched OR we fixed it
        if not os.path.isdir(exonDir+chromID): os.system('mkdir '+exonDir+chromID)
        if not os.path.isdir(exonDir+chromID+'/'+geneID): os.system('mkdir '+exonDir+chromID+'/'+geneID)

        # we want to keep track of the PROTEIN SEQUENCE, the CDNA SEQUENCE, and also the FIXED EXONS!!!!
        if False: newProtID = geneID+'.'+str(i+1).zfill(3)
        else: newProtID = ensemblProtID
        
        for seqtype,suffix in [(protSeq,'prot'),(rnaSeq,'cdna')]:
          if not os.path.isdir(exonDir+chromID): os.system('mkdir '+exonDir+chromID)
          if not os.path.isdir(exonDir+chromID+'/'+geneID): os.system('mkdir '+exonDir+chromID+'/'+geneID)
          if not os.path.isdir(exonDir+chromID+'/'+geneID):
            sys.stderr.write('Could not create directory '+exonDir+chromID+'/'+geneID+'/ !!!!\n')
            continue
            
          # Write out the correct PROTEIN and complete CDNA sequence! 
          SEQFILE = open(exonDir+chromID+'/'+geneID+'/'+newProtID+'.'+suffix+'.fa','w')
            
          SEQFILE.write('>'+newProtID+' prot:'+ensemblProtID+' ')
          
          # e.g., gene:ENSGxx refseq:X hugoID:X
          descriptors = mapping[ensemblProtID][2].split()[1:] + ['length:'+str(len(seqtype))]
          SEQFILE.write(' '.join(['chromosome:'+build+':'+chromID+':'+returnloc(dnaSeq) if d.startswith('chromosome') else d for d in descriptors])+'\n'+seqtype+'\n\n')        
          SEQFILE.close()
        
        EXONFILE = open(exonDir+chromID+'/'+geneID+'/'+newProtID+'.exons.txt','w')
        EXONFILE.write('>'+newProtID+' prot:'+ensemblProtID+' ')
        descriptors = mapping[ensemblProtID][2].split()[1:] + ['length:'+str(len(rnaSeq))]
        EXONFILE.write(' '.join(['chromosome:'+build+':'+chromID+':'+returnloc(dnaSeq) if d.startswith('chromosome') \
                                 else d for d in descriptors])+'\n')
        for exonRange, exon in dnaSeq.iteritems(): EXONFILE.write(exonRange+'\t'+exon+'\n')
        EXONFILE.close()
        
        i += 1

  PROGRESS.close()

###############################################################################

def ensembltoHGNC(hgncfile=DBPATH+'ensembl/Homo_sapiens.GRCh37/Homo_sapiens.GRCh37.toHGNC.tsv', 
                  refseqfile=DBPATH+'ensembl/Homo_sapiens.GRCh37/Homo_sapiens.GRCh37.toRefSeq.tsv',
                  entrezfile=DBPATH+'ensembl/Homo_sapiens.GRCh37/Homo_sapiens.GRCh37.toEntrez.tsv',
                  locfile=DBPATH+'ensembl/Homo_sapiens.GRCh37/Homo_sapiens.GRCh37.genelocs.tsv'):
  """Return a mapping of Ensembl ProtIDs to Hugo Gene Names and RefSeq IDs"""
  tohgncID, tohgncsymbol, torefseq, toentrez, togene, totranscript, toloc = {},{},{},{},{},{},{}

  with open(hgncfile) as x:
    x.next()
    for l in x:
      ensemblgene, ensemblprot, hgncID, hgncSymbol, geneName = l[:-1].split('\t')[:5]
      if hgncID!='':
        if ensemblprot!='': tohgncID[ensemblprot] = hgncID
        if ensemblgene!='': tohgncID[ensemblgene] = hgncID
            
      if hgncSymbol!='':
        if ensemblprot!='': tohgncsymbol[ensemblprot] = hgncSymbol
        if ensemblgene!='': tohgncsymbol[ensemblgene] = hgncSymbol
      elif geneName!='': # only IF hgncSymbol is blank!
        if ensemblprot!='': tohgncsymbol[ensemblprot] = geneName
        if ensemblgene!='': tohgncsymbol[ensemblgene] = geneName
  
  with open(entrezfile) as x:
    x.next()
    for l in x:
      ensemblprot, entrezid = l[:-1].split('\t')[:2]
      if ensemblprot!='' and entrezid!='': 
        toentrez[ensemblprot] = entrezid
  
  with open(refseqfile) as x:
    x.next()
    for l in x:
      if not l.strip().startswith('ENSP') or len(l.strip().split())<2: continue
      ensemblprot, refseqid = l.strip().split()[:2] # this automatically excludes the PREDICTED ID, if there is one
      torefseq[ensemblprot] = refseqid
        
  exons = {}
  with open(locfile) as x:
    x.next()
    for l in x:
      if len(l.strip().split('\t')) < 8: continue
      ensemblgene, ensembltrans, ensemblprot, chrom, strand, rank, start, end = l.strip().split('\t')[:8]
      if ensemblprot not in exons: exons[ensemblprot] = set()
      exons[ensemblprot].add((int(rank), int(start), int(end), chrom, strand))
      togene[ensemblprot] = ensemblgene
      totranscript[ensemblprot] = ensembltrans

  for pID,allx in sorted(exons.items()):
    loc = ','.join([str(exon[1])+'..'+str(exon[2]) for exon in sorted(list(allx))])
    if ',' in loc: loc = 'join('+loc+')'
    if '-' in list(allx)[0][4]: loc = 'complement('+loc+')'        
    toloc[pID] = 'GRCh37:'+list(allx)[0][3]+':'+loc
        
  return tohgncID, tohgncsymbol, torefseq, toentrez, togene, totranscript, toloc


###############################################################################


def _update_fasta_genelocs_altids(infile=DBPATH+'ensembl/pep/Homo_sapiens.GRCh37.pep.all.fa.gz'):
  """Add gene locations and all alternate IDs to an existing fasta file."""

  tohgncID, tohgncsymbol, torefseq, toentrez, togene, totranscript, toloc = ensembltoHGNC()
  NEWOUT = gzip.open(infile.replace('.pep.all.fa.gz','.pep.all.withgenelocs.fa.gz'),'w')

  head = ''
  seq = []
  for l in gzip.open(infile):
    if l.startswith('>'): 
      if len(seq)>0: NEWOUT.write(head+'\n'+''.join(seq)+'\n')
      pepID = l.strip().split()[0][1:]
      desc = {a[:a.find(':')]:a[a.find(':')+1:] for a in l.strip().split()[1:]}
      geneID = desc.get('gene',togene.get(pepID.split('.')[0],'UNKNOWN'))
        
      seq = []      
      head = ' '.join(['>'+pepID, 
                            'pep:'+desc.get('pep','unknown'), 
                            'chromosome:'+toloc.get(pepID.split('.')[0], desc.get('chromosome','UNKNOWN')),
                            'gene:'+geneID,
                            'transcript:'+desc.get('transcript', totranscript.get(pepID.split('.')[0],'UNKNOWN')),
                            'gene_biotype:'+desc.get('gene_biotype','UNKNOWN'),
                            'transcript_biotype:'+desc.get('transcript_biotype','UNKNOWN'),
                            'hgncID:'+tohgncID.get(pepID.split('.')[0], tohgncID.get(geneID.split('.')[0], 'UNKNOWN')),
                            'hugoSymbol:'+tohgncsymbol.get(pepID.split('.')[0], tohgncsymbol.get(geneID.split('.')[0],'UNKNOWN'))]+\
                      (['refseq:'+torefseq[pepID.split('.')[0]]] if pepID.split('.')[0] in torefseq else [])+\
                      (['entrez:'+toentrez[pepID.split('.')[0]]] if pepID.split('.')[0] in toentrez else []))
    else: seq.append(l.strip())
        
  if len(seq) > 0: 
    NEWOUT.write(head+'\n'+''.join(seq)+'\n')
  NEWOUT.close()
  print infile.replace('.pep.all.fa.gz','.pep.all.withgenelocs.fa.gz')
    
    
    
###############################################################################    

def _getMapping(infile=DBPATH+'ensembl/pep/Homo_sapiens.GRCh37.pep.all.withgenelocs.fa.gz'):
  """Return a dictionary of chromID -> ensemblGeneID -> ensemblProtID -> (ensemblCDNA, loc)
  so that we can create the 'exon' files later"""

  mapping = {}

  for l in gzip.open(infile):
    if l.startswith('>'):
      ensemblProt  = l.strip().split()[0][1:] # e.g., ENSP00000263100
      ensemblGene  = l[l.find('gene:')+5:].strip().split()[0] # e.g., ENSG00000121410
      ensemblCdna  = l[l.find('transcript:')+11:].strip().split()[0] # e.g., ENST00000263100
    
      # e.g., GRCh37, 19, complement(join(58864770..58864658,58864658..58864693))
      build,chromosome,instructions   = l[l.find('chromosome:')+11:].strip().split()[0].split(':')[:3]
    
      if chromosome not in mapping: mapping[chromosome] = {}
      if ensemblGene not in mapping[chromosome]: mapping[chromosome][ensemblGene] = {}
      mapping[chromosome][ensemblGene][ensemblProt] = (ensemblCdna, instructions, l.strip())
  # chromID -> ensemblGeneID -> ensemblProtID -> (ensemblCdnaID, loc, descriptionline)
  return mapping          


###############################################################################

def _createnonredundant(infile=DBPATH+'ensembl/pep/Homo_sapiens.GRCh37.pep.all.withgenelocs.fa.gz'):
  """Create a nonredundant version of an input file (to run on fewer sequences..."""

  seqs = {}
  OUT = gzip.open(infile.replace('.fa.gz','_nonredundant.fa.gz'),'w')
  with gzip.open(infile) as x:
    for l in x:
      if l.startswith('>'):
        seq = x.next().strip()
        protID = l.strip().split()[0][1:]        
        if seq not in seqs: seqs[seq] = set()
        seqs[seq].add(protID)
  
  idlen = len(str(len(seqs.keys())))
  for i,seq in enumerate(sorted(seqs.keys())):
    if len(seq)>0:
      OUT.write('>seq'+str(i).zfill(idlen)+' '+':'.join(sorted(list(seqs[seq])))+'\n'+seq+'\n\n')
  OUT.close()

###############################################################################
  
def _fixtoplevel(infile=DBPATH+'ensembl/dna/Homo_sapiens.GRCh37/Homo_sapiens.GRCh37.dna_sm.toplevel.fa.gz'):
  """Given a fasta filename, return a dictionary of ID -> seq"""

  if not os.path.isfile(infile): 
    sys.stderr.write('Could not open '+infile+'\n')
    return

  currID = ''

  with gzip.open(infile) as x:
    for l in x:
      if l.startswith('>'):
        if currID !='': 
          OUT.write('\n')
          OUT.close()
          print DBPATH+'ensembl/dna/Homo_sapiens.GRCh37/Homo_sapiens.GRCh37.dna_sm.'+currID+'.fa.gz'
        
        currID = l.strip().split()[0][1:]
        OUT = gzip.open(DBPATH+'ensembl/dna/Homo_sapiens.GRCh37/Homo_sapiens.GRCh37.dna_sm.'+currID+'.fa.gz','w')
        OUT.write(l)
            
      else:
        OUT.write(l.strip())
        
    OUT.write('\n')
    OUT.close()
    print DBPATH+'ensembl/dna/Homo_sapiens.GRCh37/Homo_sapiens.GRCh37.dna_sm.'+currID+'.fa.gz'
        
    
    

###############################################################################
# MAIN PROGRAM                                                                #
###############################################################################

if __name__=="__main__":
  
  if False:
    """Given an input FASTA file, create a new FASTA file where alternate IDs
    and location information has been added to the header sequences.
    infile = original fasta peptide file from Ensembl 
             e.g., DBPATH+'ensembl/pep/Homo_sapiens.GRCh37.pep.all.fa.gz' 
             
    We also need the following files obtained from BioMart:
    hgncfile=DBPATH+'ensembl/Homo_sapiens.GRCh37/Homo_sapiens.GRCh37.toHGNC.tsv', 
    refseqfile=DBPATH+'ensembl/Homo_sapiens.GRCh37/Homo_sapiens.GRCh37.toRefSeq.tsv',
    entrezfile=DBPATH+'ensembl/Homo_sapiens.GRCh37/Homo_sapiens.GRCh37.toEntrez.tsv',
    locfile=DBPATH+'ensembl/Homo_sapiens.GRCh37/Homo_sapiens.GRCh37.genelocs.tsv'
    
    (See write-up for more information)
    """
    
    _update_fasta_genelocs_altids()
    sys.exit(1)

  if False:
    """All the chromosome files can be downloaded from Ensembl. The separate 
    files only include chromosomes 1-22, X, Y, MT though. In order to get the 
    same fasta files for non-chromosomal DNA, the "top level" file must be 
    downloaded and processed instead.
    
    infile = DBPATH+'ensembl/dna/Homo_sapiens.GRCh37/Homo_sapiens.GRCh37.dna_sm.toplevel.fa.gz'
    """
    _fixtoplevel()
    sys.exit(1)
    
    
  if True:
    """Given an input FASTA file WITH GENE LOCATION and other alternate ID 
    information, verify all the sequences. We MUST have the soft-masked DNA 
    sequences and the cDNA sequences downloaded from Ensembl, too."""
    
    build = 'GRCh37'
    exonDir = DBPATH+'ensembl/Homo_sapiens.'+build+'/' 
    
    # (1) get the mapping (by chromosome)
    #     # chromID -> ensemblGeneID -> ensemblProtID -> (ensemblCdnaID, loc, descriptionline)
    mapping  = _getMapping(DBPATH+'ensembl/pep/Homo_sapiens.'+build+'.pep.all.withgenelocs.fa.gz')
    
    ##build protein sequences
    if not VALID_CHROMOSOMES:
      VALID_CHROMOSOMES = [a.replace('.fa.gz','').split('.')[-1] \
                           for a in os.listdir(DBPATH+'ensembl/dna/Homo_sapiens.'+build) if a.endswith('.fa.gz')]

    # (2) store all protein sequences by Ensembl Prot ID
    protSeqs = _getSeqs(DBPATH+'ensembl/pep/Homo_sapiens.'+build+'.pep.all.withgenelocs.fa.gz')

    # (3) store all cDNA sequences by Ensembl Trans ID
    cdnaSeqs = _getSeqs(DBPATH+'ensembl/cds/Homo_sapiens.'+build+'.cds.all.fa.gz')
    
    for chromID in sorted(VALID_CHROMOSOMES):
      if chromID not in mapping: continue # we don't have any genes on this chromosomal nubbin
        
      if not os.path.isdir(exonDir+chromID): os.system('mkdir '+exonDir+chromID)
        
      # (4) store the DNA sequence for that chromosome
      # we do some funny stuff with the chromID just incase there are multiple sequences
      # in the same file... which there shouldn't be.
      seq = _getSeqs(DBPATH+'ensembl/dna/Homo_sapiens.'+build+\
                     '/Homo_sapiens.'+build+'.dna_sm.'+chromID+'.fa.gz', [chromID])[chromID]
      

      for ensemblGene in sorted(mapping[chromID].keys()):
        
        if not os.path.isdir(exonDir+chromID+'/'+ensemblGene): 
          os.system('mkdir '+exonDir+chromID+'/'+ensemblGene)        
        
        currprotSeqs = {ID:protSeqs[ID].replace('*','X') for ID in mapping[chromID][ensemblGene].keys()}
        currcdnaSeqs = {ID:cdnaSeqs[ID] for ID in [a[0] for a in mapping[chromID][ensemblGene].values()]}
        
        sys.stdout.write(chromID+'\t'+ensemblGene+'\n')
            writeSeq(build, chromID, seq, ensemblGene, mapping[chromID][ensemblGene], currprotSeqs, currcdnaSeqs, cd Y)
        
  if False:
    """Some garbage"""
    _createnonredundant()
    sys.exit(1)
        
    
    