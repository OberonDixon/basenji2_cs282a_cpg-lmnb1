#!/clusterfs/nilah/oberon/dimelo_arbitrary_basemod_dev/bin/python
import argparse
import dimelo as dm
import pysam

parser = argparse.ArgumentParser(description="This script runs the dimelo package to create bigwig files for a specified .bam reads file and chromosome.")

parser.add_argument("-c","--chromosome",help="chromosome to run",type=str)
parser.add_argument("-o","--output_folder",help="folder in which to save outputs",type=str)
parser.add_argument("-f","--bam_file",help="aligned reads to process with dimelo package",type=str)
parser.add_argument("-n","--name",help="prefix name for the output file",type=str)
parser.add_argument("-g","--reference_genome",help="path to the appropriate reference fasta file",type=str)

args = parser.parse_args()

chromosome = args.chromosome
output_folder = args.output_folder
filepath = args.bam_file
genome = args.reference_genome
sample_name = args.name + "_" + chromosome

cores = 24
bases_per_core = 10000000

genome_fasta = pysam.FastaFile(genome)
chr_len = genome_fasta.get_reference_length(chromosome)
region_str = f'{chromosome}:0-{chr_len}'
print(region_str)
print(sample_name,flush=True)

dm.parse_bam(
    filepath,
    sample_name,
    output_folder,
    basemods=('mA','CpG','GpC'),
    context_check_source = 'both',
    region=region_str,
    checkAgainstRef=True,
    pipeline='nanopore_megalodon',
    thresholds=[129,129,129],
    referenceGenome=genome,
    cores=cores,
    memory=cores*bases_per_core,
    formats_list=['bigwig']
)



