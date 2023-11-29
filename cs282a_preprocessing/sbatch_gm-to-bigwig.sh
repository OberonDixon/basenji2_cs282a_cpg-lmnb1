#!/bin/bash
#SBATCH --job-name=gm-to-bigwig
#SBATCH --account=fc_nilah
#SBATCH --partition=savio2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=02:00:00
#SBATCH --output=/clusterfs/nilah/oberon/basenji2_cs282a_cpg-lmnb1/cs282a_preprocessing/slurm_logs/gm-to-bigwig_%A_%a.out
#SBATCH --error=/clusterfs/nilah/oberon/basenji2_cs282a_cpg-lmnb1/cs282a_preprocessing/slurm_logs/gm-to-bigwig_%A_%a.err
#SBATCH --array=1-23
# Command(s) to run:
module load python

# Use SLURM_ARRAY_TASK_ID as the chromosome number, with 23 mapped to "X"
if [ "${SLURM_ARRAY_TASK_ID}" == "23" ]; then
    chr="X"
else
    chr="${SLURM_ARRAY_TASK_ID}"
fi

export LD_LIBRARY_PATH=/clusterfs/nilah/oberon/python/lib/:${LD_LIBRARY_PATH}

conda run -n dimelo_arbitrary_basemod_dev /clusterfs/nilah/oberon/basenji2_cs282a_cpg-lmnb1/cs282a_preprocessing/process_dimelo-to-bigwig.py --chromosome chr${chr} --output_folder /clusterfs/nilah/oberon/datasets/cs282a/gm12878/chr${chr}/ --bam_file /clusterfs/nilah/oberon/datasets/dimelo_mA-mGpC/20230702_jm_lmnb1_acessibility_redux/megalodon_all_context/mod_mappings.01.sorted.bam --name gm12878_dimelo_feb2022 --reference_genome /clusterfs/nilah/oberon/genomes/chm13.draft_v1.1.fasta