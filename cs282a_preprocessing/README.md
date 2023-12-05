#### Preparing/preprocessing labels

The core functionality here is contained in 'process_cpg,lmnb1_by_bin.ipynb'. 

This processing requires a good deal of data, and the processing steps create intermediate outputs that will take up approximately 1TB on disk. It was not economical for us to provide a download link for these intermediate processed outputs, but they can be recreated using datasets linked below.

We downloaded the genomic tracks from Shah et al, 2023 (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9869549/, see data availability) and also two low read depth DiMeLo seq runs (https://www.nature.com/articles/s41592-022-01475-6). For dimelo data, .bam files (available here, https://drive.google.com/drive/folders/1ZA8nlNrMW8K0ATsqDp4TM_5mEDTlK9Gs?usp=drive_link) were converted to .bigwig files using 'process_dimelo-to-bigwig.py' run via 'sbatch_gm-to-bigwig.sh' and 'sbatch_hek-to-bigwig.sh' using the arbitrary basemod dev version of the dimelo package, https://github.com/OberonDixon/dimelo/tree/arbitrary_basemod_dev. 

This package is under development but the latest version on the linked branch can be installed on a Linux machine and is tested and validated in this use case.


Then in cs282a_preprocessing there is code to load these datasets in chunks per sequences.bed (including a coordinate transformation using pyLiftOver for DiMeLo seq data, which is aligned to a more complete/newer reference genome for which liftover chain files are available from UCSC), scale them to be approximately 0-500 (same as the preprocessing for Basenji2), create 128bp bins and 114688bp bins, and save to an hdf5 file called dataset_14-lmnb1_4-cpg.h5 (downloadable from https://drive.google.com/drive/folders/1ZA8nlNrMW8K0ATsqDp4TM_5mEDTlK9Gs?usp=drive_link).