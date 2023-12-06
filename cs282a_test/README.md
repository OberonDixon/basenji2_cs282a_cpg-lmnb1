### How to run a test:
Requires about 4GB of disk space and may work best running Linux.

The fourth step was to run inference for the full test/train/validation dataset and save the results to .h5 files. For the three large bin models, these are saved in the cs282a_test folder. For the linear probe, the file is about 2GB and available here under the name probe_first_full_run.h5: https://drive.google.com/drive/folders/1ZA8nlNrMW8K0ATsqDp4TM_5mEDTlK9Gs?usp=drive_link.

cs282a_test contains a small toy example for running inference but one can also re-run large fractions of the genome. On a high-performance computer it should be very quick, bottlenecked by loading embeddings. 

The fifth step was to load up the whole-genome inference files and create scatterplots, genomic track plots, and simple biological analyses. We also calculated head-to-head Pearson and Spearman correlations. It is evident that the three large bin models perform about the same, while the linear probe (which has a harder, noisier task AND fewer parameters to work with) performs worse, but still clearly captures some of the major trends in the test set.

#### Setup

Clone the repo to your machine (size is about 1.6GB), 'git clone https://github.com/OberonDixon/basenji2_cs282a_cpg-lmnb1/'. To set up your environment, create a conda environment with python=3.8, activate it, and run `pip install -r requirements.txt` from within the basenji2_cs282a_cpg-lmnb1 directory. We have tested this on Linux and it may have problems on Mac or Windows.

Then, set up the environment to be used as a jupyter kernel and download the datasets specified in the visualization_tracks.ipynb notebook into the cs282a_test folder (total size is about 2.2GB). 

python -m ipykernel install --user --name cs282a_test --display-name "cs282a_test"

Peer reviewers have two simple tests available, along with all the code for running inference to get embeddings and training feature extraction models: 

1) Run inference with sample_inference.ipynb on each of the four models.

2) Use pre-loaded predictions to replicate biologically relevant genome plots using visualize_tracks.ipynb (this will require downloading some large-ish datasets). NOTE: if you have limited memory, you may want to run on a subset of chromosomes - simply take a slice of the list.



Our models are in pytorch, although if you want to run model inference on basenji with our code you'll need to create a basenji environment as described in the Basenji section. Also note that dependencies for these test files are different from those for some of the model training, which are again different from those for embeddings extraction, and different again from those for DiMeLo-seq data processing. Each should be done in a different conda environment if you want to run all of those. Other than the basenji package and the dimelo arbitrary_basemod_dev branch, you may also require various biopython installations such as pysam, pyBigWig, pyLiftOver, pybedtools, and so on.