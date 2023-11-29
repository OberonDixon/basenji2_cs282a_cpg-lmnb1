# CS282A Project
#### Oberon Dixon-Luinenburg (BioE PhD student, Ioannidis and Streets labs), Stephanie Brener (BioE PhD student, Little lab), Jimin Jung (CS/DS), Ujjwal Krishnamurthi (CS/DS), Sriteja Vijapurapu (CS)

This project applies four methods of fine tuning on the pre-trained Basenji2 model to predict markers of gene expression from a full human genome sequence. We already conducted all embedding extraction and fine tuning training, and here provide code for our reviewers to replicate validation on test data and illustrate prediction accuracy via the descriptions below. 

#### Biology background:
There are two novel markers of gene expression that we sought to predict in this work. The first is CpG methylation, where large stretches of CGCG base pairs with a methyl group on the C located near DNA promoters lead to repressed gene expression, and no CpG methylation leads to more gene expression. The second is nuclear lamina association, where DNA that is closer to the edge of the nucleus experiences less gene expression, and DNA that is in the center of the nucleus has more gene expression. 
	Current models which seek to predict gene expression given a human genome sequence do not capture these known effects of CpG methylation and nuclear lamina association on gene expression. Our goal is to fine tune the pre-trained Basenji2 model with these new labels to improve overall gene expression prediction beyond the current ~80% accuracy by incorporating these effects.

#### Basenji2 model architecture:
This model has three parts:
7 CNN heads
CNN
GeLU
Max pool
11 Dilated CNN heads
Dilated CNN
Skip connection
1 Linear transform
1-D CNN

To prepare for fine tuning, we extracted the embeddings from the penultimate layer (before the linear transform) by running inference with the original data and saved model parameters. In fine tuning, we then used the same original data but added 18 new labels to the data which come from CpG methylation and nuclear lamina association tracks.

#### Our fine tuning approaches:
We applied three different fine tuning methods covering a range of complexity to see which method would lead to greater prediction accuracy using our new labels.

1) Linear probe

2) Convolutional perceptron

3) Max-pool perceptron

4) Transformer with self attention

For the linear probe, we adapted the final linear transform layer to do a 1-D CNN on our new labels. This sought to check if the pretrained model with no adaptations could accurately predict our new labels. We then sought to apply feature extraction by training three different smaller networks on top of the pre-trained Basenji model. In the first case, we created a convolutional perceptron and fed the pre-trained embeddings into it with the new labels. In the second case, we created a max-pool perceptron. In the third case, we created a transformer with self attention. Our goal with the first two is that they are a simpler network which should capture information from the new labels and be quicker networks to train. For the transformer, we expect this to have a more successful impact in prediction accuracy because it is a more comprehensive model which will look across the genomic sequence and due to self attention, incorporate more information shared across the genome. 

### How the Code Base Works

### General Overview

This code base is based on a fork of the Calico Basenji repository. Basenji provides code to build a class of convolutional models to predict gene expression from sequence, coming out of a series of publications starting with Basset in 2016 and continuing into 2023 with the new Borzoi model.

Models are constructed with the seqnn class. We used a parameter set provided with the cross2020 manuscript, dubbed Basenji2 in the corresponding paper (see the manuscripts folder which contains code to pull those parameters from the cloud). These models operate in tensorflow and can be run by following the Basenji installation instruments provided later in this readme file - however, if you only want to assess our fine tuning results it is in fact unnecessary to recapitulate the Basenji inference runs, which took many hours on the 38171-long datasets.

Basenji2 is trained using both a human and mouse genome, but we only focused on the human targets because our fine-tuning data is for the human genome. In the cs282a_test folder you can find a copy of the sequences.bed file that specifies the test/train/validation split of the human genome: there are 23 chromosomes (Y is omitted in many of these sorts of models) which are between 250 million and 50 million base pairs (A, T, C, or G) long. The Basenji2 architecture predicts in 128bp bins with a receptive field of about 20kbp, and predicts on regions of 131kb at a time to yield an output vector of lenght 896. Thus the training set contains 131kb chunks of the genome which are used as inputs to predict genomic state information from 114688bp of that sequence (so that all predictions have roughly full receptive field).

#### Embeddings of Basenji2

Our first step was to pull out embeddings using the code in cs282a_embeddings. This involves downloading the cross2020 model (see manuscripts/cross2020 for the code we used, and https://drive.google.com/drive/folders/1hgjXinKLIWnjFK4c5hvYCq_NOuS0Pnu-?usp=sharing for a copy of the relevant files) and human reference genome hg38 (https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz) then going through the sequences.bed file and using pysam to pull out the relevant sequences and using the basenji dna_io module to one hot encode. We can then run model inference, using seqnn's embedding-extraction functionality to go down one layer from the outputs and grab an 896x1536 embeddings vector for each input and save that to an hdf5 file. The file is about 100GB in size and is stored/downloadable from wget https://cs282-datasets.s3.us-west-1.amazonaws.com/embeddings.h5 (we also describe code later for how to download subsets).

#### Preparing/preprocessing labels

The second step was to prepare labels using 'process_cpg,lmnb1_by_bin.ipynb'. We downloaded the genomic tracks from Shah et al, 2023 (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9869549/, see data availability) and also two low read depth DiMeLo seq runs (https://www.nature.com/articles/s41592-022-01475-6). For dimelo data, .bam files (available here, https://drive.google.com/drive/folders/1ZA8nlNrMW8K0ATsqDp4TM_5mEDTlK9Gs?usp=drive_link) were converted to .bigwig files using 'process_dimelo-to-bigwig.py' run via 'sbatch_gm-to-bigwig.sh' and 'sbatch_hek-to-bigwig.sh' using the arbitrary basemod dev version of the dimelo package, https://github.com/OberonDixon/dimelo/tree/arbitrary_basemod_dev. Then in cs282a_preprocessing there is code to load these datasets in chunks per sequences.bed (including a coordinate transformation using pyLiftOver for DiMeLo seq data, which is aligned to a more complete/newer reference genome for which liftover chain files are available from UCSC), scale them to be approximately 0-500 (same as the preprocessing for Basenji2), create 128bp bins and 114688bp bins, and save to an hdf5 file called dataset_14-lmnb1_4-cpg.h5 (downloadable from https://drive.google.com/drive/folders/1ZA8nlNrMW8K0ATsqDp4TM_5mEDTlK9Gs?usp=drive_link).

#### Training feature extractions

The third step was to train different feature extraction models. There are folders for each of cs282a_linear-probing, cs282a_conv1d_perceptron, cs282a_perceptron-maxpool, and cs282a_self-attention. The linear probing model was trained using code that pulls segments of the embeddings directly from aws, so is most easily reproducible by someone without a bigmem node. However, this same approach is possible (albeit slow) for the other models as well, though the current training code is for training with the full dataset in ~200GB of memory on the Savio high performance cluster. Models are trained with pytorch and saved with the pytorch dict saving method. Test/train/validation split followed the specifications of sequences.bed. Training curves are available in our report.

#### Checking performance

The fourth step was to run inference for the full test/train/validation dataset and save the results to .h5 files. For the three large bin models, these are saved in the cs282a_test folder. For the linear probe, the file is about 2GB and available here under the name probe_first_full_run.h5: https://drive.google.com/drive/folders/1ZA8nlNrMW8K0ATsqDp4TM_5mEDTlK9Gs?usp=drive_link.

cs282a_test contains a small toy example for running inference but one can also re-run large fractions of the genome. On a high-performance computer it should be very quick, bottlenecked by loading embeddings. 

The fifth step was to load up the whole-genome inference files and create scatterplots, genomic track plots, and simple biological analyses. We also calculated head-to-head Pearson and Spearman correlations. It is evident that the three large bin models perform about the same, while the linear probe (which has a harder, noisier task AND fewer parameters to work with) performs worse, but still clearly captures some of the major trends in the test set.

### How to run a test:
To set up your environment, run `pip install -r requirements.txt`. Our models are in pytorch, although if you want to run model inference on basenji with our code you'll need to create a basenji environment as described in the Basenji section. It is within the realm of possibility that we have missed some dependencies here; you should be fine to use pip to install basically anything we use in the code. We endeavor to clean up the dependencies in the final submission. Also note that dependencies for these test files may be different from those for some of the model training, which are again different from those for embeddings extraction, and different again from those for DiMeLo-seq data processing. Each should be done in a different conda environment if you want to run all of those. Other than the basenji package and the dimelo arbitrary_basemod_dev branch, you may also require various biopython installations such as pysam, pyBigWig, pyLiftOver, pybedtools, and so on.

Peer reviewers have two simple tests available, along with all the code for running inference to get embeddings and training feature extraction models: 

1) Run inference with sample_inference.ipynb on each of the four models.

2) Use pre-loaded predictions to replicate biologically relevant genome plots using visualize_tracks.ipynb (this will require downloading some large-ish datasets).

### Explanation of files:
This repo includes files which come from the original Banseji2 repo, and files that we created for our project. We put them all together because depending on how deep you want to dive in, you might need files from the original repo. Our files all start with the prefix “cs282a.”

Cs282a_conv1d_perceptron

Run mlp_extraction_inference.ipynb to obtain predictions from the model.

Cs282a_embeddings

You don’t have to run this.

Basenji_embeddings.ipynb is where we ran inference on the original Basenji2 model with the original input data/labels to obtain the penultimate embeddings

Cs282a_linear-probing

Run linear_probing_inference.ipynb to obtain predictions.

Cs282a_perceptron-maxpool

Run maxpool-perceptron_extraction_inference.ipynb to obtain predictions.

Cs282a_preprocessing

You don’t have to run this. Preprocesses the human epigenetic factor tracks. 

Cs282a_self-attention

Run transformer_inference.ipynb to obtain predictions.


<img src="docs/basset_image.png" width="200">

# Basenji (pre-existed code base)
#### Sequential regulatory activity predictions with deep convolutional neural networks.

Basenji provides researchers with tools to:
1. Train deep convolutional neural networks to predict regulatory activity along very long chromosome-scale DNA sequences
2. Score variants according to their predicted influence on regulatory activity across the sequence and/or for specific genes.
3. Annotate the distal regulatory elements that influence gene activity.
4. Annotate the specific nucleotides that drive regulatory element function.

---------------------------------------------------------------------------------------------------
#### Basset successor

This codebase offers numerous improvements and generalizations to its predecessor [Basset](https://github.com/davek44/Basset), and I'll be using it for all of my ongoing work. Here are the salient changes.

1. Basenji makes predictions in bins across the sequences you provide. You could replicate Basset's peak classification by simply providing smaller sequences and binning the target for the entire sequence.
2. Basenji intends to predict quantitative signal using regression loss functions, rather than binary signal using classification loss functions.
3. Basenji is built on [TensorFlow](https://www.tensorflow.org/), which offers myriad benefits, including distributed computing and a large and adaptive developer community.

However, this codebase is general enough to implement the Basset model, too. I have instructions for how to do that [here](manuscripts/basset).

---------------------------------------------------------------------------------------------------
# Akita
#### 3D genome folding predictions with deep convolutional neural networks.

Akita provides researchers with tools to:
1. Train deep convolutional neural networks to predict 2D contact maps along very long chromosome-scale DNA sequences
2. Score variants according to their predicted influence on contact maps across the sequence and/or for specific genes.
3. Annotate the specific nucleotides that drive genome folding.

---------------------------------------------------------------------------------------------------
# Saluki
#### mRNA half-life predictions with a hybrid convolutional and recurrent deep neural network.

Saluki provides researchers with tools to:
1. Train deep convolutional and recurrent neural networks to predict mRNA half-life from an mRNA sequence annotated with the first frame of each codon and splice site positions.
2. Score variants according to their predicted influence on mRNA half-life, on full-length mRNAs or for a set of pre-defined variants.

A full reproduction of the results presented in the paper, involving variant prediction, motif discovery, and insertional motif anlaysis, can be found [here](https://github.com/vagarwal87/saluki_paper).

---------------------------------------------------------------------------------------------------

### Installation

Basenji/Akita were developed with Python3 and a variety of scientific computing dependencies, which you can see and install via requirements.txt for pip and environment.yml for [Anaconda](https://www.continuum.io/downloads). For each case, we kept TensorFlow separate to allow you to choose the install method that works best for you. The codebase is compatible with the latest TensorFlow 2, but should also work with 1.15.

Run the following to install dependencies and Basenji with Anaconda.
```
    conda env create -f environment.yml
    conda install tensorflow (or tensorflow-gpu)
    python setup.py develop --no-deps
```

Alternatively, if you want to guarantee working versions of each dependency, you can install via a fully pre-specified environment.
```
    conda env create -f prespecified.yml
    conda install tensorflow (or tensorflow-gpu)
    python setup.py develop --no-deps
```

Or the following to install dependencies and Basenji with pip and setuptools.
```
    python setup.py develop
    pip install tensorflow (or tensorflow-gpu)
```

Then we recommend setting the following environmental variables.
```
  export BASENJIDIR=~/code/Basenji
  export PATH=$BASENJIDIR/bin:$PATH
  export PYTHONPATH=$BASENJIDIR/bin:$PYTHONPATH
```

To verify the install, launch python and run
```
    import basenji
```


---------------------------------------------------------------------------------------------------
### Manuscripts

Models and (links to) data studied in various manuscripts are available in the [manuscripts](manuscripts) directory.


---------------------------------------------------------------------------------------------------
### Documentation

At this stage, Basenji is something in between personal research code and accessible software for wide use. The primary challenge is uncertainty in what the best role for this type of toolkit is going to be in functional genomics and statistical genetics. The computational requirements don't make it easy either. Thus, this package is under active development, and I encourage anyone to get in touch to relate your experience and request clarifications or additional features, documentation, or tutorials.

- [Preprocess](docs/preprocess.md)
  - [bam_cov.py](docs/preprocess.md#bam_cov)
  - [basenji_hdf5_single.py](docs/preprocess.md#hdf5_single)
  - [basenji_hdf5_cluster.py](docs/preprocess.md#hdf5_cluster)
  - [basenji_hdf5_genes.py](docs/preprocess.md#hdf5_genes)
- [Train](docs/train.md)
  - [basenji_train.py](docs/train.md#train)
- [Accuracy](docs/accuracy.md)
  - [basenji_test.py](docs/accuracy.md#test)
  - [basenji_test_genes.py](docs/accuracy.md#test_genes)
- [Regulatory element analysis](docs/regulatory.md)
  - [basenji_motifs.py](docs/regulatory.md#motifs)
  - [basenji_sat.py](docs/regulatory.md#sat)
  - [basenji_map.py](docs/regulatory.md#map)
- [Variant analysis](docs/variants.md)
  - [basenji_sad.py](docs/variants.md#sad)
  - [basenji_sed.py](docs/variants.md#sed)
  - [basenji_sat_vcf.py](docs/variants.md#sat_vcf)

---------------------------------------------------------------------------------------------------
### Tutorials

These are a work in progress, so forgive incompleteness for the moment. If there's a task that you're interested in that I haven't included, feel free to post it as an Issue at the top.

- Preprocess
  - [Preprocess new datasets for training.](tutorials/preprocess.ipynb)
- Train/test
  - [Train and test a model.](tutorials/train_test.ipynb)
- Study
  - [Execute an in silico saturated mutagenesis](tutorials/sat_mut.ipynb)
  - [Compute SNP Activity Difference (SAD) and Expression Difference (SED) scores.](tutorials/sad.ipynb)
