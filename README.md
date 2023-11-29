# CS282A Project
#### Oberon Dixon-Luinenburg (BioE PhD student, Ioannidis and Streets labs), Stephanie Brener (BioE PhD student, Little lab), Jimin Jung (CS/DS), Ujjwal Krishnamurthi (CS/DS), Sriteja Vijapurapu (CS)

This project applies four methods of fine tuning on the pre-trained Basenji2 model to predict markers of gene expression from a full human genome sequence. We already conducted all embedding extraction and fine tuning training, and here request our reviewers to replicate validation on test data and illustrate prediction accuracy via the descriptions below. 

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
Linear probe
Convolutional perceptron
Max-pool perceptron
Transformer with self attention

For the linear probe, we adapted the final linear transform layer to do a 1-D CNN on our new labels. This sought to check if the pretrained model with no adaptations could accurately predict our new labels. We then sought to apply feature extraction by training three different smaller networks on top of the pre-trained Basenji model. In the first case, we created a convolutional perceptron and fed the pre-trained embeddings into it with the new labels. In the second case, we created a max-pool perceptron. In the third case, we created a transformer with self attention. Our goal with the first two is that they are a simpler network which should capture information from the new labels and be quicker networks to train. For the transformer, we expect this to have a more successful impact in prediction accuracy because it is a more comprehensive model which will look across the genomic sequence and due to self attention, incorporate more information shared across the genome. 

#### How to run a test:
To set up your environment, run `pip install -r requirements.txt`. Our models are in pytorch, although if you want to run model inference on basenji with our code you'll need to create a basenji environment as described in the Basenji section.

Peer reviewers have two simple tests available, along with all the code for running inference to get embeddings and training feature extraction models: 

1) Run inference with sample_inference.ipynb on each of the four models.

2) Use pre-loaded predictions to replicate biologically relevant genome plots using visualize_tracks.ipynb (this will require downloading some large-ish datasets).

#### Explanation of files:
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
Cs292a_test


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
