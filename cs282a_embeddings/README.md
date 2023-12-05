#### Embeddings of Basenji2

Our first step was to pull out embeddings using the code in cs282a_embeddings. 

The core functionality is contained in basenji_embeddings.ipynb, which saves to an .h5 file named embeddings.h5. This file will be about 100GB.


Running the code involves downloading the cross2020 model (see manuscripts/cross2020 for the code we used, and https://drive.google.com/drive/folders/1hgjXinKLIWnjFK4c5hvYCq_NOuS0Pnu-?usp=sharing for a copy of the relevant files) and human reference genome hg38 (https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz) then going through the sequences.bed file and using pysam to pull out the relevant sequences and using the basenji dna_io module to one hot encode. We can then run model inference, using seqnn's embedding-extraction functionality to go down one layer from the outputs and grab an 896x1536 embeddings vector for each input and save that to an hdf5 file. The file is about 100GB in size and is stored/downloadable from wget https://cs282-datasets.s3.us-west-1.amazonaws.com/embeddings.h5 (we also describe code later for how to download subsets).

The primary requirement to run this code is installing the basenji package. Instructions for installation are provided in the top-level Readme, in the Basenji section. Whereas our feature extraction models use pytorch, basenji is built in tensorflow.

In addition to the core basenji package, containing seqnn and dna_io, please install pandas, scikit-learn, pysam, and tqdm using pip conda. NOTE: this should be done in a different environment than the pytorch downstream analysis.