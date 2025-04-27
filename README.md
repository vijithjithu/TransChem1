# TransChem: An Ensemble Framework for Polymer Property Prediction Using Transformers and Cheminformatics Descriptors



### Vijith Parambil<sup>1</sup>, Hritik Goyal<sup>1</sup>, Ujjwal Tripati<sup>1</sup>, Rohit Batra<sup>1*</sup>
##### <sup>1</sup> Department of Metallurgical and Materials Engineering, Indian Institute of Technology Madras, Chennai 600036, India


# TransChem

We have developed GPR, Trnspolymer and TransChem files for 10 different polymer datasets. 
The TransChem model has been developed for 10 different datasets. The data files used for each model can be found inside the data folder

Nine out of 10 datasets are curated from the publication "TransPolymer: a Transformer-based language model for polymer property predictions"  by Xu et al. and the Atmoization energy dataset is taken from the Polymer genome. The folder "data" contains all the input data for GPR, TransPolymer and TransChem models. The folders with "_opt" contain the TransChem architecture and corresponding  hyperparameters, which are stored in config_rfe files. The "commonfiles" folder has the libraries for polymer tokenization and data processing.
