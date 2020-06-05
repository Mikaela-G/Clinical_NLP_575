# Clinical_NLP_575 - 2010 i2b2 Concept Extraction Task

## Abstract
This project reproduces Gurulingappa et al's (2010) challenge paper for the 2010 i2b2 Concepts Task. In order to extract medical concepts (problems, treatments, tests) from unstructured discharge summaries and progress notes, we train a CRF on various combinations of features(morphological, context-based, grammatical, and MetaMap-based) extracted from the data using sklearn-crfsuite in Python. Our data was downloaded from the [DMBI Data Portal](https://portal.dbmi.hms.harvard.edu/) and consists of 170 training documents and 256 test documents from Partners Healthcare and Beth Israel Deaconess Medical Center. Our usage of MetaMapLite is courtesy of the U.S. National Library of Medicine.

## Contributors
* Mikaela Guerrero
* Nitya Sampath