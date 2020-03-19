## Overview
Solution to named-entity-recognition problem, using Glove word embeddings with BiLSTM model implemented in pytorch. 
The main file is solution.ipynb file in the root folder. <br>
The html version of this file can be found in the "html" folder<br>
All of the supporting code is extracted to logically separated .py files inside "scripts" folder<br>

## Task in details
1. <b>Implement functionality to read and process NER 2003 English Shared Task data in CoNNL file format, data will be provided (10% of score).</b><br>
Needed functionality can be found in scripts/util.py file
2. <b>Implement 3 strategies for loading the embeddings</b><br>
Needed functionality is located in scripts/embedding_fabric.py
3. <b>Implement training on batches</b><br>
The function for batching is in scripts/utils.py file. The logic for training in batches is implemented in scripts/training_model.py
4. <b>Implement the calculation of token-level Precision / Recall / F1 / F0.5 scores for all classes in average.</b><br>
Implementation is in scripts/metrics.py
5. <b>Provide the report the performances (F1 and F0.5 scores) on the dev / test subsets w.r.t epoch number during the training for the first 5 epochs for each strategy of loading the embeddings</b><br>
The expirement execution and results can be recreated by running solution.ipynb files. 
I have not followed the instructions strictly: for each model and epoch I validated the results on dev set, but the performance for test subset is done only after the training<br>
Sorry about that, I noticed this line too late.
