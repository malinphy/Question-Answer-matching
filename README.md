#  Question Answer Matching with Universal Sentence Encoders(USE)

Overview :
Tensorflow/Keras implementation of Question answer matching system. 
In short, this project is designed for finding the best possible 
answers for related questions. 


Universal Sentence Encoder (USE) is the back bone of this system.
USE uses the pretrained BERT to generate sentence embeddings. In 
dataset, each question is paired with a correspondings answer.
For each answer one random negative sample created. Questions, 
answers and negative sentences were treated as anchor, positive 
and negative samples respectively. VEctor values of questions,
positive and negative samples were extracted using USE. 
Employing the triplet loss, distance between anchor
and corresponding positive vectors is minimized,
while increase the distance between anchor and negative vectors.
In this way, pre-trained USE has been fine tuned for question-
mathcing system. 

In question matching models, question-answer dataset is composed of 
paraphrased version of same question or question form of 
corresponding answer to reduce number new sentences for prediction
process. However, fine tuned USE is capable of generate embeddings 
which already capture meaning and context. The fine-tuned USE can 
vectorize the user-given question, placing it close to the same-
meaning question in the trained dataset.

For simplicity, every vectorized answer can be stored in a vector
database such as ScaNN. During the prediction, user given question
is vectorized using fine tuned USe and closest vector can be 
extracted from vector database and returned to to user as answer.

Data :<br/>
----

ELI 5 (from reddit forum explain like I am 5) Dataset : 
https://facebookresearch.github.io/ELI5/download.html
<br/>
for simplicity dataset hugginface datasets :
https://huggingface.co/datasets/eli5

File Description :
----
- data_loader.py : loads the dataset and splits questions and answers
- HelperFunctions.py : Data preparation for model training
- model.py : nueralnetwork model written with tensoflow/keras
- negative_maker.py : generation of random negative samples
- train.py : training file
- prediction.py : prediction file for deployment purpose
- sent_bert_mnr_cli5_v2.ipynb : notebook that is suitable for colab.





EVALUATION
----------
```
              |@1    |@3    |@5    |@10   |@20   |
 -------------|------|------|------|------|------|
 precision@K  |0.2376|0.1210|0.0924|0.0557|0.0321|
 -------------|------|------|------|------|------|
 recall@K     |0.2376|0.3899|0.4618|0.5572|0.6421|   
```
EXAMPLE
----------

**TEST QUESTION** : What's the difference between a bush, a shrub, and a tree?

**TEST ANSWER** : Shrubs and trees are both specifically *woody* plants with stems that survive throughput the winter. A tree has a clear central trunk whereas a shrub has multiple stems rising from the ground.'Bush' is a more general term for any plant with multiple stems rising from the ground, and that can be either woody or what's called herbaceous, herbaceous plants are ones where the stems die back completely or substantially in the winter leaving the plant with just its roots and new stems grow next spring.
<br />
<br />
**ALTERNATIVE QUESTION** : Is there any difference between a bush, a shrub, and a tree?

**ANSWER FOR ALTERNATIVE QUESTION** : Shrubs and trees are both specifically *woody* plants with stems that survive throughput the winter. A tree has a clear central trunk whereas a shrub has multiple stems rising from the ground.
'Bush' is a more general term for any plant with multiple stems rising from the ground, and that can be either woody or what's called herbaceous, herbaceous plants are ones where the stems die back completely or substantially in the winter leaving the plant with just its roots and new stems grow next spring.


**ALTERNATIVE QUESTION** : the difference between a bush, a shrub, and a tree?

**ANSWER FOR ALTERNATIVE QUESTION** : Shrubs and trees are both specifically *woody* plants with stems that survive throughput the winter. A tree has a clear central trunk whereas a shrub has multiple stems rising from the ground.
'Bush' is a more general term for any plant with multiple stems rising from the ground, and that can be either woody or what's called herbaceous, herbaceous plants are ones where the stems die back completely or substantially in the winter leaving the plant with just its roots and new stems grow next spring.


## Note: Vector Similarity Search (ScaNN) has suitable distribution for only linux. However, colab suitable notebook 'sent_bert_mnr_cli5_v2.ipynb' capable of running without a problem.
