# Redundant Question Classification

The goal of this project is to understand what makes two questions semantically the same (according to Quora). The labeled data come from Quora via a Kaggle competition. For Quora, such a classifier would help improve user experience and reduce website maintenance costs.

## Features
Because the problem has been solved best by complex deep learning models, we sought to create a model that uses interpretable features as inputs. Using Spacy's pretrained language model and processing pipeline with fuzzywuzzy match ratios, we engineered 17 features:
<ul>
<li>Question Similarity
<li>Similarity of Different Words
<li>Entity Type Match Ratio 
<li>Entity Match Ratio  
<li>Proper Noun Match Ratio
<li>Noun Match Ratio
<li>Noun Similarity
<li>Similarity of Different Nouns
<li>The previous three points, for verbs, adjectives, and adverbs
</ul>

**Similarity** refers to the cosine similarity of the aggregate word embeddings by document or subdocument<br>
**Entity Type** refers to Spacy's named entity recognition. These are "real world objects with names", ie person, country, place, money, date  <br>
**Entity** refers to the entity instance, ie Theresa May, Great Britain, $12.12, October 1999<br>


## Models
### Logistic Regression
Mean Cross Val Score: 67.17% <br>

### Random Forest 
Mean Cross Val Score: 73.39% <br>
Feature Importance:<br>

### XGBoost 
Mean Cross Val Score: 73.64% <br>
Feature Importance:<br>


