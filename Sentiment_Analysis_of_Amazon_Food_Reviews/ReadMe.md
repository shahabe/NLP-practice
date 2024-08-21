# Sentiment Analysis with Python

In this practice, we are going to create a sentiment classifier on the amazon fine food reviewes.

## Amazon Fine Food Reviews
https://www.kaggle.com/snap/amazon-fine-food-reviews
This dataset consists of reviews of fine foods from amazon.
The data span a period of more than 10 years, including all ~500,000 reviews up to October 2012.
Reviews include product and user information, ratings, and a plain text review.
It also includes reviews from all other Amazon categories.

Data includes:

- Reviews from Oct 1999 - Oct 2012
- 568,454 reviews
- 256,059 users
- 74,258 products
- 260 users with > 50 reviews
- wordcloud

If you publish articles based on this dataset, please cite the following paper:
J. McAuley and J. Leskovec. From amateurs to connoisseurs: modeling the evolution of user expertise through online reviews. WWW, 2013.

### Prepare the dataset
You need to download the `Reviews.csv` file (300.9 MB) from [here](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews?select=Reviews.csv).
It will give a zip file that you need to extract it to the `./data/` folder, and use the provided code to practice creating the classifier.
The file/folder structure would be like this:  
📦Sentiment_Analysis_of_Amazon_Food_Reviews  
 ┣ 📂data  
 ┃ ┗ 📜Reviews.csv  
 ┣ 📜ReadME.md  
 ┗ 📜analysis.ipynb  

 Now you can follow the `analysis.ipynb` to practice sentiment classification.
