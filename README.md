# SWM_Project_Fake_Review_Detection

The following group members were involved in creating this project:
* Hastin Himanshubhai Modi
* Visaj Nirav Shah
* Mohil Devan Khimani
* Shivam Milind Bodiwala
* Shivani Shailesh Nandani
* Rahil Ashish Shah

**Objective:**
The main aim of this project is to create machine learning algorithms which can detect fake reviews with a high accuracy.

**Data:**
The [data](https://github.com/hastinmodi/SWM_Project_Fake_Review_Detection/blob/main/Data/Original_GPT_2/fake_reviews_dataset.csv) used contains real reviews from Amazon for 10 product categories. There are a total of 40K reviews with 20K of them being real and 20K being fake which are generated using GPT-2.

A [new dataset](https://github.com/hastinmodi/SWM_Project_Fake_Review_Detection/blob/main/Data/New_Back_Translation/new_fake_reviews_data.csv) was also created to ensure that our results are robust enough to give accurate results with high confidence. In this dataset, the fake reviews are generated using back translation, where we are converting the real reviews from Amazon in English to French and then back again to English to generate fake reviews.

**Models:**
The following machine learning [models](https://github.com/hastinmodi/SWM_Project_Fake_Review_Detection/tree/main/Code) were evaluated on both the datasets and the results are shown in the Evaluation folder.

Traditional models:
* Logistic Regression
* K-nearest neighbors
* Support Vector Classifier
* Naive Bayes
* Decision Tree
* Random Forests
* AdaBoost
* XGBoost

Advanced models:
* BERT
* CNN
* CNN + Attention
* BiLSTM

A demo website using Streamlit has also been [created](https://github.com/hastinmodi/SWM_Project_Fake_Review_Detection/blob/main/Code/streamlit_ui.py) and the required libraries for running it can be installed from the [requirements.txt](https://github.com/hastinmodi/SWM_Project_Fake_Review_Detection/blob/main/Code/requirements.txt) file.

