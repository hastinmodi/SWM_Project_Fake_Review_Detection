import streamlit as st
import numpy as np
import pandas as pd
import pickle
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re
from lime.lime_text import LimeTextExplainer
from lime import lime_text
import streamlit.components.v1 as components

nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')

lemmatizer = WordNetLemmatizer()

def lemmatize(text):
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])

def preprocess_text(text): 

    return re.sub(r'\s+', ' ', text)

def infer_result(text, category, rating):
    
    # initialize list of lists
    data = [[text, category, rating]]

    # Create the pandas DataFrame
    df = pd.DataFrame(data, columns=['text_final', 'category', 'rating'])
    
    df['text_final'] = df['text_final'].apply(preprocess_text)
    
    df['text_final'] = df['text_final'].apply(lemmatize)
    
    # create a new column for sentiment based on the rating
    df['sentiment'] = [1 if int(rating) > 3 else 0 for rating in df['rating']]
    
    # create a new column containing the count of words
    df['word_count'] = df['text_final'].str.split().str.len()
    
    # converting the count of words column to a categorical one
    # define the ranges for the categorical values
    bins_ori = [0, 10, 20, 50, 100, 200]

    # define the labels for the categorical values
    labels = [0, 1, 2, 3, 4]

    # convert the numerical column to a categorical column based on the ranges and labels
    df['word_count_categories'] = pd.cut(df['word_count'], bins=bins_ori, labels=labels)
    
    # delete the unnecessary columns
    df = df.drop(['rating', 'word_count'], axis=1)
    
    # Load the trained model using pickle
    with open("lr_model.pkl", "rb") as f:
        lr_model = pickle.load(f)

    with open("text_transformer.pkl", "rb") as f:
        text_transformer = pickle.load(f)

    with open("cat_transformer.pkl", "rb") as f:
        cat_transformer = pickle.load(f)
        
    pred = lr_model.predict_proba(df)
    
    return pred

def assign_cat(category):
    if category == "Kindle_Store_5":
        return 1
    elif category == "Books_5":
        return 2
    elif category == "Pet_Supplies_5":
        return 3
    elif category == "Home_and_Kitchen_5":
        return 4
    elif category == "Electronics_5":
        return 5
    elif category == "Sports_and_Outdoors_5":
        return 6
    elif category == "Tools_and_Home_Improvement_5":
        return 7
    elif category == "Clothing_Shoes_and_Jewelry_5":
        return 8
    elif category == "Toys_and_Games_5":
        return 9
    else:
        return 10

def return_df(text, category, rating):
    
    # initialize list of lists
    data = [[text, category, rating]]

    # Create the pandas DataFrame
    df = pd.DataFrame(data, columns=['text_final', 'category', 'rating'])
    
    df['text_final'] = df['text_final'].apply(preprocess_text)
    
    df['text_final'] = df['text_final'].apply(lemmatize)
    
    # create a new column for sentiment based on the rating
    df['sentiment'] = [1 if int(rating) > 3 else 0 for rating in df['rating']]
    
    # create a new column containing the count of words
    df['word_count'] = df['text_final'].str.split().str.len()
    
    # converting the count of words column to a categorical one
    # define the ranges for the categorical values
    bins_ori = [0, 10, 20, 50, 100, 200]

    # define the labels for the categorical values
    labels = [0, 1, 2, 3, 4]

    # convert the numerical column to a categorical column based on the ranges and labels
    df['word_count_categories'] = pd.cut(df['word_count'], bins=bins_ori, labels=labels)
    
    # delete the unnecessary columns
    df = df.drop(['rating', 'word_count'], axis=1)
    
    return df

def generate_inputs(texts, categorical_features, tokenizer, max_length):
    # Tokenize the input texts
    tokens = tokenizer.batch_encode_plus(
        texts,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='tf'
    )

    # Extract the token ids, attention masks, and token type ids
    input_ids = np.array(tokens['input_ids'])
    attention_masks = np.array(tokens['attention_mask'])
    token_type_ids = np.array(tokens['token_type_ids'])

    # Convert the categorical features to one-hot encoding
    num_classes = [10, 2, 5]  # number of classes for each categorical feature
    categorical_inputs = np.zeros((len(texts), sum(num_classes)))
    for i, num_class in enumerate(num_classes):
        categorical_inputs[np.arange(len(texts)), categorical_features[:, i]-1 + sum(num_classes[:i])] = 1
        
    # Return the inputs as a list of NumPy arrays
    return [input_ids, attention_masks, token_type_ids, categorical_inputs]

def preprocessing(df):
    
    df['category_final'] = df['category'].apply(assign_cat)

    df = df.astype({'sentiment':'int'})
    
    # Separate out text and categorical features for each dataset
    text = df["text_final"].values.astype("str")
    cat_features = df[['category_final', 'sentiment', 'word_count_categories']].values

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    max_length = 128
    
    # Tokenize the texts
    tokens = tokenizer.batch_encode_plus(
        text,
        add_special_tokens = True,
        max_length=max_length,
        padding = True,
        return_attention_mask = True,
        truncation=True,
        return_tensors='tf'
    )
    
    # Generate the input data for train, validation, and test
    inputs = generate_inputs(text, cat_features, tokenizer, max_length)
    
    return inputs

def infer_result_exp(text):
    
    # initialize list of lists
    data = [[text]]

    # Create the pandas DataFrame
    df = pd.DataFrame(data, columns=['text_final'])
    
    df['text_final'] = df['text_final'].apply(preprocess_text)
    
    df['text_final'] = df['text_final'].apply(lemmatize)
        
    # Load the trained model using pickle
    with open("lr_model (1).pkl", "rb") as f:
        lr_model = pickle.load(f)

    with open("text_transformer (1).pkl", "rb") as f:
        text_transformer = pickle.load(f)
        
    pred = lr_model.predict_proba(df)
    
    return pred

def infer_result_lime(texts):
    
    preds = []
    
    for text in texts:
    
        # initialize list of lists
        data = [[text]]

        # Create the pandas DataFrame
        df = pd.DataFrame(data, columns=['text_final'])

        df['text_final'] = df['text_final'].apply(preprocess_text)

        df['text_final'] = df['text_final'].apply(lemmatize)
        
        df['text_final'] = df['text_final'].apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in stopwords.words('english')]))

        # Load the trained model using pickle
        with open("lr_model (1).pkl", "rb") as f:
            lr_model = pickle.load(f)

        with open("text_transformer (1).pkl", "rb") as f:
            text_transformer = pickle.load(f)

        pred = lr_model.predict_proba(df)
        
        preds.append(pred)
    
    res = np.array(preds).reshape(-1,2)
    print(res.shape)
    return res

def infer_result_lime_with_stopwords(texts):
    
    preds = []
    
    for text in texts:
    
        # initialize list of lists
        data = [[text]]

        # Create the pandas DataFrame
        df = pd.DataFrame(data, columns=['text_final'])

        df['text_final'] = df['text_final'].apply(preprocess_text)

        df['text_final'] = df['text_final'].apply(lemmatize)
        
        # Load the trained model using pickle
        with open("lr_model (1).pkl", "rb") as f:
            lr_model = pickle.load(f)

        with open("text_transformer (1).pkl", "rb") as f:
            text_transformer = pickle.load(f)

        pred = lr_model.predict_proba(df)
        
        preds.append(pred)
    
    res = np.array(preds).reshape(-1,2)
    print(res.shape)
    return res

def main():
    st.title("Fake Reviews Detection")
    
    links = st.expander("Related Links")
    if links:
        links.write("[Dataset utilized](https://github.com/hastinmodi/SWM_Project_Fake_Review_Detection/blob/main/Data/Original_GPT_2/fake_reviews_dataset.csv)")
        links.write("[GitHub](https://github.com/hastinmodi/SWM_Project_Fake_Review_Detection)")
        
    # --CHECKBOXES--
    st.subheader("Information on the Classifier")
    if st.checkbox("About Classifer"):
        st.markdown('**Model:** Logistic Regression')
        st.markdown('**Vectorizer:** TfidfVectorizer')

    #--IMPLEMENTATION OF THE CLASSIFIER--
    st.subheader("Fake Review Classifier")
    review = st.text_area("Enter Review: ")
    category = st.selectbox('Category', ('Kindle_Store_5', 'Books_5', 'Pet_Supplies_5', 'Home_and_Kitchen_5', 'Electronics_5', 'Sports_and_Outdoors_5', 'Tools_and_Home_Improvement_5', 'Clothing_Shoes_and_Jewelry_5', 'Toys_and_Games_5', 'Movies_and_TV_5'))
    rating = st.selectbox('Rating', ('1', '2', '3', '4', '5'))

    if st.button("Check"):
        lr_class0 = infer_result(review, category, rating)[0][0]
        lr_class1 = infer_result(review, category, rating)[0][1]
        if lr_class0 > lr_class1:
            st.write('This review is **real** and the model is', round(lr_class0*100, 1),'%', 'confident about it.')
        else:
            st.write('This review is **fake** and the model is', round(lr_class1*100, 1),'%', 'confident about it.')

        explainer = LimeTextExplainer(class_names=[0,1])

        st.write('The explanation of the model output by keeping stopwords.')
        exp_with = explainer.explain_instance(review, infer_result_lime_with_stopwords, num_samples=25)
        html_with = exp_with.as_html()
        components.html(html_with, height=200, scrolling=True)

        st.write('The explanation of the model output by removing stopwords.')
        exp = explainer.explain_instance(review, infer_result_lime, num_samples=25)
        html = exp.as_html()
        components.html(html, height=200, scrolling=True)


#RUN MAIN        
main()

