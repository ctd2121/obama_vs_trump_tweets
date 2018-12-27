import pandas as pd
import warnings
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction import text

# Input number of topics to produce for each Twitter account
n_topics = 5

warnings.simplefilter(action='ignore', category=DeprecationWarning) # to remove warnings

# Read in datasets and parse dates
obama = pd.read_csv("./data/obama.csv",
                    parse_dates=["Date"])
trump = pd.read_csv("./data/trump.csv",
                    parse_dates=["created_at"])
trump = trump.drop(['source'], axis=1)

# Maintain list of additional stopwords
additional_stop_words = ["retweet","http","twitter","com","pic","rt",
                         "ofa","president","bo","10","000","www","don",
                         "wh","00","et","11","ed","op","ve","https",
                         "amp","realdonaldtrump","ly"]

def generate_topics(president_df, n_topics):
    """
    Computes the most frequently tweeted-about topics in a given collection of tweets
    
    Args:
        president_df: A Pandas DataFrame, the first column of which consists of tweets
        n_topics: The number of topics that should be printed out
    """
    # Drop NAs
    president_df = president_df.dropna(axis=0, how="any")
    
    # Create corpus of tweets by appending values from first column of df
    tweets = []
    for i in range(len(president_df)):
        tweets.append(president_df.iloc[i,0])
        
    # Instantiate CountVectorizer, a scikit-learn object that defines how we will create our vocabulary
    # All lowercase, remove English stop words and additional stop words, and create 1- and 2-word n-grams
    cv = text.CountVectorizer(lowercase=True,
                              stop_words=text.ENGLISH_STOP_WORDS.union(additional_stop_words),
                              ngram_range=(1,2))
    
    # Create term-document matrix
    # Rows represent tweets, columns represent words in vocabulary
    tfidf = cv.fit_transform(tweets)
    # Ensure number of rows in matrix equals the number of tweets in original df
    assert tfidf.shape[0] == len(president_df)
    
    # Create LDA model
    lda = LatentDirichletAllocation(n_components=n_topics)
    
    # Run LDA on the term-frequency vectorizer object
    # Note: this takes around five minutes, depending on the machine
    X_lda = lda.fit_transform(tfidf)
    
    # Get topics and print them to screen
    features = cv.get_feature_names()
    print_top_words(lda, features, 8) # Print the eight top words that make up each topic (can change this number if desired)

# An auxiliary function to print out the most likely terms for each topic
# Taken from https://scikit-learn.org/stable/auto_examples/applications/plot_topics_extraction_with_nmf_lda.html
def print_top_words(model, feature_names, n_top_words):
    """
    Prints out a fixed number of topics
    
    Args:
        model: A Latent Dirichlet Allocation model
        feature_names: The topics/features to be printed, as computed by the LDA model
        n_top_words: The number of words to be printed for each topic
    """
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic {:#2d}: ".format(topic_idx+1)
        message += ", ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)

if __name__ == "__main__":
    print("\nTop " + str(n_topics) + " topics for Barack Obama...\n")
    generate_topics(obama, n_topics)
    print("\nTop " + str(n_topics) + " topics for Donald Trump...\n")
    generate_topics(trump, n_topics)