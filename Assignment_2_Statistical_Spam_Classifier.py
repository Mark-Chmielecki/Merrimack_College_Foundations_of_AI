#########################################################################################
# Research by Mark Chmielecki for CS6313 - Foundations of Artificial Intelligence       #
# Project 2 due Monday, September 14, 2026.                                             #
#                                                                                       #
# Questions from assignment:                                                            #
# 1) Why is this more scalable?  I can pass in a lot of messages to train the model, and#
#     the model infer whether or not the test message is spam, as opposed to me         #
#     trying to figure out every case, and writing a rule for that case.                #
# 2) Explain what each library is used for - see in line comments below                 #
# 3) What is each line and variable doing? see in line comments below                   #
# 4) Compare this approach to Spam Classifer approach in week 1:                        #
#     -This approach is using a statistical AI method, as opposed to a a symbolic       #
#      model.                                                                           #
#     -Instead of passing a message through clearly defined rules, it uses a statistical#
#      model, where based on training, it determines the probability of whether or not  #
#      the message is Spam.                                                             #  
#########################################################################################  


from sklearn.feature_extraction.text import CountVectorizer 
# Imports a feature package/class from https://scikit-learn.org/, self described as a way to convert a collection of text documents to a matrix of token counts.
# In this simple example, we are passing in a list of messages, not text documents.  Counts the number of times words show up in a document or message. 

from sklearn.naive_bayes import MultinomialNB
# Imports the Multinomial Naive Bayes classifier from https://scikit-learn.org/
# Leverages the Bayes Theorem to calculate the probablility of something given a set of training data and training lables, in this case whether or not the message is spam.
# Naive means each word is independent of every other word, instead of taking into account a phrase.

def statistical_classifier(train_messages, train_labels, test_message):

    vectorizer = CountVectorizer() # Instantiating vectorizer as a CountVectorizer object

    X_train = vectorizer.fit_transform(train_messages) # transforms raw training messages into a numerical format the ML can understan, in this case MultinomialNB  

    model = MultinomialNB() # Instantiating the Multinomial Naive Bayes model as the variable "model"  

    model.fit(X_train, train_labels) # Trains the machine learning "model" using the already esablished training messages, with the training messages lables

    X_test = vectorizer.transform([test_message]) # Converts test message passed in from main() into the same format as the training data 

    prediction = model.predict(X_test)[0] # Predict() runs the model in inference mode, passing the input date through the forward pass to return the model's prediction
    # adding the zero at the end simply returns the first value in the list

    return prediction # returns answer

def main():
    ########################################################################################################
    # Simple main function.  Passes a list of training messages and lables, and prints the model's answer. #
    # I've left in some test cases in comments.                                                            #
    ########################################################################################################

    train_messages = [
        "BUY NOW", 
        "Limited time offer", 
        "Exclusive deal", 
        "Can we meet at 3pm tomorrow?",
        "Please send me the updated Powerpoint deck"
    ]
    
    train_labels = [
        "Spam",
        "Spam",
        "Spam",
        "Not Spam",
        "Not Spam"]
    
    #test_message = "BUY NOW!!! Limited time offer $$$"
    #test_message = "Catch up with you tomorrow"

    test_message = input("Enter a message: ")   
    statistical_classifier_answer = statistical_classifier(train_messages, train_labels, test_message)
    print(statistical_classifier_answer)

if __name__ == "__main__":
    main()