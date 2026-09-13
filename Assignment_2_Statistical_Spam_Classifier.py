#########################################################################################
# Research by Mark Chmielecki for CS6313 - Foundations of Artificial Intelligence        #
# Project 2 due Monday, September 14, 2026.                                             #
#                                                                                       #
# This code is a symbolic rule based SPAM email or text classifer.  The main function   #
# is called first, which calls the symbolic_classifier subroutine, passing in a string  #
# for evaluation.  The string is evaluated on keywords, length, excessive               #
# capitalization, and excessive punctuation, and returns spam or not spam.              #
# Finally, the evaluation is printed by the main function.                              #
#########################################################################################  




from sklearn.feature_extraction.text import CountVectorizer 
#Imports a feather package/class from https://scikit-learn.org/, self described as a way to convert a collection of text documents to a matrix of token counts. 


from sklearn.naive_bayes import MultinomialNB

def statistical_classifier(train_messages, train_labels, test_message):

    vectorizer = CountVectorizer() #instantiating vectorizer as a CountVectorizer object

    X_train = vectorizer.fit_transform(train_messages) #passes in raw documents

    model = MultinomialNB()

    model.fit(X_train, train_labels)

    X_test = vectorizer.transform([test_message])

    prediction = model.predict(X_test)[0]

    return prediction

def main():
    #############################################################################
    # Simple main function, asks for the user to enter a message, then passes   #
    # that message to a classifer sub routine, then prints the answer.          #
    # I've left in some test cases in comments.                                 #
    #############################################################################

    train_messages = [
        "BUY NOW", 
        "Limited time offer", 
        "Exclusive deal", 
        "Can we meet at 3pm tomorrow?",
        "Please send me the updated Powerpoint deck"
    ]
    
    train_lables = [
        "Spam",
        "Spam",
        "Spam",
        "Not Spam",
        "Not Spam"]
    
    #test_message = "BUY NOW!!! Limited time offer $$$"
    test_message = "Catch up with you tomorrow"
    

   
    statistical_classifier_answer = statistical_classifier(train_messages, train_lables, test_message)
    print(statistical_classifier_answer)

if __name__ == "__main__":
    main()