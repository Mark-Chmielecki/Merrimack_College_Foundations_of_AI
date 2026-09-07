#########################################################################################
# Written by Mark Chmielecki for CS6313 - Foundations of Artificial Intelligence        #
# Project 1 due Monday, September 7, 2026.                                              #
#                                                                                       #
# This code is a symbolic rule based SPAM email or text classifer.  The main function   #
# is called first, which calls the symbolic_classifier subroutine, passing in a string  #
# for evaluation.  The string is evaluated on keywords, length, excessive               #
# capitalization, and excessive punctuation, and returns spam or not spam.              #
# Finally, the evaluation is printed by the main function.                              #
#########################################################################################                                          

def symbolic_classifier(string):
    ##############################################################################################
    # This subroutine takes a string as input representing an email or a text message,           #
    # evaluates the message against a set of rules, and returns either "spam" or "not spam"      #                                                                      
    ##############################################################################################

    evaluation = "not spam" #Sets response to not spam from start, then variable is set to spam if certain conditions exist
    
    #Evaluates the string for keywords, sets everything to lower case to make the match, capitalization will be picked up later in the subroutine 
    keywords = ["BUY NOW","Congratulations!","Past Due Invoice", "Click Here", "Free", "Winner"]
    for word in keywords:
        if word.lower() in string.lower():
            evaluation = "spam"

    #Evaluates the string for message lenth, too short or too long will flag the message as spam 
    if len(string) > 30 or len(string) < 5:
        evaluation = "spam"
    
    #Evaluates excessive capitalization, the tolerance can be adjusted in the future through the capitalization_tolerance variable 
    capitalization_count = 0
    capitalization_tolerance = 2
    for character in string:
        if character.isupper(): 
            capitalization_count = capitalization_count + 1
        if capitalization_count > capitalization_tolerance:
            evaluation = "spam"

    #Evaluates excessive punctuation, the tolerance can be adjusted in the future through the punctuation_tolerance variable 
    punctuation = ["~","!","@","#","$","%","^","&","*","(",")"]
    punctuation_count = 0
    punctuation_tolerance = 2
    for character in string:
        if character in punctuation: 
            punctuation_count = punctuation_count + 1
        if punctuation_count > punctuation_tolerance:
            evaluation = "spam"

    return evaluation

def main():
    #############################################################################
    # Simple main function, asks for the user to enter a message, then passes   #
    # that message to a classifer sub routine, then prints the answer.          #
    # I've left in some test cases in comments.                                 #
    #############################################################################

    #Test Cases
    #string = "BUY NOW!!! Limited time offer $$$"
    #string = "Can we meet at 3pm tomorrow?"

    string = input("Enter a message: ")
    symbolic_classifer_answer = symbolic_classifier(string)
    print(symbolic_classifer_answer)

if __name__ == "__main__":
    main()