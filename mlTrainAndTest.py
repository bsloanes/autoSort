import pandas as pd
def featureEngineer(file):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.preprocessing import LabelEncoder
    from sklearn.svm import SVC

    #load training data
    train_data = pd.read_excel(file)

    #"x" and "y" to train
    comments_train = train_data["mlResponses"]
    sorting_data_train = train_data.drop(columns=["mlResponses"])
    
    #labels to numerical format
    #redundant if we're using the TF-IDF
    #label_encoder = LabelEncoder()
    #y_train_encoded = y_train_labels.apply(label_encoder)

    #feature engineering preprocessing TF-IDF
    #BEFORE TF-IDF: ACCOUNT SIMILAR MEANING WORDS (E.G. AMERICA + US)
    #give a quick look 
    #pay special attention to internal lingo in cleaning process
    tfidf_vectorizer = TfidfVectorizer(max_features=5000)
    X_train = tfidf_vectorizer.fit_transform(comments_train).toarray()
    print("TARGET VARIABLE DIMENSIONS: ", sorting_data_train.shape)
    return X_train, sorting_data_train, tfidf_vectorizer

def featureEngineering(train_data):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.preprocessing import LabelEncoder
    from sklearn.svm import SVC

    #load training data
    #train_data = pd.read_excel(file)

    #"x" and "y" to train
    comments_train = train_data["mlResponses"]
    sorting_data_train = train_data.drop(columns=["mlResponses"])
    
    #labels to numerical format
    #redundant if we're using the TF-IDF
    #label_encoder = LabelEncoder()
    #y_train_encoded = y_train_labels.apply(label_encoder)

    #feature engineering preprocessing TF-IDF
    #BEFORE TF-IDF: ACCOUNT SIMILAR MEANING WORDS (E.G. AMERICA + US)
    #give a quick look 
    #pay special attention to internal lingo in cleaning process
    tfidf_vectorizer = TfidfVectorizer(max_features=5000)
    X_train = tfidf_vectorizer.fit_transform(comments_train).toarray()
    print("TARGET VARIABLE DIMENSIONS: ", sorting_data_train.shape)
    return X_train, sorting_data_train, tfidf_vectorizer

def randomForest(X_train, sorting_data_train, tfidf_vectorizer):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import hamming_loss
    from sklearn.metrics import classification_report
    #Random Forest
    classifier = RandomForestClassifier(n_estimators=100, random_state=13)
    classifier.fit(X_train, sorting_data_train)

    #either test then predict or go into predicting right away
    test_data = pd.read_excel("combined_test_sheets.xlsx")
    comments_test = test_data["mlResponses"]
    sorting_data_test = test_data.drop(columns=["mlResponses"])
    X_test = tfidf_vectorizer.transform(comments_test).toarray()
    y_pred = classifier.predict(X_test)
    
    #TRY GRIDSEARCHCV TO FIND OPTIMAL AMOUNT OF N_ESTIMATORS FOR HYPERPARAMETER READJUSTING
    
    #accuracy_score, hamming loss only for single output, not multilabel output
    #Try: average accuracy
    #accuracy = accuracy_score(sorting_data_test, y_pred)
    #print("ACCURACY OF MODEL:", accuracy)
    #hamming_loss_value = hamming_loss(sorting_data_test, y_pred)
    #print("Hamming Loss of Model:", hamming_loss_value)
    #print("LENGTH OF SORTING_DATA_TEST", len(sorting_data_test))
    #print("LENGTH OF Y_PRED",len(y_pred))
    
    label_accuracies = [accuracy_score(sorting_data_test[label], y_pred[:, i]) for i, label in enumerate(sorting_data_test.columns)]
    print("RANDOM FOREST ACCURACY:")
    for label, accuracy in zip(sorting_data_test.columns, label_accuracies):
        print(label, accuracy)
        
    print("RANDOM FOREST CLASSIFICATION REPORTS FOR:")    
    for label in sorting_data_test.columns:
        print("CLASSIFICATION REPORT FOR: ", "Label:", label)
        print(classification_report(sorting_data_test[label], y_pred[:, sorting_data_test.columns.get_loc(label)]))
    #for j in range(len(sorting_data_test)):
    #    print(sorting_data_test.columns[j], label_accuracies[j])

    #average accuracy across all labels
    #show top 5 and bottom 5
    #average_accuracy = sum(label_accuracies) / len(label_accuracies)

    #print("Average Accuracy of Model:", average_accuracy)
    
    #report = classification_report(sorting_data_test, y_pred, target_names=sorting_data_test.columns)
    #LOOK INTO API TO FIND AVERAGE-FRIENDLY METHOD
    #print("ACCURACY REPORT: ", report)

    
    #predict w blank test sheet
    blank_test_data = pd.read_excel("blank_ultimate_test_sheets.xlsx")
    comments_blank_test = blank_test_data["mlResponses"]
    X_blank_test = tfidf_vectorizer.transform(comments_blank_test).toarray()
    predictions = classifier.predict(X_blank_test)

    #this will be the file to compare to the original test doc
    output_df = pd.DataFrame(data=predictions, index=comments_blank_test, columns=sorting_data_train.columns)
    output_df.to_excel("output_from_test_randomforest.xlsx")

def naiveBayes(X_train, sorting_data_train, tfidf_vectorizer):
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.multioutput import MultiOutputClassifier
    
    #Create and train Naive Bayes model
    naive_bayes_classifier = MultinomialNB()
    multioutput_nb_classifier = MultiOutputClassifier(naive_bayes_classifier)
    multioutput_nb_classifier.fit(X_train, sorting_data_train)
    #naive_bayes_classifier.fit(X_train, sorting_data_train)
    
    test_data = pd.read_excel("combined_test_sheets.xlsx")
    comments_test = test_data["mlResponses"]
    sorting_data_test = test_data.drop(columns=["mlResponses"])
    X_test = tfidf_vectorizer.transform(comments_test).toarray()
    y_pred = multioutput_nb_classifier.predict(X_test)
    
    
    label_accuracies = [accuracy_score(sorting_data_test[label], y_pred[:, i]) for i, label in enumerate(sorting_data_test.columns)]
    print("NAIVE BAYES ACCURACY:")
    for label, accuracy in zip(sorting_data_test.columns, label_accuracies):
        print(label, accuracy)
        
    print("RANDOM FOREST CLASSIFICATION REPORTS FOR:")    
    for label in sorting_data_test.columns:
        print("CLASSIFICATION REPORT FOR: ", "Label:", label)
        print(classification_report(sorting_data_test[label], y_pred[:, sorting_data_test.columns.get_loc(label)]))

def svm(X_train, sorting_data_train, tfidf_vectorizer):
    from sklearn.metrics import accuracy_score
    from sklearn.svm import SVC
    from sklearn.multiclass import OneVsRestClassifier
    
    #train svm model
    classifier = OneVsRestClassifier(SVC(kernel='rbf'))  #different kernels include 'linear', 'poly', 'rbf', etc. I picked rbf for accuracy and OneVsRestClassifier so as to account for multiple labels to a question
    
    #1d array for y
    y_train = sorting_data_train.iloc[:,0].values
    
    
    #CHECK AND FIX: the original sorting_data_train seemed to work fine in random forest. May just replace with the same method above in random forest. 
    classifier.fit(X_train, y_train)
    
    # Load the testing data
    test_data = pd.read_excel("testing_data.xlsx")

    #separate the comments (X) from the testing data
    X_test = test_data["mlResponses"]

    #feature engineering: convert text data into TF-IDF representation for testing
    X_test_tfidf = tfidf_vectorizer.transform(X_test).toarray()

    #predict testing data
    y_pred = classifier.predict(X_test_tfidf)

    #output DataFrame with corresponding 0s and 1s
    #FIX
    output_df = pd.DataFrame(data=y_pred, index=X_test, columns=sorting_data_train.columns)
    #output_df = pd.DataFrame(data=y_pred, index=X_test, columns=y_train)


    # Save the output DataFrame to an Excel file
    output_df.to_excel("output_from_test_svm.xlsx")
    
def alternateSVM(X_train, sorting_data_train, tfidf_vectorizer):
    from sklearn.metrics import accuracy_score
    from sklearn.svm import SVC
    from sklearn.multiclass import OneVsRestClassifier
    
    #classifier = OneVsRestClassifier(SVC(kernel='rbf'))
    classifier = SVC(kernel='rbf')
    classifier.fit(X_train, sorting_data_train)
    
    #predict w blank test sheet
    blank_test_data = pd.read_excel("blank_ultimate_test_sheets.xlsx")
    comments_blank_test = blank_test_data["mlResponses"]
    X_blank_test = tfidf_vectorizer.transform(comments_blank_test).toarray()
    predictions = classifier.predict(X_blank_test)

    #this will be the file to compare to the original test doc
    output_df = pd.DataFrame(data=predictions, index=comments_blank_test, columns=sorting_data_train.columns)
    output_df.to_excel("alternate_output_from_svm.xlsx")