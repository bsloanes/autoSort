from dataFunctionsReOrg import *
from mlTrainAndTest import *

def main():
    # Install necessary packages
    loadPackages('packagesSorting.txt')
    # Get a list of all files in the current directory
    file_names = os.listdir()
    #dataframe of all responses and matches from all four survey questions
    combined_df = combine_and_pivot_sheets2("Final")
    #preprocess
    preprocessed_data = load_and_preprocess_data2(combined_df)
    #print(preprocessed_data)
    #split into training and testing sets
    train_sets, test_sets = newestTrainTestSplit(preprocessed_data)
    print(test_sets)
    #train_sets, test_sets = newTestTrain(preprocessed_data)
    
    #preprocessed_sheets = []

    #for file_name in file_names:
    #    if "Final" in file_name:
    #        preprocessed_data = load_and_preprocess_data(file_name)
     #       preprocessed_sheets.append(preprocessed_data)
            #preprocessed_sheets[file_name] = load_and_preprocess_data(file_name) 
    #combined_df = combineSheets(preprocessed_sheets)        
    #train_sets, test_sets = trainTestSplit(combined_df)
    
    toTestBlank0(test_sets)
    
    
    X_train, sorting_data_train, tfidf_vectorizer = featureEngineering(train_sets)
    randomForest(X_train, sorting_data_train, tfidf_vectorizer)
    naiveBayes(X_train, sorting_data_train, tfidf_vectorizer)
    #alternateSVM(X_train, sorting_data_train, tfidf_vectorizer)
    
            
if __name__ == "__main__":
    main()