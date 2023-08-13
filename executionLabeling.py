from dataFunctionsReOrg import *
from mlTrainAndTest import *

def main():
    # Install necessary packages
    loadPackages('packagesSorting.txt')
    # Get a list of all files in the current directory
    file_names = os.listdir()
    print(file_names)

    # Loop through the files to process only those with "Final" in the name
    # Close/quit Excel. Otherwise, this will not run.
    # Customer should not have any excel documents in their working directory with "Final" in the title other than training/test 
    # Assuming the last or only sheet in each file is desired one to process
    for file_name in file_names:
        if "Final" in file_name:
            data_df = load_and_preprocess_data(file_name)           
            trainNtest(data_df, file_name)        
        
    combine_and_pivot_sheets("test")
    combine_and_pivot_sheets("train")
    #splitByLabel("combined_train_sheets.xlsx","combined_test_sheets.xlsx")
    blank("combined_test_sheets.xlsx")
    X_train, sorting_data_train, tfidf_vectorizer = featureEngineer("combined_train_sheets.xlsx")
    randomForest(X_train, sorting_data_train, tfidf_vectorizer)
    naiveBayes(X_train, sorting_data_train, tfidf_vectorizer)
    #alternateSVM(X_train, sorting_data_train, tfidf_vectorizer)
    
            
if __name__ == "__main__":
    main()