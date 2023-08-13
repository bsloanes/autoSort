import os
import pandas as pd
import subprocess 


def loadPackages(file_name):
    with open(file_name, 'r') as file:
        for package in file:
            package = package.strip()  #get rid of whitespace
            # check if pkg is installed, attempt import.
            if package:
                subprocess.run(['pip', 'install', package])


def preprocess_text(text):
    #ML text cleaning
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    
    
    #DISCONNECT FROM VPN FOR DOWNLOADS
    nltk.download('stopwords')
    nltk.download('wordnet')
    #no capital letters
    text = text.lower()
    #no special characters in comments
    text = ''.join(char for char in text if char.isalnum() or char.isspace())
    #split text
    words = text.split()
    #words that do not contribute to meaning
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    
    
    #bin like-meaning words either here
    compactBins = {
    #"keyword1": "mappedKeyword",
    #add more mappings as needed
}
    for word in words:
        if word in compactBins:
            words[words.index(word)] = compactBins[word]
    #base words
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    ##

    
    #put it all together
    preprocessed_text = ' '.join(words)
    return preprocessed_text
    
def load_and_preprocess_data2(df):
    
    #df = pd.read_excel(originalSurvey, engine="openpyxl") # We can use this if we want all info from multiple sheets in one output sheet. ExcelFile is much better suited for solely getting the survey comments sheet, though.
    
    #only the comments (last) sheet in each excel doc
    #excel_file = pd.ExcelFile(originalSurvey, engine="openpyxl")
    #sheet_names = excel_file.sheet_names
    #last_sheet_name = sheet_names[-1]
    #df = excel_file.parse(last_sheet_name)
    #print("Pre-Processed DF:")
    #print(df.iloc[:, :2])
   
    survey_question = df.columns[0]  #survey question. If it's at a different spot, can also do df.iloc[x, y]
    
    df = df.iloc[1:]  # Skip the first row (header row)
    df.dropna(subset=[survey_question], inplace=True)  # Drop rows with blank comments
    
    
    #df.rename(columns={survey_question: "mlResponses"}, inplace=True)
    #PREPROCESS
    #get rid of any erroneous files and repeated names, even if in an enclosing folder
    df["mlResponses"] = df["mlResponses"].astype(str)
    df["mlResponses"] = df["mlResponses"].apply(preprocess_text)
 

    labelsSorting = [label for label in df.columns]
    df = df[["mlResponses"] + labelsSorting]  # Reorder columns with comments first
    #print("Processed DF:")
    #print(df.iloc[:, :2])
    
    return df
    
def load_and_preprocess_data(originalSurvey):
    
    #df = pd.read_excel(originalSurvey, engine="openpyxl") # We can use this if we want all info from multiple sheets in one output sheet. ExcelFile is much better suited for solely getting the survey comments sheet, though.
    
    #only the comments (last) sheet in each excel doc
    excel_file = pd.ExcelFile(originalSurvey, engine="openpyxl")
    sheet_names = excel_file.sheet_names
    last_sheet_name = sheet_names[-1]
    df = excel_file.parse(last_sheet_name)
    #print("Pre-Processed DF:")
    #print(df.iloc[:, :2])
   
    survey_question = df.columns[0]  #survey question. If it's at a different spot, can also do df.iloc[x, y]
    
    df = df.iloc[1:]  # Skip the first row (header row)
    df.dropna(subset=[survey_question], inplace=True)  # Drop rows with blank comments
    
    
    df.rename(columns={survey_question: "mlResponses"}, inplace=True)
    #PREPROCESS
    #get rid of any erroneous files and repeated names, even if in an enclosing folder
    df["mlResponses"] = df["mlResponses"].astype(str)
    df["mlResponses"] = df["mlResponses"].apply(preprocess_text)
 

    labelsSorting = [label for label in df.columns]
    df = df[["mlResponses"] + labelsSorting]  # Reorder columns with comments first
    #print("Processed DF:")
    #print(df.iloc[:, :2])
    
    return df

#def combineSheets(preprocessed_sheets):
#    combined_df = pd.concat(preprocessed_sheets, ignore_index=True)
#    pivot_table = combined_df.pivot_table(index="mlResponses", aggfunc="first", columns="label_column_name")
#    combined_df = pivot_table.reset_index()
#    combined_df.fillna(0, inplace=True)
#    return combined_df

def combineSheets2(preprocessed_sheets):
    combined_df = pd.concat(preprocessed_sheets, ignore_index=True)
    pivot_table = combined_df.pivot_table(index="mlResponses", aggfunc="first", columns="label_column_name")
    
    # Check for and remove any duplicate index values
    if pivot_table.index.duplicated().any():
        pivot_table = pivot_table.loc[~pivot_table.index.duplicated(keep='first')]
    
    combined_df = pivot_table.reset_index()
    combined_df.fillna(0, inplace=True)
    return combined_df

def combineSheets(preprocessed_sheets):
    # Create an empty DataFrame to store the combined data
    combined_df = pd.DataFrame()

    #loop through files to combine data
    for i in range(len(preprocessed_sheets)):
        #responses
        comments_column_index = 0
        preprocessed_sheets[i] = preprocessed_sheets[i].iloc[:, comments_column_index:]
        
        #combine manual sorting and comments
        #This is where the pandas.errors.InvalidIndexError: Reindexing only valid with uniquely valued Index objects occurs
        
        combined_df = pd.concat([combined_df, preprocessed_sheets[i]], ignore_index=True)
        combined_df.reset_index(inplace=True, drop=True)
    
    combined_df.fillna(0, inplace=True)

    #pivot tables help us to automatically rearrange data and avoid repeat labels
    pivot_df = pd.pivot_table(combined_df, index=['mlResponses'], aggfunc='sum')

    #reconstruct column
    pivot_df.reset_index(inplace=True, drop=True)

    return pivot_df

def newTestTrain(df):

    # Extract the unique labels
    unique_labels = df.iloc[0, 1:].values.tolist()

    # Create dictionaries to hold the training and testing data for each label
    train_data = {label: pd.DataFrame(columns=df.columns) for label in unique_labels}
    test_data = {label: pd.DataFrame(columns=df.columns) for label in unique_labels}

    # Iterate through each label and partition the data
    for label in unique_labels:
        label_df = df[df[label] == 1].copy()  # Filter rows with the specific label

        # Shuffle the rows randomly
        label_df = label_df.sample(frac=1, random_state=42)

        # Calculate the split index
        split_idx = int(0.8 * len(label_df))

        # Split into training and testing sets (80% - 20%)
        train_df = label_df.iloc[:split_idx, :]
        test_df = label_df.iloc[split_idx:, :]

        # Add the split data to the corresponding dictionaries
        train_data[label] = pd.concat([train_data[label], train_df])
        test_data[label] = pd.concat([test_data[label], test_df])

    # Concatenate the training and testing data across all labels
    train_data_combined = pd.concat(train_data.values())
    test_data_combined = pd.concat(test_data.values())

    # Drop any duplicates that might have occurred during the process
    train_data_combined.drop_duplicates(inplace=True)
    test_data_combined.drop_duplicates(inplace=True)

    # Remove the first row, which contains the original label row
    train_data_combined = train_data_combined.iloc[1:, :]
    test_data_combined = test_data_combined.iloc[1:, :]

    # Save the data to new Excel files
    train_data_combined.to_excel('training_data.xlsx', index=False)
    test_data_combined.to_excel('testing_data.xlsx', index=False)
    return train_data_combined, test_data_combined 


def newTTS(df):
    from sklearn.model_selection import train_test_split

    # Load the data from Excel
    #df = pd.read_excel('data.xlsx')

    # Extract the unique labels
    unique_labels = df.iloc[0, 1:].values.tolist()

    # Create dictionaries to hold the training and testing data for each label
    train_data = {label: pd.DataFrame(columns=df.columns) for label in unique_labels}
    test_data = {label: pd.DataFrame(columns=df.columns) for label in unique_labels}

    # Iterate through each label and partition the data
    for label in unique_labels:
        label_df = df[df[label] == 1].copy()  # Filter rows with the specific label

        # Shuffle the rows randomly
        label_df = label_df.sample(frac=1, random_state=42)

        # Split into training and testing sets (80% - 20%)
        train_df, test_df = train_test_split(label_df, test_size=0.2, random_state=42)

        # Add the split data to the corresponding dictionaries
        train_data[label] = pd.concat([train_data[label], train_df])
        test_data[label] = pd.concat([test_data[label], test_df])

    # Concatenate the training and testing data across all labels
    train_data_combined = pd.concat(train_data.values())
    test_data_combined = pd.concat(test_data.values())

    # Drop any duplicates that might have occurred during the process
    train_data_combined.drop_duplicates(inplace=True)
    test_data_combined.drop_duplicates(inplace=True)

    # Remove the first row, which contains the original label row
    train_data_combined = train_data_combined.iloc[1:, :]
    test_data_combined = test_data_combined.iloc[1:, :]

    # Save the data to new Excel files
    train_data_combined.to_excel('training_data.xlsx', index=False)
    test_data_combined.to_excel('testing_data.xlsx', index=False)
    return train_data_combined, test_data_combined

def newestTrainTestSplit(combined_df):
    #iterate through labels
    #check if there is a 1 in each cell for that column
    #if there is, grab the corresponding comment and entire row's contents. 
    #this will effectively create as many mini dataframes as there are comments
    
    #iterate through each of these mini dataframes and split 80-20
    #concatenate the 80s and the 20s together into training and testing, respectively
    #scrub repeats
    #return two dataframes

def newerTrainTestSplit(combined_df):
    from sklearn.model_selection import train_test_split

    #lists to store the training and testing sets for each label
    #or initialize dataframes and .concatenate
    train_sets = []
    test_sets = []

    #iterate over each label (excluding "mlResponses")
    for label in combined_df.columns[1:]:
        X = combined_df[combined_df[label] == 1]["mlResponses"]  
        y = combined_df[combined_df[label] == 1][label]  #does this label relate to this comment?

        # Split the data into 80% train and 20% test for the current label
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=13)

        # Append the train and test sets to the respective lists
        train_sets.append((X_train, y_train))
        test_sets.append((X_test, y_test))

    return train_sets, test_sets


def trainTestSplit2(combined_df):
    from sklearn.model_selection import train_test_split

    #dataframes to store the training and testing sets for each label
    #change back to lists if not working with append or extend
    train_sets = pd.DataFrame()
    test_sets = pd.DataFrame()

    # Iterate over each label (excluding "mlResponses")
    for label in combined_df.columns[1:]:
        X = combined_df["mlResponses"]  # Input features
        y = combined_df[label]  # Target variable

        # Split the data into 80% train and 20% test for the current label
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Append the train and test sets to the respective lists
        train_sets.append((X_train, y_train))
        test_sets.append((X_test, y_test))

    return train_sets, test_sets


def trainTestSplit(combined_df):
    from sklearn.model_selection import train_test_split

    # Lists to store the training and testing sets for each label
    train_sets = []
    test_sets = []

    # Iterate over each label (excluding "mlResponses")
    for label in combined_df.columns[1:]:
        X = combined_df["mlResponses"]  # Input features
        y = combined_df[label]  # Target variable

        # Split the data into 80% train and 20% test for the current label
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Append the train and test sets to the respective lists
        train_sets.append((X_train, y_train))
        test_sets.append((X_test, y_test))

    return train_sets, test_sets

def toTestBlank2(test_sets):
    blank_combined_test_df = pd.DataFrame()
    for label_test_set in test_sets:
        label_name = label_test_set[1].reset_index(drop=True)
        test_set_df = pd.DataFrame({label_name.name: label_name})
        blank_combined_test_df = pd.concat([blank_combined_test_df, test_set_df], axis=1)

    blank_combined_test_df.replace({1: "", 0: ""}, inplace=True)

    #add the "mlResponses" column back
    blank_combined_test_df.insert(0, "mlResponses", test_sets[0][0])

    # Save to Excel
    output_file_name = "blank_ultimate_test_sheets.xlsx"
    blank_combined_test_df.to_excel(output_file_name, index=False)


def trainTestSplit0(combined_df):
    from sklearn.model_selection import train_test_split

    #list to store the training and testing sets for each label
    #USE LISTS IF DICTIONARIES DONT WORK
    #train_sets = []
    #test_sets = []
    train_sets = {}
    test_sets = {}


    #iterate over each label (excluding "mlResponses")
    for label in combined_df.columns[1:]:
        X = combined_df["mlResponses"]  # Input features
        y = combined_df[label]  # Target variable

        #split the data into 80% train and 20% test for the current label
        #check that this is happening along the labels themselves
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        #append the train and test sets to the respective lists IN LISTS
        #INDEX IF DICTIONARY
        #train_sets.append((X_train, y_train))
        #test_sets.append((X_test, y_test))
        train_sets[label] = (X_train, y_train)
        test_sets[label] = (X_test, y_test)
    #print("TRAIN SET: ", train_sets)
    #print("TEST SET: ", test_sets)
    return train_sets, test_sets

def toTestBlank1(test_sets):
    blank_combined_test_df = pd.DataFrame()
    
    #iterate over each label in the test_sets dictionary
    for label, (X_test, y_test) in test_sets.items():
        # Create a DataFrame for the current label's test data
        test_df = pd.DataFrame({"mlResponses": X_test, label: y_test})
        
        # Replace 1s and 0s with blank values in the current label's test data
        test_df[label] = test_df[label].replace({1: "", 0: ""})
        
        # Merge the current label's test data with the blank_combined_test_df
        blank_combined_test_df = pd.concat([blank_combined_test_df, test_df], ignore_index=True)

    # Output to Excel
    output_file_name = "blank_ultimate_test_sheets.xlsx"
    blank_combined_test_df.to_excel(output_file_name, index=False)
    
def toTestBlank(test_sets):
    for i in range(len(test_sets)):
        test_sets_df = pd.DataFrame(test_sets[i])
        blank_combined_test_df = test_sets_df.copy()
        labeled_columns = list(test_sets_df.columns[1:-1])

        #new doc same content w/o manual labeling
        blank_combined_test_df = test_sets_df[["mlResponses"] + list(test_sets_df.columns[1:-1])].replace({1: "", 0: ""})
        #blank_combined_test_df = combined_test_df[["mlResponses"] + list(combined_test_df.columns[1:-1])]
        #blank_combined_test_df[label_columns] = blank_combined_test_df[label_columns].replace({1: "", 0: ""})

    #blank doc to be fed into ML method
    output_file_name = "blank_ultimate_test_sheets.xlsx"
    blank_combined_test_df.to_excel(output_file_name, index=False)

def toTestBlank0(test_sets):
    test_sets_df = pd.DataFrame(test_sets)
    blank_combined_test_df = test_sets_df.copy()
    labeled_columns = list(test_sets_df.columns[1:-1])

    #new doc same content w/o manual labeling
    blank_combined_test_df = test_sets_df[["mlResponses"] + list(test_sets_df.columns[1:-1])].replace({1: "", 0: ""})
    #blank_combined_test_df = combined_test_df[["mlResponses"] + list(combined_test_df.columns[1:-1])]
    #blank_combined_test_df[label_columns] = blank_combined_test_df[label_columns].replace({1: "", 0: ""})

    #blank doc to be fed into ML method
    output_file_name = "blank_ultimate_test_sheets.xlsx"
    blank_combined_test_df.to_excel(output_file_name, index=False)

def trainNtest2(data_df):
    #ML 
    from sklearn.model_selection import train_test_split

    comments = data_df.iloc[:, 0]  #comments
    labels = data_df.columns[1:]  #labels

    #split data into training and test sets (80-20 split)
    X_train, X_test, y_train, y_test = train_test_split(comments, data_df[labels], test_size=0.2, random_state=13)
    
    
    #delete repeat labels.  
    y_train = y_train.drop(columns=[col for col in y_train.columns if 'mlResponses' in col])
    y_test = y_test.drop(columns=[col for col in y_test.columns if 'mlResponses' in col])
    
    
    #concatenate comments and labels
    train_data = pd.concat([X_train, y_train], axis=1)
    test_data_with_labels = pd.concat([X_test, y_test], axis=1)
    return train_data, test_data_with_labels


def trainNtest(data_df, file_name):
    #ML 
    from sklearn.model_selection import train_test_split

    comments = data_df.iloc[:, 0]  #comments
    labels = data_df.columns[1:]  #labels

    #split data into training and test sets (80-20 split)
    X_train, X_test, y_train, y_test = train_test_split(comments, data_df[labels], test_size=0.2, random_state=42)
    
    
    #delete repeat labels. Refer to the survey question relabeling in load_and_process_data 
    y_train = y_train.drop(columns=[col for col in y_train.columns if 'mlResponses' in col])
    y_test = y_test.drop(columns=[col for col in y_test.columns if 'mlResponses' in col])
    
    
    #concatenate comments and labels
    train_data = pd.concat([X_train, y_train], axis=1)
    test_data_with_labels = pd.concat([X_test, y_test], axis=1)
   

    train_data.to_excel(f"train_data_{file_name}", index=False)
    test_data_with_labels.to_excel(f"test_data_with_labels_{file_name}", index=False)
    
def combine_and_pivot_sheets2(tORt):
    files_to_combine = [file for file in os.listdir() if tORt in file]
    #print(files_to_combine)

    #empty DataFrame to store the combined data
    combined_df = pd.DataFrame()

    #loop through files to combine data
    for file_name in files_to_combine:
        #excel_file = pd.read_excel(file_name, engine="openpyxl")
        excel_file = pd.ExcelFile(file_name, engine="openpyxl")
        sheet_names = excel_file.sheet_names
        last_sheet_name = sheet_names[-1]
        data_df = excel_file.parse(last_sheet_name)
        ###
        survey_question = data_df.columns[0]
        data_df.rename(columns={survey_question: "mlResponses"}, inplace=True)
        ###
        #responses
        #comments_column_index = 0
        #data_df = data_df.iloc[:, comments_column_index:]
        #new column for the document name (file name)
        #check if needed
        #data_df['Document'] = file_name
        
        #combine manual sorting and comments
        combined_df = pd.concat([combined_df, data_df], ignore_index=True)
   

    #pivot tables help us to automatically rearrange data and avoid repeat labels
    #check if document column needed
    #pivot_df = pd.pivot_table(combined_df, index=['mlResponses', 'Document'], aggfunc='sum')
    pivot_df = pd.pivot_table(combined_df, index=['mlResponses'], aggfunc='sum')

    #reconstruct column
    pivot_df.reset_index(inplace=True)

    #final test and training sheets
    output_file_name = f"combined_{tORt}_sheets.xlsx"
    pivot_df.to_excel(output_file_name, index=False)
    return pivot_df

def combine_and_pivot_sheets(tORt):
    files_to_combine = [file for file in os.listdir() if tORt in file]
    #print(files_to_combine)

    #empty DataFrame to store the combined data
    combined_df = pd.DataFrame()

    #loop through files to combine data
    for file_name in files_to_combine:
        data_df = pd.read_excel(file_name, engine="openpyxl")
        ###
        #survey_question = data_df.columns[0]
        #data_df.rename(columns={survey_question: "mlResponses"}, inplace=True)
        ###
        #responses
        comments_column_index = 0
        data_df = data_df.iloc[:, comments_column_index:]
        #new column for the document name (file name)
        #check if needed
        #data_df['Document'] = file_name
        
        #combine manual sorting and comments
        combined_df = pd.concat([combined_df, data_df], ignore_index=True)
   

    #pivot tables help us to automatically rearrange data and avoid repeat labels
    #check if document column needed
    #pivot_df = pd.pivot_table(combined_df, index=['mlResponses', 'Document'], aggfunc='sum')
    pivot_df = pd.pivot_table(combined_df, index=['mlResponses'], aggfunc='sum')

    #reconstruct column
    pivot_df.reset_index(inplace=True)

    #final test and training sheets
    output_file_name = f"combined_{tORt}_sheets.xlsx"
    pivot_df.to_excel(output_file_name, index=False)
    return pivot_df

def recombinePivot_and_freqAnalysis(excel1, excel2):
    from collections import Counter
    training_data = pd.read_excel(excel1)
    testing_data = pd.read_excel(excel2)

    # Concatenate the dataframes along the rows
    combined_data = pd.concat([training_data, testing_data], ignore_index=True)
    # Create a DataFrame with the sample data
    df = pd.DataFrame(combined_data, columns=["mlResponses"])

    # Preprocess the text and remove stop words and lemmatize
    df["preprocessed_text"] = df["mlResponses"].apply(preprocess_text)

    # Tokenize the preprocessed text and count word frequencies
    word_counts = Counter()
    for text in df["preprocessed_text"]:
        words = text.split()
        word_counts.update(words)

    # Print the word frequencies
    for word, count in word_counts.items():
        print(f"{word}: {count}")
    
def blank(doc):
    #combined_test_sheets 
    combined_test_df = pd.read_excel(doc)
    blank_combined_test_df = combined_test_df.copy()
    labeled_columns = list(combined_test_df.columns[1:-1])

    #new doc same content w/o manual labeling
    blank_combined_test_df = combined_test_df[["mlResponses"] + list(combined_test_df.columns[1:-1])].replace({1: "", 0: ""})
    #blank_combined_test_df = combined_test_df[["mlResponses"] + list(combined_test_df.columns[1:-1])]
    #blank_combined_test_df[label_columns] = blank_combined_test_df[label_columns].replace({1: "", 0: ""})

    #blank doc to be fed into ML method
    output_file_name = "blank_ultimate_test_sheets.xlsx"
    blank_combined_test_df.to_excel(output_file_name, index=False)





