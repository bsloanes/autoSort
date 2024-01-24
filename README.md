# autoSort
*UPDATES PENDING* An NLP-based program (in Python) that automates sorting responses to survey questions. Trained or prior survey submissions. Submissions should be in format of excel file. Last sheet of excel file should have raw text-based comments in first column (with a column heading), desired collation labels in the top row. 
1. Download files and save in one folder.
2. Run the ExecutionLabeling.py file, which automates necessary packages installation and full automated sorting training
3. Additionally provides accuracy score and classification report.
4. newExecutionLabeling *in progress* for train-test split by label. Currently splits 80-20 testing-training by volume alone. 
5. dataFunctionsReOrg and mlTrainAndTest contain all of the functions tested out through development, but the ones used are contained in ExecutionLabeling

