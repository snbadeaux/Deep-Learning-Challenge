# Deep-Learning-Challenge - Charity Fund Predictor


## Background
The nonprofit foundation Alphabet Soup wants a tool that can help it select the applicants for funding with the best chance of success in their ventures. With your knowledge of machine learning and neural networks, you’ll use the features in the provided dataset to create a binary classifier that can predict whether applicants will be successful if funded by Alphabet Soup.<br>

From Alphabet Soup’s business team, you have received a CSV containing more than 34,000 organizations that have received funding from Alphabet Soup over the years. Within this dataset are a number of columns that capture metadata about each organization, such as:<br>

•	EIN and NAME—Identification columns<br>
•	APPLICATION_TYPE—Alphabet Soup application type<br>
•	AFFILIATION—Affiliated sector of industry<br>
•	CLASSIFICATION—Government organization classification<br>
•	USE_CASE—Use case for funding<br>
•	ORGANIZATION—Organization type<br>
•	STATUS—Active status<br>
•	INCOME_AMT—Income classification<br>
•	SPECIAL_CONSIDERATIONS—Special considerations for application<br>
•	ASK_AMT—Funding amount requested<br>
•	IS_SUCCESSFUL—Was the money used effectively<br>

# Instructions

## Step 1: Preprocess the Data<br>
Using your knowledge of Pandas and scikit-learn’s StandardScaler(), you’ll need to preprocess the dataset. This step prepares you for Step 2, where you'll compile, train, and evaluate the neural network model.<br>

Start by uploading the starter file to Google Colab, then using the information we provided in the Challenge files, follow the instructions to complete the preprocessing steps.<br>

1.	Read in the charity_data.csv to a Pandas DataFrame, and be sure to identify the following in your dataset:<br>
•	What variable(s) are the target(s) for your model?<br>
•	What variable(s) are the feature(s) for your model?<br>
2.	Drop the EIN and NAME columns.
3.	Determine the number of unique values for each column.
4.	For columns that have more than 10 unique values, determine the number of data points for each unique value.
5.	Use the number of data points for each unique value to pick a cutoff point to bin "rare" categorical variables together in a new value, Other, and then check if the binning was successful.
6.	Use pd.get_dummies() to encode categorical variables.
7.	Split the preprocessed data into a features array, X, and a target array, y. Use these arrays and the train_test_split function to split the data into training and testing datasets.
8.	Scale the training and testing features datasets by creating a StandardScaler instance, fitting it to the training data, then using the transform function.<br>

## Step 2: Compile, Train, and Evaluate the Model<br>

Using your knowledge of TensorFlow, you’ll design a neural network, or deep learning model, to create a binary classification model that can predict if an Alphabet Soup-funded organization will be successful based on the features in the dataset. You’ll need to think about how many inputs there are before determining the number of neurons and layers in your model. Once you’ve completed that step, you’ll compile, train, and evaluate your binary classification model to calculate the model’s loss and accuracy.<br>

1.	Continue using the file in Google Colab in which you performed the preprocessing steps from Step 1.
2.	Create a neural network model by assigning the number of input features and nodes for each layer using TensorFlow and Keras.
3.	Create the first hidden layer and choose an appropriate activation function.
4.	If necessary, add a second hidden layer with an appropriate activation function.
5.	Create an output layer with an appropriate activation function.
6.	Check the structure of the model.
7.	Compile and train the model.
8.	Create a callback that saves the model's weights every five epochs.
9.	Evaluate the model using the test data to determine the loss and accuracy.
10.	Save and export your results to an HDF5 file. Name the file AlphabetSoupCharity.h5.<br>

## Step 3: Optimize the Model<br>

Using your knowledge of TensorFlow, optimize your model to achieve a target predictive accuracy higher than 75%.<br>
Use any or all of the following methods to optimize your model:<br>

•	Adjust the input data to ensure that no variables or outliers are causing confusion in the model, such as:<br>
  * Dropping more or fewer columns.<br>
  * Creating more bins for rare occurrences in columns.<br>
  *	Increasing or decreasing the number of values for each bin.<br>
  *	Add more neurons to a hidden layer.<br>
  *	Add more hidden layers.<br>
  *	Use different activation functions for the hidden layers.<br>
  *	Add or reduce the number of epochs to the training regimen.<br>

Note: If you make at least three attempts at optimizing your model, you will not lose points if your model does not achieve target performance.<br>

1.	Create a new Google Colab file and name it AlphabetSoupCharity_Optimization.ipynb.
2.	Import your dependencies and read in the charity_data.csv to a Pandas DataFrame.
3.	Preprocess the dataset as you did in Step 1. Be sure to adjust for any modifications that came out of optimizing the model.
4.	Design a neural network model, and be sure to adjust for modifications that will optimize the model to achieve higher than 75% accuracy.
5.	Save and export your results to an HDF5 file. Name the file AlphabetSoupCharity_Optimization.h5.<br>

# Analysis Report on the Neural Network Model

1. Overview of the analysis: The purpose of the model was to create an algorithm to help predict whether or not an applicant's attempt for charity funding would be successful.
2. Results:<br>
Data Preprocessing
   
 * What variable(s) are considered the target(s) for your model?<br>
   The variable for the Target was identified as the column IS_SUCCESSFUL.
 * What variable(s) are considered to be the features of your model?<br>
   The following columns were considered features of the model:
   * NAME
   * APPLICATION_TYPE
   * AFFILIATION
   * CLASSIFICATION
   * USE_CASE
   * ORGANIZATION
   * STATUS
   * INCOME_AMT
   * SPECIAL_CONSIDERATIONS
   * ASK_AMT
 * What variable(s) should be removed from the input data because they are neither targets nor features?<br>
   The column or variable that can be removed is EIN as it is an identifier for the applicant organization and has no impact on the behavior of the model.

Compiling, Training, and Evaluating the Model

 * How many neurons, layers, and activation functions did you select for your neural network model, and why?<br>
   In the Optimized model, I used 3 hidden layers, as compared to the 2 from the original model. The number of neurons was not changed, although I had to add neurons to the third layer. The activation function was also not changed. Increasing the depth of the model was done since our input data seems to have complex dependencies and hierarchies.
 * Were you able to achieve the target model performance?<br>
   Yes, by optimization, I was able to increase the accuracy of the model from 72% to 79%.
 * What steps did you take in your attempts to increase model performance?<br>
   Ultimately, not eliminating the NAME column and then filtering the column to greater than 1, meaning the applicant had applied for funding more than once, along with increasing the depth of the hidden layers helped to increase the accuracy.<br>
   
   Although not shown, I systematically changed one or two factors at a time and ran the model to see if accuracy increased. I reset factors back to match the original model before again changing new factors. I was hoping to find that only a few small changes could be made in order to increase accuracy. Also, by changing many factors, it would be hard to pinpoint what exactly made the larger difference in accuracy levels.<br>
   
   
4. Summary:<br>
The optimized neural network model achieved 79% prediction accuracy with a 0.47 loss, 3 hidden layers at 10, 8, and 8, and 100 training epochs. Keeping the Name column was crucial in achieving and going beyond the target of 75% accuracy. This shows the importance of the shape of your datasets before you preprocess them.


