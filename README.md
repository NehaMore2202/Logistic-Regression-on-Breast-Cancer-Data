# Logistic-Regression-on-Breast-Cancer-Data
Logistic Regression on Breast Cancer Data
### Task1A: 5 points
1. Is there any need to convert columns based on their Dtype? Check details about the data.
2. Check if there are any missing values. Handle the missing values if any.

Output: 
Missing Values:
id                         0
diagnosis                  0
radius_mean                0
texture_mean               0
perimeter_mean             0
area_mean                  0
smoothness_mean            0
compactness_mean           0
concavity_mean             0
concave points_mean        0
symmetry_mean              0
fractal_dimension_mean     0
radius_se                  0
texture_se                 0
perimeter_se               0
area_se                    0
smoothness_se              0
compactness_se             0
concavity_se               0
concave points_se          0
symmetry_se                0
fractal_dimension_se       0
radius_worst               0
texture_worst              0
perimeter_worst            0
area_worst                 0
smoothness_worst           0
compactness_worst          0
concavity_worst            0
concave points_worst       0
symmetry_worst             0
fractal_dimension_worst    0
dtype: int64

### Task 1B: 5 points
First things first!
Before applying feature engineering or bringing the columns to appropriate form, split the data into training/testing.
Why? because by doing this we ensure that there is no data leakage.

1. Map the target variable.
2. Split the data into training/testing with 80-20 ratio.
3. Use stratify since to ensure equal percentage of class samples into both subsamples.

Output: 
Initial dataset size: (569, 32)
No missing 'diagnosis' values to drop.
Dataset size before split: (569, 32)
Training set shape: (455, 30) (455,)
Testing set shape: (114, 30) (114,)

### Task 1C: 10 points
Let's look at mutlicollinearity of the data

1. Plot correlation table for all the number varaibles.
2. Create a list of variables that produces multicollinearity and drop them
3. Plot correlation table for the new set(remaining) of variables.
4. drop the columns for test data as well.

Hint: For detecting multicollinearity variables, look at the diagonal line in the matrix and remove one of the two variables that shows extreme collinearity.

Output:
![image](https://github.com/NehaMore2202/Logistic-Regression-on-Breast-Cancer-Data/assets/154467395/2a614530-d8f9-43a5-ac70-6d9a7c6b2630)


### Task 1D: 5 points
Scale training and testing data using StandardScaler method.

Tip: only transform the testing data.

Output:
     diagnosis  radius_mean  texture_mean  perimeter_mean  area_mean  \
68   -0.768706    -1.440753     -0.435319       -1.362085  -1.139118   
181   1.300887     1.974096      1.733026        2.091672   1.851973   
63   -0.768706    -1.399982     -1.249622       -1.345209  -1.109785   
248  -0.768706    -0.981797      1.416222       -0.982587  -0.866944   
60   -0.768706    -1.117700     -1.010259       -1.125002  -0.965942   

     smoothness_mean  compactness_mean  concavity_mean  concave points_mean  \
68          0.780573          0.718921        2.823135            -0.119150   
181         1.319843          3.426275        2.013112             2.665032   
63         -1.332645         -0.307355       -0.365558            -0.696502   
248         0.059390         -0.596788       -0.820203            -0.845115   
60          1.269511         -0.439002       -0.983341            -0.930600   

     symmetry_mean  ...  radius_worst  texture_worst  perimeter_worst  \
68        1.092662  ...     -1.232861      -0.476309        -1.247920   
181       2.127004  ...      2.173314       1.311279         2.081617   
63        1.930333  ...     -1.295284      -1.040811        -1.245220   
248       0.313264  ...     -0.829197       1.593530        -0.873572   
60        3.394436  ...     -1.085129      -1.334616        -1.117138   

     area_worst  smoothness_worst  compactness_worst  concavity_worst  \
68    -0.973968          0.722894           1.186732         4.672828   
181    2.137405          0.761928           3.265601         1.928621   
63    -0.999715         -1.438693          -0.548564        -0.644911   
248   -0.742947          0.796624          -0.729392        -0.774950   
60    -0.896549         -0.174876          -0.995079        -1.209146   

     concave points_worst  symmetry_worst  fractal_dimension_worst  
68               0.932012        2.097242                 1.886450  
181              2.698947        1.891161                 2.497838  
63              -0.970239        0.597602                 0.057894  
248             -0.809483        0.798928                -0.134497  
60              -1.354582        1.033544                -0.205732  

[5 rows x 31 columns]

### Task 1E: 10 points

Finally!
1. Define Logistic Regression.
2. Use Repeated stratified K Fold method with 5 splits, 3 repeats and roc_auc scoring.
3. Print the mean of roc_auc scores.
4. Fit the Training data

Output:
Mean ROC_AUC Score: 0.9951993832070806
Model fitting complete.

### Task 1F: 10 points

1. Predict y test probability values from model.
2. Plot the precision -recall curve.
2. Obtain the best threshold value using precision_recall curve method and print them along with the f score.
3. Using best threshold, classify the y test probability.
5. Print the Final recall score and print th2 confusion matrix.

Note: y test probablity values are for the event(ie 1, Malignant)

Output:
![image](https://github.com/NehaMore2202/Logistic-Regression-on-Breast-Cancer-Data/assets/154467395/c0990d13-3e85-4818-b12f-915ccc0bc88d)

Best Threshold: 0.7028078329630452, with F-Score: 0.975609756097561
(0.9523809523809523,
 array([[72,  0],
        [ 2, 40]], dtype=int64))

### Task 1G: 10 points

1. Apply SVM model with linear, rbf and poly kernel.
2. Print accuracy and recall score for all kernels.
3. Comment your interpretations for all the models applied on the data.

Output:
{'linear': {'Accuracy': 0.956140350877193, 'Recall': 0.8809523809523809},
 'rbf': {'Accuracy': 0.9035087719298246, 'Recall': 0.7380952380952381},
 'poly': {'Accuracy': 0.8947368421052632, 'Recall': 0.7142857142857143}}
