# Tanzanian Water Wells Status Prediction

**Authors**: Melody Bass

## Overview

Tanzania is a developing country that struggles to get clean water to its population of 59 million people. According to WHO, 1 in 6 people in Tanzania lack access to safe drinking water and 29 million don't have access to improved sanitation. The focus of this project is to build a classification model to predict the functionality of waterpoints in Tanzania given data provided by Taarifa and the Tanzanian Ministry of Water. The model was built from a dataset containing information about the source of water and status of the waterpoint (functional, functional but needs repairs, and non functional) using an iterative approach and can be found [here](./data/training_set_values.csv). The dataset contains 60,000 waterpoints in Tanzania and the following features will be used in our final model:

* amount_tsh - Total static head (amount water available to waterpoint)
* gps_height - Altitude of the well
* installer - Organization that installed the well
* longitude - GPS coordinate
* latitude - GPS coordinate
* basin - Geographic water basin
* region - Geographic location
* population - Population around the well
* recorded_by - Group entering this row of data
* construction_year - Year the waterpoint was constructed
* extraction_type_class - The kind of extraction the waterpoint uses
* management - How the waterpoint is managed
* payment_type - What the water costs
* water_quality - The quality of the water
* quantity - The quantity of water
* source_type - The source of the water
* waterpoint_type - The kind of waterpoint

The first sections focus on investigating, cleaning, wrangling, and reducing dimensionality for modeling. The next section contains 6 different classification models and evaluation of each, ultimately leading to us to select our best model for predicting waterpoint status based on the accuracy of the model. Finally, I will make recommendations to the Tanzanian Government and provide insight on predicting the status of waterpoints.

## Business Problem

Tanzania is a developing country that struggles to provide it's 59 million people with access to clean drinking water.  DrivenData has started a competition to help officials predict statuses of waterpoints by building a model to predict which pumps are functional, functional but need repair, and non functional.  The data is provided by Taarifa and the Tanzanian Ministry of Water with the hope that the information provided by each waterpoint can aid understanding in which waterpoints will fail to improve the government's maintenance operations and ensure that it's residents have access to safe drinking water. 

## Data Understanding

The dataset used for this analysis can be found [here](./data/training_set_values.csv).  It contains a wealth of information about waterpoints in Tanzania and the status of their operation. The target variable has 3 different options for it's status:

* functional - the waterpoint is operational and there are no repairs needed
* functional needs repair - the waterpoint is operational, but needs repairs
* non functional - the waterpoint is not operational

## Modeling

1. Dummy Classifier Model <br />

<img src = "./images/dummy_cm.png" width=40%> <br />
Our baseline dummy model performed very poorly with an accuracy score of 46%. Our data is heavily imbalanced, which explains how our ternary model performed close to 50%.<br />
    
2. Logistic Regression Model <br />

<img src = "./images/log_cm.png" width=40%> <br />
Our logistic regression model is improved to 75% accuracy over the dummy model.  This model struggled to predict wells that were functional but needed repairs, likely due to class imbalances. <br />
    
3. K Nearest Neighbors Model <br />

<img src = "./images/knn_cm.png" width=40%> <br />
The K Nearest Neighbors model outperformed the Logistic Regression model.  Number of neighbors was hypertuned by running and GridSearch and optimal parameters were put into our pipe.  Our K Nearest Neighbors model is not overfitting as the accuracy of training and test sets are 80.23% and 76.03%, respectively.<br />

4. Decision Tree Model <br />

<img src = "./images/dt_cm.png" width=40%> <br />
Our decision tree model once again improved our test accuracy scores to 78%, but the model is highly overfitting with training accuracy at 89%.<br />
    
5. Random Forests Model <br />

<img src = "./images/rf_cm.png" width=40%> <br />
Upon running GridSearch with our Random Forests Pipeline, we improved our baseline accuracy to 81.22% testing accuracy.  The model is still overfitting the training data, as the training accuracy is 93.35%.  The RF model also had the 2nd highest AUC scores at 89.9%. <br />
<img src = "./images/roc_rf.png" width=50%> <br />
    
6. XG Boost Model <br />

<img src = "./images/xgb_cm.png" width=50%> <br />
Our best performing model ended up being the XG Boost model with tuned hyperparameters, although the random forests model was not far behind with 81.22% testing accuracy.  The model has overfitted the training data, but the testing accuracy is overperforming any other model at 81.61%.  The XG Boost model also boasted the highest AUC scores at 90.5%. <br />
<img src = "./images/roc_xgb.png" width=50%> <br />

## Conclusions

XG Boost was our top performing model, although Random Forests was not far behind.  The poor performance of the K Nearest Neighbors, Decision Tree, and Logistic Regression models indicate that the data is not easily separable.  Our XG Boost model performs with an 81.61% testing accuracy.  The XG Boost model also boasted the highest AUC scores at 90.5%.

Based on my findings, I am confident to partner with the Tanzanian government to help solve their water crisis by predicting water pump failure. As we illustrated above, there is a high rate of non functional waterpoints in the southeast corner of Tanzania in Mtwara and Lindi, as well as up north in Mara, and the southwest in Rukwa. These areas need immediate attention as the situations here are critical.   <br />
<img src = "./images/region_function.jpeg" width=70%> <br />
There are a high number of functional wells in Iringa, Shinyanga, Kilimanjaro, and Arusha. There is a cluster of functional but need repair waterpoints in Kigoma, these should be addressed to prevent failure which can be more expensive to repair. <br />
<img src = "./images/map_function.png" width=70%> <br />
Several of our models showed one of it's most important features to be quantity enough for the waterpoint.  There are over 8,000 waterpoints that have enough water in them but are non functional.  These are a high priority to address as well since there is water present. <br />
<img src = "./images/quantity_function.jpeg" width=70%> <br />
Wells with no fees are more likely to be non functional. Payment provides incentive and means to keep wells functional. <br />
<img src = "./images/payment_function.jpeg" width=70%> <br />
The Government, District Council, and Fini Water all have a high rate of pump failure. Investigate why these installers have such a high rate of failure or use other installers. <br />
<img src = "./images/installer_function.jpeg" width=70%> <br />
Future work for this project involve improving the quality of the data moving forward. Better data trained in our model will improve the predictions. We will also monitor the wells and update the model regularly to continuously improve our strategy.

## For More Information

Please review my full analysis in [my Jupyter Notebook](./student.ipynb) or my [presentation](./presentation.pdf).

For any additional questions, please contact **Melody Bass @ meljoy1099@gmail.com**

## Repository Structure

```
├── README.md                           <- The top-level README for reviewers of this project
├── student.ipynb                       <- Narrative documentation of analysis in Jupyter notebook
├── presentation.pdf                    <- PDF version of project presentation
├── data                                <- Both sourced externally and generated from code
└── images                              <- Both sourced externally and generated from code
└── code.                               <- Both sourced externally and generated from code
```
