# PRODIGY_ML_02
Customer Segmentation using K-Means Clustering
This project implements K-Means Clustering, an unsupervised machine learning algorithm, to group retail store customers based on their purchase behavior.
The goal is to help businesses understand customer segments for targeted marketing and decision-making.

Problem Statement
To group customers of a retail store into different segments based on:
->Annual Income
->Spending Score
using the K-Means clustering algorithm.

Dataset Information
Dataset Type: Retail / Mall Customers Dataset
File Name: Mall_Customers.csv

Dataset Columns Used
 Column Name             Description                             
 ----------------------  --------------------------------------- 
 Annual Income (k$)      Customer’s annual income                
 Spending Score (1-100)  Spending behavior score                 
 Cluster                 Cluster assigned by K-Means (generated) 

Technologies & Tools Used
  ->Python 3.x
  ->Pandas
  ->Matplotlib
  ->Seaborn
  ->Scikit-learn
  ->Visual Studio Code
Installation & Setup
1. Clone the Repository
   git clone https://github.com/your-username/customer-segmentation-kmeans.git
cd customer-segmentation-kmeans
2. Install Required Python Libraries
   python -m pip install pandas matplotlib seaborn scikit-learn
3. Dataset Setup
  Place the dataset file Mall_Customers.csv in the project root directory.
How to Run the Project
  python customer_segmentation.py

Project Workflow
  ->Load customer dataset using Pandas
  ->Select relevant features for clustering
  ->Apply feature scaling using StandardScaler
  ->Determine optimal number of clusters using the Elbow Method
  ->Apply K-Means clustering
  ->Visualize customer segments
  ->Assign cluster labels to customers
  
Model & Visualization
  ->Elbow Method Graph → Determines optimal number of clusters
  ->Scatter Plot → Visualizes customer groups based on income and spending score
