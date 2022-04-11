# Projects

### 1. **Rubrik Pipeline Order Prediction** <br />
  This code is from an internship I did with a company called Rubrik over this past summer and I cleared some outputs to hide sensitive information for the company. <br />
##### _Overview:_ <br />
  Employees, especially the engineers, within the company can place orders for items in order to build and test code. These orders are then sent through an extensive pipeline to build the order to the specified requirements. This process can take time, especially when orders are placed at the same time. To reduce time spent waiting for orders, ideally items can be prepared before the order was even placed. Thus, the main goal of the project was to produce a forecasting algorithm that can predict how many items will be ordered in a certain timeframe, so they can already be ready for when someone orders it. <br />
##### _The Code:_ <br />
  * The goal was to have a specific alorithm for different item types, and possibly different requirements within that item type. **'Pipelines - CDM Clusters.ipynb'** was data exploration to gain understanding on the distribution on different item types and requirements within that item type. The item type 'CDM cluster' was the most common item ordered, so we focused first predicting orders of CDM clusters. <br />
  * **'Order Predictions - Daily.ipynb'** contains data manipulation and then prediction. To simplify our efforts at first, we wanted to focus on CDM cluster orders placed in California on a weekday. This code has predictions by day, however we also did by hour and by 15 minute intervals. Once the data was manipulated and ready for training, we focused on three common forecasting algorithms that gave promising results.

### 2. **Expected Goal Prediction** <br />
##### _Overview:_ <br />
For a class project, my partner and I were determined to created an accurate and efficient expected goal prediction algorithm. In soccer, many shots happen yet few result in goals. A common metric in soccer is called expected goal (xG)
  This is a link to the project report that explains the project goal, methods, and results: [The Report](https://docs.google.com/document/d/1bPAh-uQeEXdd-RPVDspWQmzxO4cd1NX5eImcP377p44/edit?usp=sharing)  <br />
  Please note that this project was completed with a partner, however I was responsible for all the code.
##### _The Code:_ <br />
* **'xG Data Cleaning.ipynb'** contains code used to parse through many json files of events and then concatenate into a single dataframe. We then filtered events to leave only shot data, and also parsed through a dictionary of shot characteristics to create additional columns describing a given shot.
* **'xG Data Manipulation and Predictions.ipynb'** contains additional data manipulation and feature engineering, and then training and bootstrapping several classification algorithms. The best algorithm that can predict whether a given shot was a goal or not was gradient boosting with an accuracy of about 89%.

### 3. **NBA 2K League** <br />
##### _Overview:_ <br />
  The NBA 2K came to us with a goal to increase global viewership and identify potential locations where they can create a new Esports 2K team. Provided with data by country, we used unsupervised learning to cluster countries in order to gain insights. For instance, if a country was in the same cluster as the United States (a known successful country for the 2K league), that country may share similar characteristics and provide similar, significant interest in NBA 2K. This is a link to the powerpoint that explains the goal, methods, and results: [NBA 2K Powerpoint](https://docs.google.com/presentation/d/1-oNF9Gzr4s-hrpwlVNC6T_1O5SMVAqQDRb2Eddig3PM/edit?usp=sharing) <br /> Also note that this was a group project, however the code that is provided is my work and my responsibility. <br />
##### _The Code:_ <br />
* **'NBA 2K - PCA+Clustering'** is the code used to perform the clustering, both K-means clustering and isolation forest, and feature importance through PCA to understand which features have the biggest impact in determining the clusters.
 
### 4. **Breast Cancer Prediction** <br />
##### _Overview:_ <br />
  For a class project, my group decided on the task to predict whether a breast cancer tumor was cancerous or benign based on many characteristics of a tumor such as radius, texture, and smoothness. This is a link to the project slides that goal, methods, and results: [Breast Cancer Slides](https://docs.google.com/presentation/d/1no9PbZZ4Ck9jWfjIcfAOuGsYN3_ZSh_imzwxN0PGua4/edit?usp=sharing) <br /> This was also a group project, and in the code below I was responsible for the machine learning and model training portion.
##### _The Code:_ <br />
* **'Breast Cancer Prediction.ipynb'** is the group code for the project. While I soley did the model training (logistic regression, KNN, decision tree, random forest, neural network, SVM, AdaBoost), I also helped in the portion above conducting data exploration and visualization.
