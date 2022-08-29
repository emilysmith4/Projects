# Projects


### 1. **Swoops** <br />
##### _Overview:_ <br />
  Swoops is a NFT basketball game where users can own, trade, and sell their players and compete in cash prize contests. Along with player generation and analysis below, building the basketball game simulator through statistical methods and machine learning algorithms is a key responsibility of mine. This is the link to the free to play game: [Swoops GM](https://gm.playswoops.com) <br />
##### _The Code:_ <br />
  * **'swoops_player_factory.ipynb'** contains the code I have built to soon generate 1,000+ swoops players. The initial phase of generation will initialize the league with veterans, and every new season will create rookies to be entered into the population to be sold. The code involves accessing the NBA API to attach a NBA player's age, positions, height, weight, and rookie season. It uses current NBA player ratings to create distributions through Kernel Density Estimation to sample each attribute rating, and utilizes a Missing Forest algorithm and an eucliden distance method to impute a swoops player's height, weight, age, and how many stars that player is rated. Additionally, given requirements from leadership, the code will generate a specified number of players and produce visualizations to describe what the league population would look like. <br />
  * **'pbp_analysis.ipynb'** contains the analysis I've done on our play by play data as we build the next version of the simulator. Some insights on what the code includes is:
    1. Parse through data to understand the distribution of types of shots
    2. Visualize shot distance and shot accuracy by distance
    3. Visualize event distribution
    4. Create a function that takes an event as the input, and outputs a visualization of the probabilites of various events to follow.
    5. Find out how likely a block will happen on a 2 point shot vs. a 3 point shot
    6. Determine how likely a shooting foul will occur on an shot attempt and shot make (and 1)
    7. Create a function that will analyze end of game boxscore statistics

### 2. **Rubrik Pipeline Order Prediction** <br />
  This code is from an internship I did with a company called Rubrik over summer 2021 and I cleared some outputs to hide sensitive information for the company. <br />
##### _Overview:_ <br />
  Employees, especially the engineers, within the company can place orders for items in order to build and test code. These orders are then sent through an extensive pipeline to build the order to the specified requirements. This process can take time, especially when orders are placed at the same time. To reduce time spent waiting for orders, ideally items can be prepared before the order was even placed. Thus, the main goal of the project was to produce a forecasting algorithm that can predict how many items will be ordered in a certain timeframe, so they can already be ready for when someone orders it. <br />
##### _The Code:_ <br />
  * The goal was to have a specific alorithm for different item types, and possibly different requirements within that item type. **'Pipelines - CDM Clusters.ipynb'** was data exploration to gain understanding on the distribution on different item types and requirements within that item type. The item type 'CDM cluster' was the most common item ordered, so we focused first predicting orders of CDM clusters. <br />
  * **'Order Predictions - Daily.ipynb'** contains data manipulation and then prediction. To simplify our efforts at first, we wanted to focus on CDM cluster orders placed in California on a weekday. This code has predictions by day, however we also did by hour and by 15 minute intervals. Once the data was manipulated and ready for training, we focused on three common forecasting algorithms that gave promising results.
  
### 3. **NBA 2K League** <br />
##### _Overview:_ <br />
  The NBA 2K came to us with a goal to increase global viewership and identify potential locations where they can create a new Esports 2K team. Provided with data by country, we used unsupervised learning to cluster countries in order to gain insights. For instance, if a country was in the same cluster as the United States (a known successful country for the 2K league), that country may share similar characteristics and provide similar, significant interest in NBA 2K. This is a link to the powerpoint that explains the goal, methods, and results: [NBA 2K Powerpoint](https://docs.google.com/presentation/d/1-oNF9Gzr4s-hrpwlVNC6T_1O5SMVAqQDRb2Eddig3PM/edit?usp=sharing) <br /> Also note that this was a group project, however the code that is provided is my work and my responsibility. <br />
##### _The Code:_ <br />
* **'NBA 2K - PCA+Clustering'** is the code used to perform the clustering, both K-means clustering and isolation forest, and feature importance through PCA to understand which features have the biggest impact in determining the clusters.

### 4. **Expected Goal Prediction** <br />
##### _Overview:_ <br />
For a class project, my partner and I were determined to created an accurate and efficient expected goal prediction algorithm. A common metric in soccer is called expected goal (xG) which is essentially the probability that a given shot will be a goal. This metric can give various insights to a team like how well the offense performed and which shots were good shots.
  This is a link to the project report that explains the project goal, methods, and results: [The Report](https://docs.google.com/document/d/1bPAh-uQeEXdd-RPVDspWQmzxO4cd1NX5eImcP377p44/edit?usp=sharing)  <br />
  Please note that this project was completed with a partner, however I was responsible for all the code.
##### _The Code:_ <br />
* **'xG Data Cleaning.ipynb'** contains code used to parse through many json files of events and then concatenate into a single dataframe. We then filtered events to leave only shot data, and also parsed through a dictionary of shot characteristics to create additional columns describing a given shot.
* **'xG Data Manipulation and Predictions.ipynb'** contains additional data manipulation and feature engineering, and then training and bootstrapping of several classification algorithms. The best algorithm that can predict whether a given shot was a goal or not was gradient boosting with an accuracy of about 89%.
