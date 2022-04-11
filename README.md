# Projects

### 1. **Rubrik Pipeline Order Prediction** <br />
  This code is from an internship I did with a company called Rubrik over this past summer and I cleared some outputs to hide sensitive information for the company. <br />
##### _Overview:_ <br />
  Employees, especially the engineers, within the company can place orders for items in order to build and test code. These orders are then sent through an extensive pipeline to build the order to the specified requirements. This process can take time, especially when orders are placed at the same time. To reduce time spent waiting for orders, ideally items can be prepared before the order was even placed. Thus, the main goal of the project was to produce a forecasting algorithm that can predict how many items will be ordered in a certain timeframe, so they can be prepared for when someone orders it. <br />
##### _The Code:_ <br />
  * The goal was to have a specific alorithm for different item types, and possibly different requirements within that item type. 'Pipelines - CDM Clusters.ipynb' was data exploration to gain understanding on the distribution on different item types and requirements within that item type. The item type 'CDM cluster' was the most common item ordered, so we focused first predicting orders of CDM clusters. <br />
  * Order Predictions - Daily.ipynb' contains data manipulation and then prediction. To simplify our efforts at first, we wanted to focus on CDM cluster orders placed in California on a workday. This code has predictions by day, however we also did by hour and by 15 minute intervals. Once the data was manipulated and ready for training, we focused on three common forecasting algorithms that gave promising results.
