# Predicting-US-stock-price-variations-based-on-financial-statements-

![](RackMultipart20200507-4-1dbjc1r_html_dc2d0a10b0a55359.png)

# **Financial Indicators of US stocks (2018)**

PROJECT DELIVERABLE II

FRAMEWORKS AND METHODS II

Luis Echeverri

Kerim Birgun

Srikrishna Murali

Karin Osorio

1.
## Statement of the problem

Stock prices are influenced by several factors, which can be divided into two categories: macroeconomic and microeconomic variables. This project will focus on microeconomic factors, especially profitability and performance ratios, that stem financial statements. Since financial ratios show the performance of companies, these indicators are very helpful to guide investment decisions. In this project, we will try to predict stock price variations (Year over Year) using information derived from annual financial statements released by companies.

With this work, we would like to probe whether a 10k report provides enough information to predict stock price variation accurately. In this sense, the questions that we want to solve using data and analytical techniques are:

- Based on 10k reports&#39; information, can we accurate predict stock prices YoY variation?
- Do dimension reduction and clustering techniques really help to enhance the results of ML algorithms such as random forest and GB?
- What variables, included or derived from annual financial statements or 10k reports, drive stock price variation?

1.
## The data

The dataset contains 222 financial indicators reported by more than 4,000 U.S publicly listed companies in their 2018-10K reports. The 10K report is an annual report required by the U.S securities and exchange commission, that gives a comprehensive summary of a company&#39;s financial performance. Additionally, the third-to-last column, Sector, lists the sector of each company as in the U.S stock market each company is part of a sector that classifies in one of eleven macro sectors (basic materials, communications services, consumer cyclical, consumer defensive, energy, financial services, healthcare, industrials, real state, technology, and utilities).

![](RackMultipart20200507-4-1dbjc1r_html_ca8739edc63dcd8.jpg)

The second-to-last column lists the percent variation of each stock for the year. The percent variation was estimated considering the year&#39;s first and last trading day. The last column, class, determines whether a stock is buy-worthy or not buy worthy. Therefore, these two last variables allow us to run whether classification or regression tasks to define if a stock should be subject or not of trading decisions.

1.
## The process

To predict the stock price variation, we performed the following steps:

  1.
## Data exploration and cleaning

- Explore the response variable: price variation {distribution and outliers}

![](RackMultipart20200507-4-1dbjc1r_html_4e042cd8ab346a4f.jpg) ![](RackMultipart20200507-4-1dbjc1r_html_9f073fcaf6119663.jpg)

- Distributions, ranges and quality of other predictors

![](RackMultipart20200507-4-1dbjc1r_html_ca3224aac4a46718.jpg)

- Imbalances across the dataset (class variable – further data split)
- Predictors with Nas\&gt;15% | Drop predictors with Nas \&gt; threshold
- Imputation of missing values

 ![](RackMultipart20200507-4-1dbjc1r_html_c3ebbb07dd455600.gif)

Several regressions were ran using a wide range of models seeking to find the best approach to reduce the RMSE and higher R2. As part of the process, we tried to cluster the stocks to predict their performance, but results didn&#39;t outperform previous approaches.

  1.
## Feature selection

Part of this work aims to understand what variables (stated or estimated) from 10k filings drive stock price variation. Therefore, techniques such as PCA cannot be used, given that we would lose model interpretability. Therefore, due to the high dimensionality (initially, there were 225 predictors in the dataset), this work explored two feature selection approaches:

- Analytical methods (variable importance):
  - Lasso regression – features based on coefficients
  - Random Forest – features importance

![](RackMultipart20200507-4-1dbjc1r_html_a9609a1bb2c97f8d.png)

- Literature review (previous studies – variables selections [ratios])

We reviewed different papers that tried to prove if it exists a relationship between different financial indicators and the stock price of a company. According to the study of Drummen and Zimmermann (1992) the individual characteristics of companies affect up to 50 % of stock prices. The results suggest that there is significant relationship between accounting earnings and stock prices. The results show that the ratios of working capital to total assets and net profit to sales have a negative impact on stock returns, while the ratios of net profit to total assets and sales to total assets positively affect returns.

In the paper called &quot;Relation between share price and financial indicators in the brazilian stock market&quot; – 2016, the authors explore the relationship between the share prices of companies listed on the BM&amp;FBovespa with some financial indicators, including earnings per share, book value per share and total assets. This paper states that increases of company profit or assets culminate in a higher price of its shares.

Another work from the Australian stock market, Brimble and Hodgson (2007), proved the relationship between book value and profit, and book value and dividends.

  1.
## Feature engineering

- Scaling
- Normalization -\&gt; bestNormalize()

![](RackMultipart20200507-4-1dbjc1r_html_1912148c21742315.png)

We observed the distribution of our dependent variable and its subseqent residual plot based on linear regression.

Apparently there was a skewness in the distribution and we observed high variance in the residual plot Accordingly, we utilized bestnormalization package in R to choose the best normalization function and then we applied it.

As a result of this process, residual variances are observed to be converged and besides, the R squared of the model increased from .11 to .15

1.
## Analytical techniques

Several regressions were ran using a wide range of models seeking to find the best approach to reduce the RMSE and higher R2. As part of the process, we tried to cluster the stocks to predict their performance, but results didn&#39;t outperform previous approaches. The models run are the following:

- Linear Regression
- Random Forest
- Gradient Boosting
- Neural Network

1.
## Results

![](RackMultipart20200507-4-1dbjc1r_html_3ded833faa65a9a.png)

To assess our results, we utilized linear regression, lasso regression, random forest, gradient boosting and neural network models. There are 3 noteworthy findings here:
 -First, linear model with unnormalized Y variable showed the weakest performance
 -Second, Ensemble models, RF and GB performed quite similar results
 -And third, despite a weaker R Squared, Neural Network yielded the better Test RMSE

![](RackMultipart20200507-4-1dbjc1r_html_2afa90ce61bd5ab8.png)

In terms of Train vs Test performance comparisons, we observed 2 major outcomes:

First, unlike any other models, Train error of Gradient Boosting is lower than that of its Test error indicating that Gradient Boosting model tends to overfit.

Secondly, we also took a stab at combining RF + GB and Lasso yet it didn&#39;t outperform others.

![](RackMultipart20200507-4-1dbjc1r_html_c360f81e2059d6c.png)

Apart from these, we also employed K-means clustering to measure its impact on prediction results. Based on total within sum of squares plot, we defined the optimal clusters as 2. However, as opposed to what we expected, clustered models did not improve our results.

![](RackMultipart20200507-4-1dbjc1r_html_954355a1b09bda44.png)

1.
## Conclusions and recommendations

Having analyzed the result of the predicting models and have made some research on the existing literature, we can conclude that indicators in 10k filings that publicly traded companies release yearly affect in some degree the stock price variation, but they are not enough to accurately predict it (high RMSE and low R2).

Regularly, clustering algorithms tend to increase the accuracy of predicting models. However, that is not a rule of thumb, and this case seems to be one of the exceptions. The heterogeneity of the data does not allow clustering methods to help to improve the accuracy of stock price variation predictions.

The best models are Random Forests, Gradient Boosting and Neural Networks. We had hoped for a significant improvement in predictions with Neural Networks, but the magic didn&#39;t happen due to the scarce volume of data.

Finally, this work agrees with other studies that appear in the existent literature in that stock price variations depend on a wide array of factors. Although company performance is one of these factors, it is not the one that drives the stock behavior.

6

