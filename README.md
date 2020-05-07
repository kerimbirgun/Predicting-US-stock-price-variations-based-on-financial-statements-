# Predicting-US-stock-price-variations-based-on-financial-statements-



### Abstract

Financial markets are continuously affected by several types of factors that disturb stock performance and lead to arbitrage opportunities. Investors around the world are continually monitoring the markets for such disturb causes to anticipate stock price changes and maximize their profits. However, factors can stem from a wide range of situations that take place in the global and business environment. The disclosure of annual financial information has shown some effect on stock price behavior. For this reason, the purpose of this study is to evaluate whether a 10k report provides enough information to predict stock price variation that can be leveraged to anticipate arbitrage conditions. The target population is represented by +4,000 U.S companies that were listed in the N.Y Stock Exchange during 2018. The work results were obtained using several machine learning models learned during the Frameworks &amp; Methods I and II courses and show the impact of financial information on the stock price behavior.

### Introduction

A winning rule on capital markets is to anticipate stock prices to devise a successful trading strategy that maximizes profits. For this reason, the information offered by listed companies represents essential support for investors seeking to anticipate capital markets and take advantage of stock price variations. However, recent history has shown that company performance is just one of many other elements that influence stock prices.

Financial information provided by reports, like the 10k filings in the U.S stock market, are the main mean of communication of companies with investors and securities authorities. Besides, several studies and specialists have conducted studies trying to unveil the most significant financial indicators, within the statements, that influence on stock performance, measured in terms of stock price changes.

The purpose of this project is to understand whether it is possible to predict stock price variations -year over year- using information from annual financial statements released by U.S., publicly traded companies.

The utility of the study goes beyond putting in practice some concepts learned during Framework &amp; Methods courses. The ability to accurately predict stock performance after financial information is released may offer a wining advantage to investors on capital markets. Therefore, in order to make viable this goal, a sample was collected, analyzed, and some predicting models were built. The sample consisted of 200 financial indicators of more than 4,000 US publicly traded companies in 2018.

### Financial indicators in determining stock performance

The connection between the information provided by the financial statements and stock prices was proved since 1968 by Ray Ball and Philip Brown in their study released by the Journal of Accounting Research [CITATION Bal68 \l 1033]. Some other studies had stated that, even though ratios concerning value creation offer more information than traditional ratios in explaining the overall performance of a stock, they must not replace traditional ratios such as Earnings Per Share (EPS), Return On Assets (ROA) or Return On Equity (ROE) [CITATION Che97 \l 1033].

Moreover, other authors have investigated the linkage between company&#39;s value creating ration (EVA), traditional accounting ratios and their capacity to explain stock behavior. Some of them concluded that ROA has the highest explanatory power concerning this return, followed by ROE and EPS (Arab Salehi and Mahmoodi, 2011).

### Problem statement

This work aims to probe whether a 10k report provides enough information to predict stock price variation accurately. In this sense, the questions that we want to solve using data and analytical techniques are:

- Based on 10k reports&#39; information, could it be possible to predict stock prices YoY variation accurately?
- Do dimension reduction and clustering techniques can help to enhance the results of ML algorithms -such as random forest and GB- predicting stock price changes?
- What variables, included or derived from annual financial statements or 10k reports, drive stock price variation explanation?

### The Data

The sample constitutes a dataset containing 222 financial indicators that were reported by more than 4,000 U.S publicly listed companies in their 2018-10k filings. A Form 10-K is an annual report required by the U.S Security and Exchange Commission (SEC), that provides a comprehensive summary of a company&#39;s financial performance. The 10-K includes information such as company history, organizational structure, executive compensation, equity, subsidiaries, and audited financial statements [CITATION Wik201 \l 1033].

In addition to these 222 financial indicators, the third-to-last column, &quot;Sector,&quot; lists the sector of the Stock Exchange to which each company belongs. The graphic below illustrates the eleven sectors and the number of companies, or stocks, by each of them (basic materials, communications, services, consumer cyclical, consumer defensive, energy, financial services, healthcare, industrials, real state, technology, and utilities).

The second-to-last column contains the percentage variation of each stock price. The percentage variation was estimated based on the year&#39;s first and last trading day, providing the twelve months rolling difference.

![](RackMultipart20200507-4-1xo3llz_html_ca8739edc63dcd8.jpg)

_Figure 1 Stock Exchange sectors within the dataset_

The stock price percentage variation was defined as the response -or dependent- variable across the predicting models built in this study. Whereas the last column, &quot;Class,&quot; is a Boolean variable containing 1 and 0 depending on whether the stock was a buy-worthy or not-buy-worthy option. The stock classification was based on the price percentage variation. An unfavorable change meant a not-buy-worthy share; meanwhile, a positive percentual change meant a buy-worthy option.

This project sought to follow the next process during the models-building and the estimation of the predictions:

## Data Exploration

The purpose of this step was to explore the structure and features of the dataset, the distribution of the predictors and response variable, and the number of missing values across the entire dataset.

![](RackMultipart20200507-4-1xo3llz_html_efa8a012796840ef.jpg)

_Figure 2 Response variable&#39;s initial distribution_

As part of this process was to identify potential outliers in the response variable that had to be corrected in the dataset to train the models better and reduce overfitting issues.

![](RackMultipart20200507-4-1xo3llz_html_fba4aa9f01ddce6f.jpg)

_Figure 3 Outliers in the response variable per sector_

_Table 1 response variable (price percentage variation) quartile distribution by sector_

![](RackMultipart20200507-4-1xo3llz_html_36ed09875738fa90.png)

Likewise, exploring other predictors was possible to identify imbalances across the dataset that could then affect during the data splitting step and the accuracy of the models&#39; predictions.

![](RackMultipart20200507-4-1xo3llz_html_a272c4d2db6d6292.jpg)

_Figure 4 Stock Class initial imbalance across the dataset_

Missing values are a constant issue when dealing with vast amounts of data, and this dataset was not the exception. During the exploration phase, it was possible to identify that almost 10% of the data was missing.


## Data Cleaning

During this phase, missing values were imputed using the missranger package, which estimates the new values based on the variable&#39;s distribution and the surrounding data. Something important to highlight here is the quality of the dataset, given that such amount of missing values was not typical as 10-K forms cannot have missing data.

![](RackMultipart20200507-4-1xo3llz_html_44b65f441cbea236.jpg)

_Figure 5 Missing values distribution_

Correcting outliers was also part of this phase. This study defined a strategy that seeks to eliminate all these extreme values within the response variable. In this sense, all companies, or rows, with percentage variations greater than 150%, were dropped from the dataset. The rationale for this decision was that companies do not tend to have disruptive events that derive such stock price changes.

![](RackMultipart20200507-4-1xo3llz_html_871796f75240dbf.jpg)

_Figure 6 Response variable&#39;s distribution after correcting outliers_

-
## Feature selection

Part of this work aims to understand what variables (stated or estimated) from 10k filings drive stock price variation. Therefore, techniques such as PCA cannot be used, given that we would lose model interpretability. Therefore, due to the high dimensionality (initially, there were 225 predictors in the dataset), this work explored two feature selection approaches:

- Data-driven methods (RF – variable importance):
- Lasso regression – features based on coefficients
- Random Forest – features importance

![](RackMultipart20200507-4-1xo3llz_html_cf2fafce7d5a34b9.jpg)

- Literature review (previous studies variable selection (financial ratios))

We reviewed different papers that tried to prove if it exists a relationship between different financial indicators and the stock price of a company. According to the study of Drummen and Zimmermann (1992) the individual characteristics of companies affect up to 50 % of stock prices. The results suggest that there is significant relationship between accounting earnings and stock prices. Results show that the ratios of working capital to total assets and net profit to sales have a negative impact on stock returns, while the ratios of net profit to total assets and sales to total assets positively affect returns.

In the paper called &quot;Relation between share price and financial indicators in the Brazilian stock market&quot; – 2016, the authors explore the relationship between the share prices of companies listed on the BM&amp;FBovespa with some financial indicators, including earnings per share, book value per share and total assets. This paper states that increases of company profit or assets culminate in a higher price of its shares.

Another work from the Australian stock market, Brimble and Hodgson (2007), proved the relationship between book value and profit, and book value and dividends.

Finally, we found a research that finds some linkages between stock prices and profitability ratios (the ROA, the ROE, the ROCE, among others).
# 1


## Feature Engineering

During the data exploration phase was possible to identify that several indicators had different scales and many of them had skewed distributions. Therefore, this phase aimed to correct these two issues by scaling and normalizing the data.

![](RackMultipart20200507-4-1xo3llz_html_1912148c21742315.png)

An initial linear regression model was used to analyze the dependent variable&#39;s residuals. The residuals&#39; plot showed a high variance as a result of a skewed distribution. Hence, using the bestnormalization R&#39;s package was possible to correct the distribution, reduce residuals variance, and thereby improve the R squared of the predicting models.

-
## Analytical Techniques

Several regressions were ran using a wide range of models seeking to find the best approach to reduce the RMSE and find a higher R2. As part of the process, stocks were clustered to predict their performance, but results didn&#39;t outperform previous approaches. The models that were the following:

- Linear Regression
- Lasso Regression (linear regression using shrinkage)
- Random Forest
- Gradient Boosting
- Average of Random Forest + GBM + Linear Regression
- Neural Network

-
## Results

To run the predicting models, the data was split into train and test samples using CreateDataPartition function of Caret package. The Class variable was used to divide the dataset and fix the imbalance issue.

Once the models were run and the stock price variations were estimated using the train sample, then the models were validated through the test sample, and three noteworthy findings came up:

- First, linear model with unnormalized Y variable showed the weakest performance,

• Second, Ensemble models, RF and GB performed quite similar results,

• And third, despite a weaker R Squared, Neural Network yielded the better Test RMSE.

![](RackMultipart20200507-4-1xo3llz_html_3ded833faa65a9a.png)

In terms of Train vs Test performance comparisons, we observed 2 major outcomes:

- First, unlike any other models, Train error of Gradient Boosting is lower than that of its Test error indicating that Gradient Boosting model tends to overfit.
- Secondly, we also took a stab at combining RF + GB and Lasso yet it didn&#39;t outperform others.

### Clustering then predict

Apart from the previous models, this study also tried to employ clustering algorithms, specifically K-means, to improve the accuracy of the stock price variation predictions. Therefore, the first step of this phase was to determine the optimal number of centroids or clusters in which group the stocks.

![](RackMultipart20200507-4-1xo3llz_html_4280e1578c70d812.jpg)

_Figure 7 Total Within Sum of Square plot_

Based on the Total Within Sum of Square plot was possible to identify that 2 clusters were the optimal number of centers for grouping the data and then estimate the stock price variations. However, the results of this entire process didn&#39;t outperform previous predictions.

![](RackMultipart20200507-4-1xo3llz_html_954355a1b09bda44.png)

### Conclusions

Having analyzed the result of the predicting models and have made some research on the existing literature, we can conclude that indicators in 10k filings that publicly traded companies release yearly affect in some degree the stock price variation, but they are not enough to accurately predict it (high RMSE and low R2).

Regularly, clustering algorithms tend to increase the accuracy of predicting models. However, that is not a rule of thumb, and this case seems to be one of the exceptions. The heterogeneity of the data does not allow clustering methods to help to improve the accuracy of stock price variation predictions.

The best models are Random Forests, Gradient Boosting and Neural Networks. We had hoped for a significant improvement in predictions with Neural Networks, but the magic didn&#39;t happen due to the scarce volume of data.

Finally, this work agrees with other studies that appear in the existent literature in that stock price variations depend on a wide array of factors. Although company performance is one of these factors, it is not the one that drives the stock behavior.

### References

#

Ball, R., &amp; Brown, P. (1968). An empirical evaluation of accounting income numbers. _Accounting Research_, 159-178.

Chen, S., &amp; Dodd, J. L. (Fall97). Economic Value Added (EVATM): An empirical examination of a new corporate performance measure. _Journal of Managerial Issues_, 318.

Mironiuc, M., &amp; Robu, M.-A. (2012). Obtaining a Practical Model for Estimating Stock Performance on. _Procedia - Social and Behavioral Sciences_, 422-427.

Wikipedia. (2020, 04 26). _Wikipedia_. Retrieved from Form 10-K: https://en.wikipedia.org/wiki/Form\_10-K

[1](#sdfootnote1anc)&quot;The relationship between financial ratios and the stock prices of selected European food companies listed on Stock Exchanges&quot;, 2019.
