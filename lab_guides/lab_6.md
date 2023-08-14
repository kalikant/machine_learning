

<img align="right" src="./logo.png">


**Lab  6: Regression Analysis - Part I**


#### Pre-reqs:
- Google Chrome (Recommended)

#### Lab Environment
Notebooks are ready to run. All packages have been installed. There is no requirement for any setup.

All examples are present in `~/work/machine-learning-essentials-module1/lab_06` folder. 

**Problem Statement**

This data is about the amount spent on advertising certain products through various media channels like TV, radio, and newspaper. The goal is to predict how the expense on each channel affects the sales and is there a way to optimize the sales?

Importing the necessary packages.

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.425.png)

Loading and exploring the data.

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.426.png)

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.427.png)

What are the **features**?

1. TV: Dollars spent on TV ads for a single product in a given market (expressed in multiple of thousands).
1. Radio: Dollars spent on Radio ads (expressed in multiple of thousands).
1. Newspaper: Dollars spent on Newspaper ads (expressed in multiple of thousands).

What is the **response**?

- **Sales**: sales of a single product in a given market (in thousands of widgets).

Dimensions of the data

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.428.png)

Find the missing values from different columns:

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.429.png)

Let us showcase the relationship between the feature and target variables:

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.430.png)

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.431.png)


From the relationship diagrams above, it can be observed that there is a linear relationship of the features such as TV ad, radio ad with the sales. A linear relationship typically looks like: 

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.432.png)

Hence, we can build a model using the Linear Regression Algorithm.

**Simple Linear Regression**

Building Simple Linear Regression Model to predict the sales based on TV ads,

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.439.png)

### **Interpreting the model**
How do we interpret the coefficient for spends on TV ad (β1)?

- A "unit" increase in spending on a TV ad is **associated with** a 0.04753 "unit" increase in sales.
- Or, an additional $1,000 on TV ads is translated to an increase in sales by $47.53.

### **Prediction using the model**
If the expense on a TV ad is $50000, what will be the sales prediction for that market?

***𝑦* =*𝛽*0 + *𝛽*1*𝑥***

Y = 7.032594 + 0.047537 \* (50)

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.440.png)

Thus, we can predict Sales of 9,409 widgets in that market.

Let’s do the same calculation using the code.

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.441.png)

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.442.png)

Plotting the least Squares Line:

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.443.png)

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.444.png)

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.445.png)

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.446.png)

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.447.png)

### **Model Confidence**
**Question:** Is the linear regression a low bias/high variance model or a high bias/low variance model?

**Answer:** It is a High bias/low variance model. Even after repeated sampling, the best fit line will stay roughly in the same position (low variance), but the average of the models created after repeated sampling does not do a great job in capturing the perfect relationship between the two variables (high bias). Low variance is helpful when we don't have less training data! 

If the model has calculated a 95% confidence interval for our model coefficients, it can be interpreted as follows: If the population, from which this sample is drawn, is **sampled 100 times**, then approximately **95 (out of 100) of those confidence intervals** shall contain the "*true*" coefficients.
###
In the coming sections, we discuss more about bias and variance in detail.

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.448.png)

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.449.png)

### **Hypothesis Testing and p-values**
**Hypothesis testing** is closely related to the confidence intervals. We start with a **null hypothesis** and an **alternate hypothesis** (*i.e.,* opposite to the *null*). Subsequently, we check whether the data **rejects the null hypothesis** or **fails to reject the null hypothesis**.

The conventional hypothesis test is as follows:

- **Null hypothesis:** No significant relationship exists between the TV advertisements and the Sales (and hence, *𝛽*1 equals zero). 
- **Alternative hypothesis:** A significant relationship exists between the TV advertisements and Sales (and hence, *𝛽*1 is not equal to zero).

How do we test this potential relationship? We reject the null hypothesis (and thus believe the alternative hypothesis) if the 95% confidence interval **does not include zero**. The **p-value** represents the probability that the coefficient is actually zero.

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.450.png)


![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.451.png)

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.452.png)

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.453.png)

If the 95% confidence interval **includes zero**, then the p-value for that coefficient will be **greater than 0.05**. If the 95% confidence interval **does not include zero**, the p-value for the coefficient will be **less than 0.05**. 

Thus, a p-value of less than 0.05 shows that the relationship between the two features in consideration is statistically significant. Conventionally, a cut-off p-value of 0.05 is used for such analysis.

In this case, the p-value for TV ads is far smaller than 0.05, which means that a statistically significant relationship exists between the TV advertisements and Sales.

Note that we generally ignore the p-value for the intercept.


**Multiple Linear Regression**

Till now, we have studied the models based on only one feature. Now, we’ll include models with multiple features and investigate the potential relationship between those features and the target column. This is called **Multiple Linear Regression**.


**Import our libraries**

The first thing we need to do is to import the libraries we will be using in this case study.

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.474.png)


**Load the Data into Pandas**

This dataset was downloaded from the World Bank website; if you intend to visit the website yourself, you can visit the following link. There is a tremendous amount of data available for free, that can be used across a wide range of models.

Link: <https://data.worldbank.org/>

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.475.png)

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.476.png)

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.477.png)



![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.478.png)

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.479.png)

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.480.png)

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.481.png)

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.482.png)

We can observe that a few missing values are represented as (..) in the dataset, we need to replace all the (..) values with "*nan*" as these represent the missing values in our dataset.

We set the index of our data frame using the set\_index() function to the Year column. This step will make the data selection easy. After we have defined the index, we convert the entire data frame to a float data type and then select the years ranging from 1969 to 2016. These years are selected because they do contain missing values.

To make the selection of the columns a little easier, we have renamed all the columns by creating a dictionary where the keys represent the old column names and the values associated with those keys are the new column names.

Finally, I have checked for potential missing values using isnull().any(), which will return *true* for a given column if any values are missing, and then printed the head of the data frame.

**Check for Multicollinearity**

The first thing to do after loading our data is to validate an assumptions of our model; in this case, we will be checking for multicollinearity.

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.483.png)

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.484.png)

Looking at the heatmap along with the correlation matrix, we can identify a few highly correlated variables. For example, if you look at the correlation between birth\_rate and pop\_growth, it shows a maximal value of 0.98. This is an extremely high correlation and requires removal. Logically, it makes sense that these two are highly correlated; if you are having more babies, then the population should be increasing.

However, we should be more systematic in our approach of removing highly correlated variables. One method for such purpose is the Variance\_Inflation\_Factor, which is a measure of how much a particular variable is contributing to the standard error in the regression model. When significant multicollinearity exists, the variance inflation factor will be huge for the variables in the calculation.

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.485.png)

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.486.png)

Looking at the above data, we get some confirmation about our suspicion. It makes sense to remove either birth\_rate or pop\_growth, and some of the consumption growth metrics. Once we remove those metrics and recalculate the VIF, we get a passing grade and can move forward.


**Describe the Dataset**

Before we get to an in-depth exploration of the data, or even building the model, we should investigate the data a little more and see how the data is distributed and look for potential outliers. I will be adding a few more metrics to the summary data frame, so that it includes a metric for three standard deviations below and above the mean.

I will store my information in a new variable called *desc\_df*.

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.487.png)

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.488.png)

-----
It is worth mentioning that we only have 50 observations, but 6 (minus the 3 we dropped) exploratory variables. Many people would argue that we need more data to have these many exploratory variables and this indeed is a correct statement. Generally, we should aim for at least 20 instances for each variable; however, the opinions vary and some statisticians argue that only 10 instances would also suffice.. Regardless,  we only end up with 4 exploratory variables, so that we will satisfy that rule.

Looking at the data frame up above, a few values are standing out, for example, the maximum value in the *broad\_money\_growth* column is almost four standard deviations above the mean, which is such an enormous value qualifies as an outlier.

### **Filtering the Dataset**

To drop or not to drop a value, that is the question. Generally, if we believe that there is an erroneous entry in the data, we should remove it. However, in this situation, the values that are being identified as outliers are correct values and are not errors. Both of these values were produced during specific moments in time. The one in 1998 was right after the Asian Financial Crisis, and the one in 2001 was right after the DotCom Bubble, so it is entirely conceivable that these values were produced in extreme albeit rare conditions. **For this reason, I will NOT be removing these values from the dataset as they recognize actual values that took place.**

Imagine if we wanted to remove the values that have an amount exceeding three standard deviations. How will we approach this? Well, if we leverage the numpy and the scipy modules, we can filter out the rows by using the *stats.zscore* function. The Z-score is the distance of a datapoint in terms of  the number of standard deviations from the mean ., Hence, if it is less than 3 we keep it in the dataset, otherwise we drop it. From here, I have also provided a way to identify what rows were removed by using the *index.difference* the function, which will show the difference between the two datasets.

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.489.png)

**Build the model**

Now that we have loaded, cleaned, and explored the data, we can proceed to the next part, which is building the model. The first thing we need to do is, to define our exploratory and our explanatory variables. From here, let’s split the data into training and testing set; a healthy ratio is 20% testing and 80% training but a 30% 70% split also works.

After splitting the data, we will create an instance of the linear regression model and pass through the X\_train and y\_train variables using the *fit( )* function.

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.490.png)

**Exploring the output**

With the data now fitted to the model, we can explore the output. The first thing we should do is to look at the intercept of the model, and then print out each coefficient value of the model. I print everything out using a loop to make it more efficient.

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.491.png)

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.492.png)

The intercept term is the value of the dependent variable, when all the independent variables are equal to zero. For each slope coefficient, it is the estimated change in the dependent variable for a one unit change in that particular independent variable, holding the other independent variables constant.

For example, if all the independent variables are equal to zero, the gdp\_growth would be 2.08%. If we look at the gross\_cap\_form\_growth, while keeping other independent variables constant, we would say that a 1 unit increase in gross\_cap\_form\_growth would lead to a 0.14% increase in GDP growth.

We can also make predictions using our newly trained model. The process is simple; we call the predict method and then pass through some values. In this case, we have some values predefined with the x\_test variable, so we will pass that through. Once we do that, we can select the predictions by slicing the array.

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.493.png)

**Evaluating the model**

**Using the Statsmodel**

For diagnosing the model easier, we will, from now on be using the statsmodel module. This module has built-in functions that will help in fast and easy calculations of metrics. However, we will need to "*rebuild*" our model using the statsmodel module. We do this by creating a constant variable, call the *OLS()* method and then the *fit()* method. Now we have a new model, and the first thing we need to do is to ensure that the assumptions of our model hold. This means that we should check the following:

- Regression residuals must be normally distributed.
- The residuals are homoscedastic.
- Absence of multicollinearity (we did this above).
- No Autocorrelation.


**Checking for HeteroScedasticity**

To check for heteroscedasticity, we can leverage the *statsmodels.stats.diagnostic* module. This module gives us a few test functions we can run, the Breusch-Pagan and the White test for heteroscedasticity. The **Breusch-Pagan is a general test for heteroscedasticity, while the White test is a unique case.**

- The null hypothesis for both the White’s test and the Breusch-Pagan test states that the variances for the errors are equal:

**H0 = σ2i = σ2**

- The alternate hypothesis (the one you’re testing), states that the variances are not equal:

**H1 = σ2i ≠ σ2**

We aim to fail to reject the null hypothesis, with a target high p-value because that implies that we found no heteroscedasticity in our dataset.

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.494.png)

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.495.png)

**Checking for Autocorrelation**

We will go to our favorite *statsmodels.stats.diagnostic* module, and utilize the *Ljung-Box* test for no autocorrelation of residuals. Here:

**H0: The data are random.**

**Ha: The data are not random.**

That means that we want to fail to reject the null hypothesis, or specifically we want a large p-value because it would imply having no autocorrelation. To use the *Ljung-Box* test, we will call the *acorr\_ljungbox* function, pass through the *est.resid* and then define the lags.

The lags can either be calculated by the function itself, or we can calculate them manually. If the function handles it, the max lag will be min((num\_obs // 2 - 2), 40), however, there is a rule of thumb that for the non-seasonal time series, the lag is min(10, (num\_obs // 5)).

Moreover, we can also visually check for the autocorrelation by using the *statsmodels.graphics* module to plot a graph of the autocorrelation factor.

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.496.png)

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.497.png)

**Checking for Normally Distributed Residuals**

This is an easy check and is visually possible. It **requires using a QQ pplot, which helps us to assess if a dataset plausibly came from some theoretical distribution such as a Normal or exponential.** Remember that it is just a visual check and not an air-tight proof, so it is somewhat subjective.

Visually what we are looking for is the data that hugs the line tightly or is sitting close to the line; this would give us confidence in our assumption that the residuals are normally distributed. Now, it is highly unlikely that the data will perfectly hug the line, so our observation and conclusion are very subjective.
##
## **Checking the Mean of the Residuals Equals 0**
Additionally, we need to check another assumption, that the mean of the residuals is equal to zero. A mean value close to zero is a good thing and we can proceed to the next step. On a side note, it is not uncommon to get a mean value that is not exactly zero; which happens due to rounding errors. However, if the mean value is very close to zero, we can confidently use it. In the example below, you may see that the mean value is not exactly zero.

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.498.png)

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.499.png)

**Measures of Error**

We next examine that how well our data fit the model, so we take the y\_predictions and compare them to our y\_actuals which will be our residuals. From here we can calculate a few metrics to help quantify how well our model fits the data. Here are a few popular metrics:

- **Mean Absolute Error (MAE):** MAE Is the mean of the absolute value of the errors. This gives us an estimate of magnitude but no sense of direction (too high or too low).
- **Mean Squared Error (MSE):** MSE Is the mean of the squared errors. MSE is more popular than MAE in such calculations, because MSE "*punishes*" more significant errors.
- **Root Mean Squared Error (RMSE):** RMSE Is the square root of the mean of the squared errors. RMSE is even more favored in such calculations, because it allows us to interpret the output in y-units.

Fortunately, both **sklearn** and **statsmodel** contain functions, that can calculate these metrics. The examples below were calculated using the *sklearn* library and the *math* library.



![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.500.png)

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.501.png)

**R-Squared**

The R-Squared metric provides us a way to measure the goodness of fit or, in other words, how well our data fits the model. A higher value of the R-Squared metric means a better fit for our model. However, one limitation of this calculation is that the value of R-Square increases as the number of features increase in our model. This implies that if we keep adding variables including poor choices, the R-Squared will go further up! **A more popular metric is the adjusted R-Square which penalizes more complex models, or in other words, the models with more exploratory variables.** In the example below, the regular R-Squared value has been calculated; however, the statsmodel summary calculates the adjusted R-Squared below.

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.502.png)

**Confidence Intervals**

Let us look at our confidence intervals. Keep in mind that by default, the confidence intervals are calculated using 95% intervals. We interpret confidence intervals by saying that if we sample our target population a 100 times, **approximately 95 of those confidence intervals would contain the "true" coefficient.**

Why do we provide a confidence range? Well, this is because we only have a sample of the population, and not the entire population to collect data from. Because of this limitation, that the "true" coefficient may exist in the interval below or it may not exist although we are not sure about that. We provide some uncertainty by providing a range, usually 95%, where the coefficient is probably in.

- Want a narrower range? **Decrease your confidence**.
- Want a wider range? **Increase your confidence**.

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.503.png)

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.504.png)

**Hypothesis Testing**

With hypothesis testing, we try to determine the statistical significance of the coefficient estimates. This test is outlined as the following:

- **Null Hypothesis:** There is no significant relationship between the exploratory and the explanatory variables.
- **Alternative Hypothesis:** There is a significant relationship between the exploratory and the explanatory variables.
- If we **reject the null**, we imply that there is a significant relationship between exploratory and explanatory variables, but the coefficients do not equal 0.
- If we **fail to reject the *null***, it means that there is no relationship, and the coefficients do equal 0.

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.505.png)

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.506.png)

This is difficult to explain, but we have a few insignificant coefficients in such analysis. The first of them is the constant itself, so technically this should be dropped. However, we will see that once we remove the irrelevant variables, the intercept becomes significant. **If it is still not significant, we start the intercept at 0 and assume that the cumulative effect of X on Y begins from the origin (0, 0).**  Along with the constant, we have *unemployment* and *broad\_money\_growth* both come out as insignificant.

**Create a Summary of the Model Output**

Let us create a summary of some of our keep metrics; *sklearn* does not have a good way of creating this output, so we calculate all the parameters ourselves. Let us avoid this calculation and use the statsmodel.api library. With this library, we can create the same model we created above, but we also leverage the *summary()* method to create an output. Some of the metrics might differ slightly, but they generally should be the same.

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.507.png)

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.508.png)

The first thing to notice is that the p-values are now easier to read and we can now remove the coefficients with p-values greater than 0.05. We also have the 95% confidence interval (described up above), coefficient estimates (described up above), standard errors, and t-values.

The other metric that stands out is the Adjusted R-Squared value which is 0.878, lower than the R-Squared value. This makes sense since we were probably docked for the complexity of our model. However, an R-Squared over 0.878 is still very strong.

The only additional metrics we describe here is the t-value which is the coefficient divided by the standard error. The higher the t-value, the more evidence we have to reject the null hypothesis. Also, remember that the standard error is the approximate standard deviation of a statistical sample population.




**Remove the Insignificant Variables**

Now that we have identified the insignificant variables, we should remove them from the model and refit the data to see what we get. The steps are the same. The only thing I have changed is the removal of additional columns from the data frame.

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.509.png)

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.510.png)

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.511.png)


**Looking at the output, we find that all the independent variables are significant and the constant is significant.** We could rerun our test for autocorrelation and, but this will take you to the same conclusions we found above, so I have decided to leave that from the lab. At this point, we can interpret the formula and begin making predictions. Looking at the coefficients, we can say that the *pop\_growth, gross\_cap\_from\_growth,* and *hh\_consum\_growth* all have a positive effect on GDP growth. Additionally, we can say that the gov\_final\_consum\_growth has a negative effect on GDP growth, This is a slightly surprising finding, but we will have to find its reason.

**Save the Model for Future Use**

We will probably use this model in the future, so let us save our work for the future use. Saving the model can be achieved by storing our model in a pickle, which is storing a *python* object as a character stream in a file, which can be later reloaded for use.

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.512.png)

**Summary**

Regression models describe the relationship between variables by fitting a line to the observed dataset. Generally, the linear regression models use a straight line. Regression allows us to estimate how a dependent variable changes with the change in the independent variables. Simple linear regression is used to estimate the relationship between two quantitative variables, whereas multiple linear regression is utilized to estimate the relationship between two or more quantitative and qualitative variables. 









**Assessment**

**Choose the appropriate option**

1) **Which one of the following are regression tasks?**

A. Predict the age of a person

B. Predict the country from where the person comes from

C. Predict whether the price of petroleum will increase tomorrow

D. Predict whether a document is related to science

2) **Which of the following plots is used for normality test?**

A. Scatter plot

B. Bar plot

C. qqplot

D. None of these

3) **Which of the following tests is used for heteroscedasticity?**

A. AD

B. Ljung-Box

C. Breusch-Pagan

D. All of the above

4) **Which of the following tests is used for *autocorrelation*?**

A. AD

B. Ljung-Box

C. Breusch-Pagan

D. White test

5) **VIF > 10 is said to be:**

A. No Multicollinearity

B. Less Multicollinearity

C. High Multicollinearity

D. None of the above

**Fill in the spaces with appropriate answers**

1) \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_ diagrams are graphs of the data that are helpful in displaying the relationships between variables.

2) The SSR is sometimes referred to as the variability in Y, as explained by \_\_\_\_\_\_\_\_\_\_\_\_\_\_.

3) If the adjusted R2 \_\_\_\_\_\_\_\_\_\_\_\_\_\_ when a new variable is added, it would be an indication that the variable should not remain in the model.

4) Regression analysis is sometimes called \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_.

5) Complete the following equation: SST = SSR + 

**True or False**

1) Simple Linear regression is built among more than two variables.

   1) True

   2) False

2) Multiple Linear regression is built between two or more than two variables:

   1) True

  2) False

3) Autocorrelation is having relationship between observation:

   1) True

   2) False

4) We drop the variable from the model if the p-value is insignificant:

   1) True

   2) False

5) Difference between actual and predicted values is called errors (or) residuals:

   1) True

   2) False


**Programming Assignment** 

Using the data in the below URL, 

<https://www.kaggle.com/ruiromanini/mtcars>

1) Build the multiple linear regression model and predict mpg.

2) Validate the assumptions for multiple linear regression.

3) Evaluate the model metrics on the test set.

**Solutions:** Refer to page 270

**Solutions for Assessment**

**Choose the appropriate options**

1) A

2) C

3) C

4) B

5) C

**Fill in the spaces with appropriate answers**

1) Scatter

2) The regression equation

3) Decreases

4) Least-Squares Regression

5) SSE

**True or False**

1) False

2) True

3) True

4) True

5) True



**Lab  6: Regression Analysis – Part II**

**AIM**

The aim of the following lab exercises is to perform various exercises by writing the corresponding python codes, so that we can get hands-on practice of the descriptive statistics.

The labs for this lab include the following exercises.

- Handling Categorical Predictors
- Regularization
- Polynomial Regression

For these exercises, we will use python3 and jupyter notebook IDE.



**Task 1): Handling Categorical Predictors with Two Categories**

Till now, all the predictors have been numeric; what if one of the predictors is categorical?

We will use advertisement data for this task. We’ll create a new feature called Scale, and will randomly assign observations as small or large.

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.569.png)

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.570.png)

For the *scikit-learn* library, all the data must be represented numerically. If a feature has only two categories, we can simply create a dummy variable which represents the categories as a combination of binary values.

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.571.png)

Let us redo the multiple linear regression problems and include the "*IsLarge*" predictor as follows:

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.572.png)

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.573.png)

How do we interpret the coefficient for IsLarge? For a given TV/Radio/Newspaper ad expenditure, if the average sales increase by 57.42 widgets, we can consider it as a large market.

What if the 0/1 encoding is reversed? The value of the coefficient will still be the same, however the sign will change from positive to negative.


**Task 2) Handling Categorical Variables with More Than Two Categories**

Let us create a new column called the **Targeted Geography**, and randomly assign observations to be **rural, suburban or urban**.

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.574.png)

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.575.png)

Here, we need to represent the ‘Targeted Geography’ column numerically. But mapping urban=0, suburban=1, and rural=2 will mean that the rural is two times suburban which is not the case. Hence, we will create another dummy variable as follows:

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.576.png)

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.577.png)

What does the encoding say?

- Rural is encoded as Targeted Geography\_suburban=0 andTargeted Geography\_urban=0
- Suburban is encoded as Targeted Geography\_suburban=1 and Targeted Geography\_urban=0
- Urban is encoded as Targeted Geography\_suburban=0 and Targeted Geography\_urban=1

Now the question is: Why have we used two dummy columns instead of three?

Because by using only two dummy columns, we can capture the information of all the 3 columns. For example, if the value for Targeted Geography\_urban and Targeted Geography\_rural are 0, we can infer that the data belongs to Targeted Geography\_suburban.

This is called as *handling the dummy variable trap*. If there are ‘’m’’ number of dummy variable columns, then the same information can be conveyed by the ‘’m-1’’ columns. Let’s include the two new dummy variables in the model.

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.578.png)

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.579.png)

How do we interpret the coefficients?

- If all other columns are constant, then the suburban geography is associated with an average **decrease** of 106.56 widgets in sales for $1000 spent.
- If $1000 is spent in an urban geography, it accounts for an average **increase** in Sales of 268.13 widgets.

A final note about the dummy encoding: If we have categories that can be ranked in an order (*i.e.,* worst, bad, good, better, and best), we can potentially represent them numerically as (1, 2, 3, 4, and 5 respectively) using a single dummy column.

**Task 3) Regularization the algorithms**

Problem statement

The dataset contains several important parameters to consider during the application for Masters Programs. This dataset was built with the purpose of helping students in shortlisting universities with their profiles. The predicted output gives them a fair idea about their chances of admission into a particular university.

**Importing necessary packages**

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.580.png)

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.581.png)

**Descriptive statistics**

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.582.png)

**Checking the missing values**

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.583.png)

There are a few missing values in GRE Score, TOEFEL Score, and the University Rating variables. Let us impute the missing values as follows:

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.584.png)

Now the data looks good and there are no missing values. Also, the first column includes the serial numbers, which are not needed for statistical analysis and can be omitted.

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.585.png)

Let’s visualize the data and analyze the relationship between independent and dependent variables.

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.586.png)

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.587.png)

The data distribution looks close to normal without showing any skewness. Great so let’s go ahead!

Let us observe the relationship between the independent and dependent variables.

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.588.png)

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.589.png)


![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.590.png)

The relationship between the dependent and independent variables looks fairly linear. Thus, our linearity assumption is satisfied.

Let us move ahead and check for the multicollinearity.

**Data Scaling**

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.591.png)

In this case, we have scaled the data before we check the multicollinearity assumption.

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.592.png)

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.593.png)

Here, we have added the correlation values for all the features. As a rule of thumb, a VIF value greater than 5 means a very robust multicollinearity. We don’t have any VIF greater than 5, so we are good to go.

Let us go ahead and use the linear regression and see how good it fits our dataset. But, first, let us split our dataset into train and test.

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.594.png)

Let’s check the top six observations of y\_train

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.595.png)

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.596.png)

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.597.png)

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.598.png)

Computing R squared value,

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.599.png)

Computing the Adjusted R Squared value,

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.600.png)

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.601.png)

The R Squared score is 84.15% and adjusted R Squared is 83.85% for our training set, which means that we are not being penalized by the use of any feature.

Let’s check how well our model fits the test data.

Now let’s check if our model is overfitting the data by using regularization.

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.602.png)

Thus, it looks like our model R squared is less on the test data.

Let’s see if our model is overfitting our training data.

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.603.png)

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.604.png)

We have derived the best possible alpha value, which is the shrinkage factor for LASSO regression.

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.605.png)

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.606.png)

Our R squared for test data (75.34%) comes the same as before using regularization. So, it is fair to say that the OLS model did not overfit the data.

**Task 4) Polynomial Regression**

Importing necessary packages

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.607.png)

Importing the data

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.608.png)

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.609.png)

Here, we can see 3 columns in the dataset. The problem statement here is to predict the salary related to the Position and Level of the employee. We may notice that the position and the level are related, as level is an alternate way of conveying the position of the employee in the company. So, essentially the position and level are conveying the same kind of information. As the level is a numeric column, let’s use that in our model. Hence, level is the feature or X variable and salary is a label or the Y variable.

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.610.png)

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.611.png)

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.612.png)

Generally, we divide our dataset into two parts:

1) The training dataset to train our model. 

\2)   The test dataset to test our prepared model.

To learn Polynomial Regression, we take a comparative approach. First, we create a linear model using Linear Regression and then we prepare a Polynomial Regression model and see how the two models compare to each other.

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.613.png)

Visualizing the Linear Regression results,

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.614.png)

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.615.png)

Here, the red dots represent the actual data points, and the blue straight line represents what our model has created. It is evident from the diagram above that a Linear Regression model does not fit our dataset well. Therefore, let us try with a Polynomial model.

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.616.png)

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.617.png)

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.618.png)

As evident, we are using the Linear Regression for the Polynomial Regression as well.

Why is it so?

It is because the ‘’Linear’’ in Linear Regression does not consider the degree of the Polynomial equation in terms of the dependent variable(X). Instead, it considers the degree of the coefficients.

Mathematically,

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.619.png)

It is considering the power of X, but the powers of a, b, c etc. And as the coefficients are only degree 1, hence the name Linear Regression.

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.620.png)

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.621.png)

Even so, a second degree is also not a good fit. Now, we will try to increase the degree of the equation *i.e.*, we will try to see whether we get a good fit at a higher degree or not. After some hit and trial, we can observe that the model provides the best fit for the 4th-degree polynomial equation.

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.622.png)

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.623.png)

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.624.png)

Here, we can observe that our model now can accurately fit the dataset. This kind of fit might not be the case with the actual business datasets. We are getting a brilliant fit, partly because of a small number of data points.








**Summary**

In this lab, we have learned the following: 

- **Cost function optimization technique,** which helps us to find the parameters with the least/minimum error using the concept of Gradient-Descent technique.
- **Categorical Features in a Regression model**, are handled by converting those categories into the Dummy variables and building the model based on these dummy variables for better computation of the model and avoiding ‘Dummy variable trap’.
- **Feature Scaling** - Two different methods of feature scaling *i.e* Normalization and Standardization. These methods make the data independent of units of measure, which ease the computation and increase the performance of the model.
- **Feature Selection** - We learned about the "*Backward, Forward, and Step-Wise elimination*" techniques to identify the best features and reduce the overfitting problems.
- **Regularization** - Regularization technique is utilized to penalize the parameters responsible for overfitting the model by shrinking the value of the parameters close to or equal to zero, using Ridge, LASSO, and Elasticnet respectively.
- **Polynomial Regression** - Polynomial Regression is employed to capture the non-linear relationship between the dependent and independent variables by including polynomial terms/variables into the model.












**Assessment**

**Choose the appropriate option**

1) **What is the cost function?**

A. Difference between predicted and actual value

B. Difference between the observations

C. Difference between the variables

D. None of the above

2) **Optimization technique is used to**

A. Find the parameters with highest error

B. Calculate the residuals

C. Find the parameters with least error

D. None of the above

3) **Feature scaling is used for**

A. Regularization of parameters

B. Making the data independent of units

C. Calculating the residuals

D. All of the above

4) **Following is not the feature selection technique**

A. Forward selection

B. Backward elimination

C. ElasticNet

D. LASSO

5) **From the below formula, lamba is called as**

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.557.png)

A. Shrinkage parameter

B. Learning rate

C. Estimating parameter

D. None of the above

**Fill in the spaces with appropriate answers**

1) Categorical variables are converted to \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_.

2) Normalization transforms the values between \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_.

3) Gradient Descent is an \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_ technique.

4) The Dummy Variables Trap leads to \_\_\_\_\_\_\_\_\_\_\_\_\_\_ in the data.

5) List down the Regularization techniques \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_.

**True or False**

1) Polynomial regression is used to capture the linear relationship between the independent and dependent variables.

   1) True

   2) False

2) Ridge Regression can completely shrink the parameter to Zero

   1) True

   2) False

3) ‘’N’’ number of categories require N-1 Dummy Variables.

   1) True

   2) False

4) Normalization Scales down the values between 0 & 1.

   1) True

   2) False

5) Bias is also called Test Error.

   1) True

   2) False

**Programming Assignment** 

Using the data in the below URL, 

<https://www.kaggle.com/ruiromanini/mtcars>

1) Build a regression model by including the categorical variables
1) Check if the model is *generalized* or not using the regularization techniques.

**Solutions:** Refer to page 47

**Solutions for Assessment**

**Choose the appropriate options**

1) A

2) C

3) B

4) C

5) A

**Fill in the spaces with appropriate answers**

1) Dummy Variables

2) 0 & 1

3) Optimization

4) Multicolinearity

5) Ridge, Lasso, Elasticnet.

**True or False**

1) False

2) False

3) True

4) True

5) False







