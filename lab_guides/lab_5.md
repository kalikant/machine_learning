
<img align="right" src="./logo.png">


**Lab  5: Data Cleaning and Exploratory Data Analysis**

Not every dataset is a perfectly curated group of observations with no missing values or anomalies. Real-world data is messy and requires cleaning and wrangling it into an acceptable format before we start the analysis. Data cleaning is an un-glamorous, but necessary part of most actual data science problems.

We will understand this lab by going through a case study.

#### Pre-reqs:
- Google Chrome (Recommended)

#### Lab Environment
Notebooks are ready to run. All packages have been installed. There is no requirement for any setup.

All examples are present in `~/work/machine-learning-essentials-module1/lab_05` folder. 

**Case Study**

Problem Definition

The first step before we obtain coding is to understand the problem that we are trying to solve with the available data. In this case study, we will work with publicly available **Building Energy** data from New York City.

Data: <https://www1.nyc.gov/html/gbee/downloads/misc/nyc_benchmarking_disclosure_data_reported_in_2017.xlsb>

General Description of the data.

<http://www.nyc.gov/html/gbee/downloads/misc/nyc_benchmarking_disclosure_data_definitions_2017.pdf>

First, we can load the data as Pandas Dataframe and take a look:

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.363.png)

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.364.png)

This is a subset of the full data which contains 60 columns. We can already see a couple of issues here: first, we know that our aim is to predict the **Energy Star score** but we don’t have the mean values for any of the columns. While this isn’t necessarily an issue --- we can get around this problem by making an accurate model of the data without any knowledge of the variables. This has practical implications since at the end of the day, we want to focus on interpretability, and it might be important to understand at least some of the columns.

We should at least understand the **Energy Star score**, which is described as:

A 1-to-100 percentile ranking based on the self-reported energy usage for the reporting year. The **Energy Star score** is a relative measure employed for comparing the energy efficiency of the buildings. 

That clears up the first problem about data. However, we have another issue about the missing values in the data, which are encoded as "Not Available". This is a "string" in Python which means that even the columns with numbers will be stored as "object" data types because Pandas converts a column with any string into a column of all strings. We can obtain the data types of the columns using the following command

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.365.png)

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.366.png)

As expected, some of the columns that clearly contain numbers (such as ft2), are stored as objects. We can’t do numerical analysis on the strings, so these columns need to be converted to number (specially float) data types before any analysis can be performed!

Here is a little Python code that replaces all the "*Not Available*" entries with ‘*not a number’* (np.nan), which can be interpreted as numbers, and then convert the relevant columns to the *float* data type.

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.367.png)

Once the correct columns have numbers, we can start to statistically analyze the data.

**Missing Values**

First, let’s get a sense of how many missing values are there in each column.

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.369.png)

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.370.png)

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.371.png)

While we always have to be careful when removing information, if a column has a high percentage of missing values, there is a  possibility that it might not be useful to our model. 

After removing the missing value, the remaining data in a column may be slightly arbitrary. So for this case study; we will remove any columns which has more than 50% of values as missing. In general, be careful about dropping any information because even if it is not there for all the observations, it may still be useful for predicting the target value.

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.372.png)

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.373.png)

#### Imputing Missing Values

While we dropped the columns with more than 50% missing while cleaning data, we still have quite a few missing observations. Machine learning models cannot deal with any absent values, so we have to fill them in, a process known as **imputation.**

Every value that is *Nan*, represents a missing observation. While there are a number of ways to fill in missing data, we will use a relatively simple method, called median imputation for numeric variables and most frequent (mode) imputation for categorical variables. This method replaces all the missing values in the columns with the median and mode values.

To do this, first, we need to subset the categorical and numerical variables for the imputation process.

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.374.png)

In the following code, we create a *Scikit-Learn* Imputer object with the strategy set to "median" (numeric variable) and "*most\_frequent*" (categorical variables) number. Subsequently, we then train this object on the numeric and category data (using *imputer.fit*) and use it to fill in the missing values in both the numeric and category data (using *imputer.transform*). This means that the missing values in the numeric and category data are filled with the corresponding median and mode values.

Import the *Scikit-Learn* package before we start the imputation process.

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.375.png)

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.376.png)

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.377.png)

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.378.png)

We concatenated the data and saved it as modified data set.

Now let’s check the presence of any null values in the data.

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.379.png)

We can see that there are no null values present in the data set.

**Note:** Theoretically, up to 25 to 30% is the maximum number of missing values in a data set. If the number of missing values is higher,  we might want to drop the variable from analysis. However, this rule varies in practice. At times we have a dataset with the ~50% of missing values but still, the customer insists to use it for analysis. In such cases for practical purposes, we treat the dataset on case to case basis.




**Univariate analysis (Single variable plots)**

The goal is to predict the Energy Star score (Renamed to score in our data), so a reasonable place to start the analysis is to exam the distribution of this variable. A histogram is a simple yet effective way to visualize the distribution of a single variable and is easy to make using *matplotlib*. 

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.382.png)

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.383.png)

This graph looks quite suspicious! The Energy Star score is a percentile rank, which means that we would expect to observe a uniform distribution, with each score assigned to the same number of buildings. However, a disproportionate number of buildings have either the highest, 100, or the lowest, 1, score (higher is better for the Energy Star score).


Let us look at the distribution of the ‘*site EUI*’ variable.

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.384.png)

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.385.png)

Well, this graph shows us another problem: outliers! The graph is incredibly skewed because of the presence of a few buildings with very high scores. It seems that we will have to take a slight detour to deal with the outliers. Let us look at the stats for this particular feature.

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.386.png)

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.387.png)

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.388.png)

Wow! One building is clearly far above the rest

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.389.png)

Looking at the data, it might be worthwhile for someone to follow up with this building owner! However, that is not our problem as statisticians, as we only need to figure out how to handle this information. Outliers can occur for several reasons: typos, malfunctions in measuring devices, incorrect units, or they can be legitimate but extreme values. Thus, the outliers can throw off a model, because they are not indicative of the actual distribution of data.

**Removing Outliers**

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.390.png)

Now, let’s check the distribution of the ‘*site EUI’* variable.

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.391.png)

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.392.png)

This plot looks a little less suspicious and is close to a normal distribution with a long tail on the right side (it has a positive skew).

**Bivariate Analysis**

In order to look at the effect of categorical variables on the score, we can make a density plot, which is colored by the value of the categorical variable. Density plots also show the distribution of a single variable and can be thought of as a smoothed histogram. If we color the density curves by a categorical variable, this will show us the change in distribution based on the class.

The first plot we will make shows the distribution of scores by the property type. In order to not clutter the plot with large data, we will limit the graph to building types that have more than 100 observations in the dataset.

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.393.png)

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.394.png)

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.395.png)

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.396.png)

This graph informs us that we should include the property type for analysis, because this information can be useful for determining the score.

To examine another categorical variable, ‘*’borough’’*, we can make the same graph, but this particular one is time colored by the *borough*.

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.397.png)

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.398.png)

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.399.png)

The borough of the building does not seem to make a significant difference in the distribution of the score in a line similar to the building type. Nonetheless, it might make sense to include the borough as a categorical variable for the final analysis.

**Correlations between Features and Target**

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.400.png)

Top 15

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.401.png)

Bottom 15

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.402.png)

We can observe a number of strong negative correlations between the features and the target. The most prominent of these correlations with the score are the different categories of Energy Use Intensity (EUI), Site EUI (kBtu/ft²), and Weather Normalized Site EUI (kBtu/ft²). Generally, these categories slightly vary in terms of the process in which they are calculated. The EUI is the amount of energy utilized by a building divided by the square footage of the buildings (i.e., unit area) and is a measure of the efficiency of a building. The lower score indicates higher efficacy. Consequently, these correlations make sense: as the EUI increases, the Energy Star Score tends to decrease.

To account for the possible non-linear relationships, we can take square root and natural log transformations of the features and then calculate the correlation coefficients with the score. In this way, we try to capture any possible relationships between the *borough* and building type.

In the following code, we take the log and square root transformations of the numerical variables, one-hot encode the two selected categorical variables (building type and borough), calculate the correlations among all of the features and the score, and display the top 15 most positive and top 15 most negative correlations. This is a lot of work, but with pandas, it is a straightforward task through each step!

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.403.png)

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.404.png)

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.405.png)

After transforming the features, we find the strongest relationships that relate to the Energy Use Intensity (EUI). We observe that the log and square root transformations did not have strong relationships. The positive linear relationships are not very strong, although we do see that a building type of office (Largest Property Use Type\_Office) is slightly positively correlated with the score. This variable is a one-hot encoded representation of the categorical variables for building type.

We can employ these correlations in order to perform the feature selection (coming up in future labs). Right now, let us draw a graph of the most significant correlation (in terms of absolute value) in the dataset which is the *Site EUI* (kBtu/ft^2). We can color the graph by building type to show how it affects the relationship.

In order to visualize the relationship between two variables, we utilize a scatter plot. Moreover, we can also include the additional variables using aspects such as color or size of the markers. Here, we plot two numeric variables against one another and use a different color to represent a third categorical variable.

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.406.png)

The graph shows  a clear negative relationship between the Site EUI and the score. The relationship is not perfectly linear (it has a correlation coefficient of -0.7, but it does look like this feature will be important for predicting the score of a building.

**Multivariate Analysis**

As a final exercise for exploratory data analysis, we can make a pairs plot among several different variables. The pairs plot is a great way to examine multiple variables at once as it shows the scatterplots between pairs of variables and histograms of single variables on the diagonal axis.

Using the seaborn PairGrid function, we can map different plots for the three aspects of the grid. The upper triangle will have the scatterplots, the diagonal will show the histograms, and the lower triangle shows both the correlation coefficient between two variables and a 2-D kernel density estimate of the two variables.

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.407.png)

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.408.png)

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.409.png)

Here, we observe that all the three variables have a negative correlation with our target variable "Score". 

We also utilize the heat map to visualize the correlation between the continuous variables.

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.410.png)

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.411.png)

Let us understand the energy distribution score by grouping the data with categorical variables using facet-grid from the seaborn package.

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.412.png)

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.413.png)

We filter the dataset based on the building types and *boroughs*.

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.414.png)

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.415.png)

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.416.png)





![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.417.png)

By looking at the graph, most of the multifamily buildings with high energy scores are located in Manhattan, Brooklynn, and Bronx *boroughs*. 

The majority of the office and hotel properties are located in Manhattan. Furthermore, it can be observed that the office buildings have high-energy scores and hotel buildings have low-energy scores.



















**Assessment**

**Choose the appropriate option**

1) **Which of the following analyses require more than two variables for analysis?**

A. Univariate analysis

B. Bivariate analysis

C. Multivariate analysis

D. None of the above

2) **Which of the following plots is used to study the relationship between two continuous variables?**

A. Scatter plot

B. Bar plot

C. Histograms

D. None of these

3) **Outliers are caused due to..?**

A. Data entry

B. Mistake in units

C. Legitimate extreme value 

D. All of the above

4) **Univariate analysis is performed on..?**

A. Single variable

B. Two variable

C. More than two variables

D. None of the above

5) **What should be your next step if a variable has more than 80% missing values and the variable is not important for the analysis?**

A. Drop the variable

B. Impute the missing values

C. Remove the observation with missing values

D. None of the above

**Programming Assignment** 

Using the data in the below URL, 

<https://www.kaggle.com/ruiromanini/mtcars>

1) Perform the univariate analysis on the mpg variable.

2) Check the correction between mpg and rest of the continuous variables.

3) Perform multivariate analysis by grouping the data with am, gear, and cyl variables.

Solutions: Refer to page 210


**Solutions for Assessment**

**Choose the appropriate options**

1) C

2) A

3) D

4) A

5) A



**Programming Assignment Solution**

Task 1)

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.418.png)

Task 2)

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.419.png)


Task 3)

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.420.png)

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.421.png)

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.422.png)















