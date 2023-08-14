
<img align="right" src="./logo.png">

#### Lab  2: INTRODUCTION TO STATISTICS


**AIM**

The aim of the following section is to perform various exercises by writing python code so that we can practice hands-on descriptive statistics.

The labs for this lab include the following exercises.

- Introduction to Python and Python versions
- Introduction to Anaconda Python Distribution and Installation
- What are Jupyter notebooks? 
- How does notebook work?
- Installing and launch the Jupyter Notebook.
- Importing necessary packages.
- Calculating the Central Tendencies.
  - Mean
  - Median
  - Mode
- Calculating Measure of dispersion
  - Standard Deviation
  - Variance
  - Range
  - Percentile
  - Quantiles
  - Boxplot
  - Skewness

We will be working with python3 and Jupyter IDE in this lab.

#### Pre-reqs:
- Google Chrome (Recommended)

#### Lab Environment
Notebooks are ready to run. All packages have been installed. There is no requirement for any setup.

All examples are present in `~/work/machine-learning-essentials-module1/lab_02` folder. 

#### JupyterLab

- Now we have launched our new notebook, we can go ahead with implementation.

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.048.png)

**Importing the necessary packages**

We always need to import the necessary packages required for the project or task.

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.049.png)

**Calculating Central Tendencies** 

**Task 1: Mean, Median, and Mode**

We will be using the data about "systolic blood pressure" and "blood cholesterol levels"  to calculate central tendencies.

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.050.png)

We first created BP (Blood Pressure) and plasma Cholestrol variables, then using numpy and scipy.stats packages we calculated the central tendencies (*i.e.* Mean, Median, Mode).

**Calculating Measure of Dispersion**

We will be using a dataset which contains the height and weight of 10 students.

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.051.png)

We have created a list of objects to store the data about height and weight.

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.052.png)

We have created Pandas Data Frame using the dictionary variable as input:



**Task 2: Standard Deviation**

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.053.png)

Standard deviation in the height variable is less than the standard deviation in weight variable. It means that the data points in the height variable are closer to mean, when compared to the data points in weight variable.

**Task 3: Variation**

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.054.png)

Variation in height variable is less compared to variation in weight variable.



**Task 4: Range and Percentile** 

Range

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.055.png)

Percentile

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.056.png)


**Task 5: Quartiles**

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.057.png)

25% of the students have the body weight less than 145.5 pounds, 50% of the students have the body weight less than 158 pounds, 75% of students have the body weight less than 164.25 and finally, 50% of students weight lies in the range of 145.5 – 164.25 (i.e. Interquartile range = 18.75).


**Task 6: Boxplot**

Boxplot is also called whisker plot. It is a robust way to identify extreme values (outliers). Boxplot contains upper whisker (*i.e.* Q3+IQR\*1.5) and lower whisker (*i.e.* Q1-1.5IQR). The data points which fall beyond those limits are considered as outlier.

Extreme Low Value

Extreme High Value
![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.058.png)![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.059.png)![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.060.png) ![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.061.png)

The above plots show an outlier in each of the height and weight variables. In the height variable, there is an extremely high value and in the weight variable, there is an extremely low value.






**Task 7: Skewness**

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.062.png)

We have generated 100 random values using np.random.randn (random normal distribution) function in numpy package and have plotted a distribution plot.  We will be further discussing normal distribution in future labs.





Negatively skewed distribution 

Outliers

Extreme Low Values
![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.063.png)![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.064.png)![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.065.png)![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.066.png)![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.067.png)












Positively skewed distribution

Outliers

Extreme High Values
![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.068.png)![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.069.png)![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.070.png)![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.071.png)![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.072.png)

The distribution of the data will be affected due to a few extreme small (or) large data points called the *outliers*. We can see in the above examples that the symmetric distribution has been completely screwed with the inclusion of few outliers into the dataset. 

**Case Study**

We start our first lab with a relatively easy problem about the reduction in the birth weights.

**Data description**

In 2015, 20.5 million newborns, an estimated 14.6 percent of all babies born globally that year suffered from the low-birth weight problem. These babies were more likely to die during their first month of life and those who survived faced lifelong complications of low birth weight; including a higher risk of stunted growth, lower IQ, and adult-onset chronic conditions, such as obesity and diabetes.

The dataset contains information about new born babies and their parents. The birth weights of the babies whose mothers smoked have been adjusted slightly to exaggerate the difference between mothers who smoked and the ones who did not smoke. 

In this lab, we will be exploring the data by using the descriptive statistics techniques, and then draw some meaningful insights from the data.

**Note**

The original dataset is available at the following link: 

<https://www.sheffield.ac.uk/polopoly_fs/1.937185!/file/Birthweight_reduced_kg_R.csv>

**general description of the variables**

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.073.png)

**Understanding the Data**

The main goal of the presented steps is to acquire the basic knowledge about the data, how its various features are distributed, and whether there are any missing volumes in it.

Please import relevant python libraries and data itself for the analysis.

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.074.png)

A good starting practice is to check the size of the data we are loading, the number of missing values of each column, and explore the top 5 observations of the dataset:

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.075.png)

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.076.png)

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.077.png)




**Descriptive Statistics**

In order to get simple statistics for the numerical columns, such as the mean, standard deviation, minimum and maximum values, and their percentiles, we can utilize the **describe** function on a **pandas dataset** object

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.078.png)

"*Birthweight*" is our dependent variable and the rest of the variables are independent. We want to analyze how the independent variables affect the *Birthweight* variable.

The mean birth weight is 3.312857 and the standard deviation is 0.603895. The maximum and minimum birth weights are 4.57 and 1.92, respectively.

We can analyze the distribution of the birth weight variable.

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.079.png)

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.080.png)

As you can see that "*Birthweight*" variable is symmetrically distributed around the mean.

Let us analyze the distribution of "**mnocig"** (Number of cigarettes smoked per day by mother) variable.

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.081.png)

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.082.png)

We can clearly observe that the data is positively or rightly skewed. Boxplot helps us to find out the presence of outliers in "*mnocig*" variable.

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.083.png)

We can justify our assumption of outlier with the help of *boxplot*. One important insight we have to note here is that the mean of "*mnocig*" variable is 9.428571 and the median is 4.5. 

Mean is always influenced by an outlier but median is not. We can conclude that mean is not the best descriptive statistic in the presence of outliers in the data.

Thus, 50% of the mothers smoke less than 5 cigarettes per day, and the other 50% of the mothers smoke more than 5 cigarettes per day. To get more precise result, we can go with percentiles.

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.084.png) 

This means that 46% of mothers do not smoke cigarettes, 29% of mothers smoke less than or equal to 16 cigarettes per day, 15% of mothers smoke less than or equal to 25 cigarettes per day and finally, only 10% of mothers smoke more than 25 cigarettes per day.

Now it is time to understand the correlation between the dependent and independent variable.

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.085.png)

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.086.png)

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.087.png)

There is a very weak negative correlation between "*mnocig*" and "*Birthweight*". We can understand that ‘mnocig’ does not play an important role in determining the "Birthweight" of the child based on the available data.








**Assessment**

**Choose the appropriate option**

1) **Which of the following are the contents of descriptive statistics?**

A. Mean
B. Median
C. Percentile
D. Standard Deviation
E. All of the above

2) **Select an appropriate skewness of the curve From the below image.**

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.088.png)

A. Positively Skewed
B. Negatively skewed
C. Symmetric

3) **From the below image select the appropriate relationship between the 2 variables:**

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.089.png)

A. Negative Relationship
B. Positive Relationship
C. No Relationship



4) **The value of Correlation between two continuous variables lies between:**

A. 1 to 100
B. 0 to  1
C. -∞ to +∞
D. -1 to 1


5) **Which of the following statements are true?**

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.090.png)

A. All of the students are less than 17 years old		
B. Atleast75% of the students are 10 years old or older
C. There is only one 16 year old at the party
D. The youngest kid is 7 years old
E. Exactly half the kids are older than 13


**Fill in the spaces with appropriate answers**

1) The method used to measure the spread of the data is \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_.?
2) \_\_\_\_\_\_\_\_\_\_\_\_\_\_ is the robust visualization used to identify the outliers in the data.
3) Mean=Median=Mode in \_\_\_\_\_\_\_\_\_\_\_\_ distribution.
4) 4 different data measurement scales are\_\_\_\_\_\_\_\_\_\_\_, \_\_\_\_\_\_\_\_\_\_,   \_\_\_\_\_\_\_\_\_\_\_, \_\_\_\_\_\_\_\_\_\_.
5) The 50th Percentile is also known as \_\_\_\_\_\_\_\_\_\_.

**Programming Assignment** 

Using the data in the below URL, 

` `<https://www.sheffield.ac.uk/polopoly_fs/1.937185!/file/Birthweight_reduced_kg_R.csv>

By referring to the code used in the case study, perform the following tasks:

1) Find the distribution of **fnocig** (Number of cigarettes smoked per day by father) variable.

2) Find the correlation between **Birthweight** and **fnocig** variables.





**Solutions for Assessment**

**Choose the appropriate options answers**

1) E
2) C
3) A
4) D
5) A)True B) False C) True D) True and E)True



**Fill in the spaces with appropriate answers:**

1) Measure of Variability (Spread / Dispersion)
2) Boxplot
3) Symmetric
4) Nominal, Ordinal, Interval, and Ratio
5) Median

**Programming Assignment Solution**

Task 1)

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.091.png)

Task 2) 

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.092.png)


