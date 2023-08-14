
<img align="right" src="./logo.png">

**Lab  4 – INFERENTIAL STATISTICS**


**AIM**

The aim of the following lab exercises is to perform partial implementation of the Normal distribution, Standard Normal distribution, Confidence Intervals, and Test of Hypothesis by writing python code so that we get hands-on practice of Inferential Statistics.

The labs for this lab include the following exercises.

- Calculating the probability using standard normal distribution.
- Calculating the confidence intervals.
- Performing test of hypothesis.

We will be working with Python3 and jupyter notebook IDE.


#### Pre-reqs:
- Google Chrome (Recommended)

#### Lab Environment
Notebooks are ready to run. All packages have been installed. There is no requirement for any setup.

All examples are present in `~/work/machine-learning-essentials-module1/lab_04` folder. 


**Importing the necessary packages**

We always need to import the necessary packages required for the project or task.

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.245.png)

**Calculating probabilities using normal** and **Standard normal distributions**

**Task 1:** Generate the data that follows the normal distribution and visualize the distribution.

Step 1: We are generating 10000 data points, which follow the normal distribution with mean = 0 and standard deviation = 1.

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.246.png)

The function "np.random.randn" samples 10000 data points from the normal distribution which has mean = 0 and standard deviation = 1.

Step 2: Visualizing the distribution of the data generated.

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.247.png)

- In line 1, we set the size of the figure to be (10, 5).
- In line 2, we plotted a histogram with the data we generated.
- Line 3 – 5, we formatted the figure with titles, x-label and y-label.

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.248.png)

We can clearly see that the histogram is bell-shaped with a mean value of = 0.

**Task 2:** Generate a normal data set with mean = 65 and standard deviation = 3, and visualize the distribution.

Step 1: In the previous task, we generated a normal data set with default mean and standard deviation values. Here in this task, we will generate a normal data set with mean = 65 and standard deviation = 3.

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.249.png)

The function "np.random.normal" helps you to generate normal data set with custom mean and standard deviation values. The function takes three important parameters into consideration:

- loc: The location where the mean of data needs to be.
- scale: It indicates the standard deviation of the data.
- size: The size parameter tells, how many data points to be sampled from the normal distribution.

Step 2: Visualizing the distribution

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.250.png)

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.251.png)

We can clearly see that the data is normally distributed with mean = 65.

**Task 3:** Students’ scores in 1000 exams are normally distributed with mean values of 65 and a standard deviation of 9. Find the percent of the scores, which are:

1. less than 55.
1. at least 85.
1. between 70 and 80.

Step 1: Generate a score of 1000 exams, which follows the normal distribution with mean = 65 and standard deviation of 9.

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.252.png)

Step 2: Visualize the distribution,

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.253.png)

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.254.png)

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.255.png)

The scores of 1000 exams are normally distributed with mean = 65 and standard deviation = 9.

**Task 3:** Calculate the following probabilities:

1. What is the probability of getting an exam score less than 55, P (X < 55)?

Step 1: Calculate the z-score for x = 55.

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.256.png)

Step 2: Calculate the area under the curve,

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.257.png)

Step 3: Visualize and interpret the results,

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.258.png)

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.259.png)

` `Python calculates the left / lower-tail probabilities by default.

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.260.png)

For this curve z = -1.11 and the area under left is 0.13. Therefore, the probability of getting an exam score less than 55 is 0.13; P (X < 55) is 13%.



1. What is the probability that the exam score will be at least 85; P(X ≥ 85)?

Step 1: Calculate the z-score for x = 85.

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.261.png)

Step 2: Calculate the area under the curve.

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.262.png)

Step 3: Visualize and interpret the results.

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.263.png)

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.264.png)

` `Python calculates the left / lower-tail probabilities by default.

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.265.png)

For this curve z = 2.22 and the area under right is 0.01. Therefore, the probability of getting an exam score greater than 85 is 0.01; P (X < 85) is 1%.



1. What is the probability that the exam score will be between 70 and 80; P (70 < X < 80)?

Step 1: Calculate the z-score for x = 70 and x = 80.

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.266.png)


Step 2: Calculate the area under the curve,

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.267.png)

Step 3: Visualize and interpret the results.

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.268.png)

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.269.png)

` `Python calculates the left / lower-tail probabilities by default.

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.270.png)

For this curve, z values are 0.56 and 1.67. Therefore, the probability of getting an exam score between 70 and 80 is 0.24; P (70 < X < 80) is 24%.

**Calculating the confidence intervals**

Computing the confidence interval for a population mean

**Task 4:**  A survey was conducted of the US companies that do business with many firms in India.  One of the survey questions was: Approximately how many years have your company been trading with the firms in India?  

A random sample of 44 responses to this question yielded a mean value of 10.455 years.  Suppose the standard deviation for this population is 7.7 years. Use this information to construct a 90% confidence interval for the mean number of years that a company has been trading in India, out of the population of US companies trading with firms in India.





Step 1: Initialize the variables required for calculating the confidence interval.

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.271.png)

Step 2: compute the standard error, margin error, and confidence intervals.

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.272.png)

The analyst is 90% confident that if a census of all the US companies trading with the firms in India were taken at the time of the survey, the actual population mean number of trading years would be between 8.45 and 12.36 years.

Calculate the confidence interval for a population mean using t-distribution.

**Task 5:** The labeled potency of a medicinal tablet is 100mg. As per the quality control specifications, 10 tablets are randomly assayed for potency testing.

A researcher wants to estimate the interval for the true mean of the batch of tablets with 95% confidence. Assume that the potency is normally distributed.

Data is as follows (in mg):

|98.6|102.1|100.7|102|97|
| -: | -: | -: | -: | -: |
|103.4|98.9|101.6|102.9|105.2|

Step 1: Initialize the variables required for calculating the confidence interval.

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.273.png)

Step 2: Calculate the standard error and margin error.

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.274.png)

Step 3: Calculate the confidence interval and interpret it.

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.275.png)

The mean for the batch is 101.24mg with an error of +/-1.67mg. The researcher is 95% confident that the average potency of the batch of tablets is between 99.57mg and 102.91mg.

Calculate the confident interval for a population proportion.

**Task 6:** Let’s suppose that on a certain website, out of 1500 visitors on a given day, 450 clicked on an ad purchased by a sponsor. Let’s construct a confidence interval for the population proportion of visitors who clicked on the ad.

Step 1: initialize the variables.

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.276.png)

We have initialized the sample proportion and number of sample variables.

Step 2: Calculate the standard error and margin error.

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.277.png)

Step 3: Compute the confidence interval for the population proportion.

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.278.png)

We are 95% confident that the population proportion falls between 0.28 and 0.32.

Calculate the confidence interval for variance.

**Task 7:** You randomly select and weigh 30 samples of an anti-allergy medicine. The sample standard deviation is 1.20 milligrams. Assuming that the weights are normally distributed, construct the 99% confidence interval for the population variance.

Step 1: initialize the variables.

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.279.png)

Step 2: Calculating the confidence intervals.

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.280.png)

We are 99% confidence that our population variance falls between 0.7979 and 3.1826.

**Performing Test of Hypothesis**

One sample t-Test

**Task 8:** Suppose a botanist wants to know if the mean height of a certain species of plant is equal to 15 inches. She collects a random sample of 12 plants and records their individual heights in inches.

Use the following steps to conduct a one sample t-test to determine if the mean height for this species of plant is equal to 15 inches.

Step 1: Create the data.

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.281.png)

Step 2: Conduct a one sample t-test.

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.282.png)

We used the "ttest\_1samp()" function from the "scipy.stats" package to conduct a one sample t-test, which uses the following syntax: 

**ttest\_1samp(a, popmean)**

Where:

- a: an array of sample observations.
- popmean: the expected population mean.

The t-test statistic is -1.6848 and the corresponding two-sided p-value is 0.1201.

The two hypotheses for the one sample t-test are as follows:

H0: µ = 15 (The mean height for this species of plant is 15 inches).

H1: µ ≠ 15 (The mean height is not 15 inches).

Step 3: Interpret the results.

Because the p-value of our test (0.1201) is greater than alpha = 0.05, we fail to reject the null hypothesis. We do not have sufficient evidence to say that the mean height for this specie of plant is different than 15 inches.

Two sample t-test

**Task 9:** Researchers want to know whether two different species of plants have the same mean height. To test this, they collect a simple random sample of 20 plants from each specie.

Use the following step to conduct a two-sample t-test to evaluate if any two species of plants have the same height.

Step 1: Create the data

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.283.png)

Step 2: Conduct a two-sample t-test.

Before we perform the test, we need to decide if the two populations have equal variances or not. As a rule of thumb, we can assume that the populations have equal variance if the ratio of the larger sample variance to the smaller sample variance is less than 4:1.

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.019.png)

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.284.png)

The ratio of the larger sample variance to the smaller sample variance is 1.586, which is less than 4. Thus, we can assume that the population variances are equal.

Next, we proceed to perform the two-sample t-test with equal variances:

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.285.png)

We have used the "ttest\_ind()" function from scipy.stats library to conduct a two-sample t-test, which uses the following syntax:

**ttest\_ind(a, b, equal\_var=True)**

Where:

- a: an array of the sample observations for group 1.
- b: an array of the sample observations for group 2.
- equal\_var: if True, perform a standard independent two sample t-test that assumes equal population variances. If False, perform "Welch’s t-test", which does not assume the equal population variances. This is True by default.

The t-test statistic is -0.6337 and the corresponding two-sided p-value is 0.53005.

Step 3: Interpret the results.

The two hypotheses for this two sample t-test are as follows:

H0: µ1 = µ2 (The mean values for the two populations are equal).

H1: µ1 ≠ µ2 (The mean values for the two populations are not equal).

Because the p-value of our test (0.53005) is greater than alpha = 0.05, we fail to reject the null hypothesis. Thus, we do not have sufficient evidence to say that the mean height of plants between the two populations is different.

Paired sample t-test 

**Task 10**: Suppose we want to know whether a certain study program significantly impacts the student performance in an exam. To test this, 15 students in a class take a pre-test. Then, the students participate in the study program for two weeks. Following this, the students retake a test of similar difficulty.

To compare the difference between the mean scores on the first and second test, we use a paired samples t-test. This is because the first test score can be paired with the second test score for each student.

Perform the following steps to conduct a paired sample t-test.

Step 1: Create the data.

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.286.png)

Step 2: Conduct a paired samples t-test.

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.287.png)

We used "ttest\_rel()" function from the ‘scipy.stats’ library to conduct a paired samples t-test the following syntax:

**ttest\_rel(a, b)**

where: 

- a: an array of the sample observations from Group 1.
- b: an array of the sample observations from Group 2.

The test statistic is -2.9732 and the corresponding two-sided p-value is 0.0101.

Step 3: Interpret the results.

In this example, the paired samples t-test uses the following Null and Alternative Hypotheses:

H0: µ1 = µ2 (The mean pre-test and post-test scores are equal).

H1: µ1 ≠ µ2 (The mean pre-test and post-test scores are not equal).

Since the p-value (0.0101) is less than 0.05, we can reject the Null Hypothesis. Consequently, we have now sufficient evidence to say that the true scores are different for the students before and after participating in the study program.

**Case Study**

We will be continuing the analysis of dataset labeled as "**birthweight reduced**". Interestingly, we did not find anystrong and meaningful correlation between "Birthweight" and "mnocig" variables (Lab I: Page 39). Such a conclusion cannot be based solely on the plot observations but has to be backed by the statistical tests.

**Hypothesis Tests**

**Task 1:** **Estimating Average Birth Weight**

Step 1: Start with computing the average birthweight. Note that this value will serve in formulating the *null* hypothesis because, here you explicitly compute the population statistic-or the average birth weight. In most cases, such quantities are not directly observable and, in general, only the estimation for the population statistics is applied:

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.288.png)

Step 2: We randomly sample the data and compute the sample mean.

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.289.png)

Step 3: Now we perform the one sample t-test to check if there is any significant difference between the sample and population mean.

H0: No difference between population and sample mean.

H1: There is a significant difference between population and sample mean.

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.290.png)

The computed p-value is equal to 0.168, which is much larger than the critical 0.05, and therefore, we cannot reject the null hypothesis.

**Task 2:** Compare the means of two independent groups.

Quite often, when performing statistical tests, we want to apply certain statistical tests on two different groups (for example, the average weight between men and women) and estimate whether there is a statistically-significant difference between the values obtained in the two groups. Let’s denote them with µ1 and µ2.

In this exercise, we will be testing the hypothesis on the "*Birthweight*" variable by grouping the data into smokers and non-smokers.

Step 1: We need to group the "*Birthweight*" data based on "smoker" (0: non-smoker, 1: smoker) variable.

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.291.png)

Step 2: Perform the two sample t-test to compare the mean.

H0: Average birth weight of smoker’s group – Average birth weight of non-smoker’s group = 0

H1: Average birth weight of smoker’s group – Average birth weight of non-smoker’s group ≠ 0

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.292.png)

The resulting p-value from this test is less than 0.05. As a conclusion, we can reject the *null* hypothesis and confirm that our initial observation (*i.e.,* Smoking status of the mother has no effect on the birth weight of the child) is wrong. There is a statistically significant difference between the birth weight of children whose mothers smoke, when compared to the children whose mothers do not smoke.














**Assessment**

**Choose the appropriate option**

1) **Following IQ scores are approximately normally distributed with a mean of 100 and standard deviation of 15. The proportion of people with IQs above 130 is:**

A. 95%

B. 68%

C. 5%

D. 2.5%

2) **Failing to reject the null hypothesis when it is false is:**

A. Alpha

B. Type I error

C. Beta

D. Type II error

3) **A statistic is:**

A. A sample characteristic

B. A population characteristic

C. Unknown

D. Normally distributed


4) **We reject the H0 Hypothesis when:**

A. P-value < alpha

B. P-value > alpha

C. P-value = alpha

5) **People commonly lie when asked questions about personal hygiene.. This is an example of:**

A. Sampling bias

B. Confounding

C. Non-response bias

D. Response bias

**Fill in the spaces with appropriate answers**

1) Because of the possibility of error in sampling from populations, researchers use \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_ statements to describe results.

2) A t-test is used to compare \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_.

3) The two forms of two sample t-tests are \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_ and \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_.

4) \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_ percent of the data will fall in two deviations in standard normal distribution.

5) An estimate of a population parameter given by two numbers between which the parameter would be expected to lie is called a \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_ of the parameter.

**Programming Assignment** 

Using the data in the below URL, 

` `<https://www.sheffield.ac.uk/polopoly_fs/1.937185!/file/Birthweight_reduced_kg_R.csv>

By referring to the code used in the case study perform the following tasks.

1) Find the correlation between "**Gestation**"** and "**Birthweight**" variables.

2) Compute two sample t-test on the "**Gestation**" variable by grouping the data on "**smoker**".



**Solutions:** Refer to page 69





**Solutions for Assessment**

**Choose the appropriate options**

1) D

2) D

3) A

4) A

5) D



**Fill in the spaces with appropriate answers**

1) Probability

2) Two means

3) Independent and dependent t-test

4) 95%

5) Interval Estimate

**Programming Assignment Solution**

Task 1)

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.293.png)

Task 2)

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.294.png)




