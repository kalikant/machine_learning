
<img align="right" src="./logo.png">


**Lab  3: PROBABILITY DISTRIBUTIONS**



**AIM**

The aim of the following lab exercises is to perform the partial implementation of the Binomial and Poisson distributions by writing python code, so that we can get hands-on practice of Discrete Probability distributions

The labs for this lab include the following exercises:

- Calculating probability using the Binomial distribution.
- Calculating probability using the Poisson distribution.

We will be working with python3 and the jupyter notebook IDE.

#### Pre-reqs:
- Google Chrome (Recommended)

#### Lab Environment
Notebooks are ready to run. All packages have been installed. There is no requirement for any setup.

All examples are present in `~/work/machine-learning-essentials-module1/lab_03` folder. 


**Importing the necessary packages**

We always need to import the necessary packages required for the project or task.

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.131.png)

**Calculating probability using binomial distribution**

**Task 1:** Hospital records show that among the patients with a certain disease, `75%` die of it. What is the probability that of six randomly selected Patients, four will recover from the disease?

**Step 1**: We need to initiate the n, x, p, and q values.


![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.132.png)


**Step 2:** Now, let’s use binom.pmf (binomial probability mass function) to calculate the probability that four out of six randomly selected patients will recover.


![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.133.png)

**Step 3:** Let’s see how the probabilities are distributed among the recovered patients.

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.134.png)

Created a dictionary to hold "*number of patients recovered*" values, we used ‘range’ function to generate the values between 0 and 6.

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.135.png)

We created the data frame using pandas, DataFrame function.

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.136.png)

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.137.png)

We defined a lambda function and named it as "prob". "*A lambda function is a small anonymous function, which can take any number of arguments, but can only have one expression*".

The objective of "prob" function is to calculate the probabilities, which use binomial probability mass function (binom.pmf).

We calculated probability for each value of "recovery" variable by passing the value to "prob" function and saved the result to a new variable called "probability".

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.138.png)

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.139.png)![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.140.png)![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.141.png)

Interpretation: The probability of recovery of four patients among six randomly selected patients is 3.29%. Similarly, probability of recovery of one patient among six randomly selected patients is 35.59% and finally, probability of recovery of all the six patients is almost 0%.

**Task 2:** In the old days, there was a probability of 0.8  to acquire success in any attempt to make a telephone call. Calculate the probability of having 7 successes (successful calls) in 10 attempts.

Solution: 

Probability of success p=0.8, so q = 0.2 

X = Success in getting through.

Probability of 7 successes in 10 attempts:

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.142.png)

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.143.png)

Therefore, there is 20% probability of getting 7 successes in 10 attempts to make a call.

**Task 3:** A company drills 9 wildcat oil exploration wells, each with an estimated probability of success of 0.1. What is the probability that all nine wells fail?

Solution:

Probability of success p = 0.1, and failure is q = 0.9

n = 9

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.144.png)

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.133.png)

There is 38.74% probability that all the nine exploratory drillings will fail.



**Calculating the probability using Poisson distribution**

**Task 4:** John is recording birds in a national park, using a microphone placed in a tree. He is counting the number of times a bird is recorded singing and wants to model the number of birds signing in a minute. For this task, he’ll assume independent of the detected birds.

Looking at the data of the last few hours, John observes that on the average, 3 birds are detected in an interval of one minute. So, the value 3 could be a good candidate for the parameter of the distribution λ. His goal is to know the probability that a specific number of birds will sing in the next minute.

For instance, the probability of John observing 4 birds singing in the next minute would be,

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.145.png)

The probability of John observing 4 birds singing in the next minute is around 16.8%

Remember, the function poisson.pmf(k, lambda) takes the value of k and λ and returns the probability to observe k occurrences (*i.e.,* to record k birds singing). 

Let’s plot the distribution for the various values of k:

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.146.png)

We created a "lambd" variable to hold average birds recorded in a minute, "k\_values" stores various values of k. Finally, we created distribution variable to store probability values.

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.147.png)

We have passed the various k values and a lambda value to poisson.pmf though loop and saved the probabilities into the distribution variable.

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.148.png)

The probabilities corresponding to the values of k are summarized in the probability mass function shown in the above figure.

Interpretation: As shown, it is most likely that John will hear two to three birds singing in the next minute.

**Task 5:** If electrical power failures occur according to the Poisson distribution with an average of ‘3’ failures every twenty weeks, calculate the probability of occurrence of 1 failure during a particular week.

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.149.png)

In problem statement, λ value was given for 20 weeks. We calculated the average λ value per week as follows:

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.150.png)

Therefore, the probability of 1 electrical power failure occurring in a week is 12.91%.


Case Study

John is recording birds’ sounds in a national park, using a microphone placed in a tree. He is counting the number of times a bird is recorded singing, and wants to model the number of birds singing per minute. For this task, he will assume the independence of the detected birds.

Looking at the data of the last few hours, John observes that on average,  3 birds are detected making sounds in an interval of one minute. Thus, the value of 3 could be a good candidate for the parameter of the distribution λ. His goal is to determine the probability that a specific number of birds will sing in the next minute.

For instance, the probability of John observing 4 birds singing in the next minute would be:

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.145.png)

The probability of John observing 4 birds singing in next minute is around 16.8%.

Remember that the function *poisson.pmf* (k, lambda) takes the value of k and λ and returns the probability to observe k occurrences (*i.e.*, to record k birds singing). 

Let’s plot the distribution for various values of k:

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.146.png)

We created a "*lambd*" variable to hold the average birds’ sounds recorded in a minute, "*k\_values*" stores various values of k and finally, we created the distribution variable to store probability values.

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.151.png)

We have passed various k values and a lambda value to *poisson.pmf* though loop and saved the probabilities into the distribution variables.

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.152.png)

The probabilities corresponding to the values of k are summarized in the probability mass function shown in the figure.

Interpretation: You can see that it is most probable that John will record two or three birds singing in the next minute.

**Assessment**

**Choose the appropriate option**

1) Which of the following statements best describes the expected value of a discrete random variable?

A. It is the geometric average of all possible outcomes.

B. It is the weighted average over all possible outcomes.

C. It is the simple average of all possible outcomes.

D. None of the above.

2) Which of the following is not a true statement about the binomial probability distribution?

A. Each outcome is independent of each other.

B. Each outcome can be classified as either success or failure.

C. The probability of success must be constant from trail to trail.

D. The random variable of interest is continuous.

3) If n=10 and p = 0.8, then the mean of the binomial distribution is?

A. 0.08

B. 1.26

C. 1.60

D. 8.00

4) If the outcomes of a discrete random variable follow a Poisson distribution, then their

A. Mean equals the variance.

B. Mean equals the standard deviation.

C. Median equals the variance.

D. Median equals the standard deviation.

5) The sum of the product of each value of a discrete random variable X times its probability is referred to as its

A. Expected value.

B. Variance.

C. Mean.

D. Both (a) and (c)

**Fill in the spaces with appropriate answers**

1) The \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_ distribution can be used to approximate the binomial distribution when the number of trails is large and the probability of success is small (np <=7).

2) If n = 10 and p = 0.8, then the standard deviation of the binomial distribution is \_\_\_\_\_\_\_\_\_\_\_\_\_.


3) If two events A and B are mutually exclusive then P (A∩B) = \_\_\_\_\_\_\_\_\_\_\_\_.

4) A coin is tossed up 4 times. The probability that tails turn up in 3 cases is \_\_\_\_\_\_\_\_\_\_.

5) Mutually Exclusive events \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_.

**Programming Assignment** 

Assignment 1

Case study

At a Biscuit factory in Slough with 120 production workers, there is a 10% chance that a worker is absent on a given day. The probability that one worker is absent, assumes that it will not affect the probability that another worker is absent. The factory can operate on any given day as long as no more than 50 workers are absent on that day. What is the probability that any 2 out of 9 randomly chosen workers will be absent on the next Monday?

Solution:

This situation can be described by a Binomial Distribution since we have:

- A fixed number (9) of trails (workers).
- Each trial has two possible outcomes, "success" (the worker is absent) or "failure" (the work is not absent).
- The probability of success (0.1) is constant.
- The outcome of each trial is independent of the outcome of all the other trials.

Using the given values, calculate the probability that any 2 out of 9 randomly-chosen workers will be absent P (X =2)?

Assignment 2

**Case study** 

The average daily sales volume of 60-inch 4K HD TVs at XYZ Electronics is five. Calculate the probability of XYZ Electronics selling nine TV sets today.

**Solution**

We have 

- λ = 5, since five 60-inch TVs in the daily sales average.
- X = 9, we want to solve for the probability of nine TVs being sold.

Using Poisson distribution find the probability, P (X = 9).

**Solutions:** Refer to page 31













**Solutions for Assessment**

**Choose the appropriate options answers**

1) A

2) A

3) D

4) A

5) C

**Fill in the spaces with appropriate answers**

1) Poisson

2) 1.26

3) 0

4) ½

5) Does not contain any common sample point

**Programming Assignment Solutions**

Task 1) 

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.153.png)

Task 2)

![](./images/da/Aspose.Words.8e0affcc-ac68-4184-94d3-69ce9332cabb.154.png)





