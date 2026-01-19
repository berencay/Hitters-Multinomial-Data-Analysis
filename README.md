# Hitters-Multinomial-Data-Analysis
Business data analysis using multinomial regression on the Hitters dataset

We R Who We R
Defne Turcan, Deniz Yakici, Mete Alper Yegengil, Beren Cay

2025-03-28

Introduction
In this project, our first aim was to define the difference between multinomial logistic regression and simple logistic regression. We worked on the dataset “Hitters” by first gathering information about the dataset’s structure, cleaning the missing values, and overall pre-processing it. Then, we created 4 different salary level categories (RookieBudget, SolidStarter, BeenThereDoneThat, TooRitchToPitch). Based on our salary levels, we set the cutoffs with relevant quartiles. With our new salary levels, to gain more information regarding the variables and their relationship with our dependent variable’s levels, we concluded an Exploratory Data Analysis using boxplots. We interpreted the results from our analysis, and determined which variables had the strongest predictive relationship and reported them. We selected the most significant variables and ran our model. While running our model, as we were required to do so, we selected one group member who had the closest birthday to 21 August, and seeded it accordingly. Our model generated LOOCV as a first method for cross-validation. Therefore, we performed LOOCV, and later we determined our error measure. In relation with our dataset, variables, model choices, we chose accuracy as an error measurement. Since we were a group of 4, we performed a second cross-validation. We wanted to continue with the closest birthday method, and selected the second group member who was closest to 21 August. In accordance with the second birthday, our model generated another cross-validation method, being 5-Fold. Lastly, we reduced the model, and tried to create a better model that is performing better than our full model. Then, we concluded both cross-validation methods again to check if our test error has decreased.

Simple Logistic Regression vs. Multinomial Logistic Regression and Formalization
To formalize the model in a generic way, we can use this formula to see an overview of how multinomial logistic regression works.

log(𝑃(𝑌=𝑘)𝑃(𝑌=baseline))=𝛽0𝑘+𝛽1𝑘𝑋1+𝛽2𝑘𝑋2+⋯+𝛽𝑝𝑘𝑋𝑝

where 𝑋1,𝑋2,…,𝑋𝑝 represent predictor variables, and 𝛽0𝑘,𝛽1𝑘,…,𝛽𝑝𝑘 are the coefficients associated with each predictor for category 𝑘.

In comparison, simple logistic regression models the probability of a binary outcome with two classes and has the following form:

log(𝑃(𝑌=1)𝑃(𝑌=0))=𝛽0+𝛽1𝑋1+𝛽2𝑋2+⋯+𝛽𝑝𝑋𝑝

So generally, as it is also written in our book ISLR, we studied that a simple logistic regression model uses variables to predict binary outcomes (0/1, yes/no). In general, it can be said that it estimates the probability of one of the two outcomes by using a logistic function. On the other hand, multinomial logistic regression is used when our dependent variable has more than two classes which are not ordered. It estimates the probability of each category in relation to a baseline/reference category.

Necessary Libraries
We decided to use tidyverse, since it is necessary for data manipulation and cleaning which are essential parts for pre-processing. We used the nnet library to fit our multinomial logistic regression model. Moreover, as we have been learning in our lab sessions, we used the ggplot library to conclude our Exploratory Data Analysis. Then, we used the caret library to perform our cross-validation as well as the evaluation of our model’s error measure. Lastly, we used MASS library for the creation of our better model, which helped us to use stepAIC() function, which was essential for using backward selection.

library(tidyverse)
library(ggplot2)
library(nnet)
library(caret)
library(MASS)
Pre-processing
In this section, we performed a thorough pre-processing on our dataset, Hitters. We divided our pre-processing into three main sections: understanding the dataset structure, identifying and cleaning missing values,and determining whether to standardize, and removing the outliers are needed. Below, we have written all the necessary functions that we used in pre-processing, and what information we received from them.

1- Dataset Structure

str(): We used it this function to examine the structure of our dataset. We received an overview of the number of observations (317), the number of variables (20), and the data types of each column.
summary(): With this function, We looked at the descriptive statistics for each numeric variable, mainly the minimum, 1st quartile, median, mean, 3rd quartile, and maximum values.
dim(): This function helped us confirm that our dataset includes 317 rows and 20 columns. This information gave us a baseline for values before and after cleaning.
2- Identifying and Addressing the Missing Values

After understanding the structure of our dataset in the first step, we proceeded with identifying any missing values, and cleaning them.

colSums(is.na(Data)): We identified all the missing values in our dataset. Based on our results, we detected only the Salary column contained exactly 58 missing values (NAs), and all the other variables are complete.
na.omit(): As our classification model which is predicting SalaryLevel depends on Salary, the missing values that we found would have stopped us from correctly binning our players into our defined salary levels. Therefore, we used this function to remove all rows that had missing values from salary.
cleaned_n <- nrow(Data): When we removed all the missing values from our dataset, this reduced our total number of 317 observations to 259. To save the remaining rows that are complete, we used this function.
3- Standardization and Outliers

Through our analysis of the dataset, and our model, we determined that standardization was not necessary. Since Multinomial Logistic Regression is not a scale-sensitive model, it focuses on log-odds estimation rather than distances. We thought that the model would still efficiently work without the need of scaling. However, we acknowledged that the variables in our dataset, have different ranges from each other, which might affect the interpretability of them.

In our dataset, we did not remove the outliers in the beginning, to see if some variables have outliers that indicate high or low-performing players. For example, there might be a case where a player with an exceptionally high number of hits may appear as an outlier statistically, but, in fact, they are not data errors, they are key influencers in salary determination.

Data <- read.csv("Hitters.csv")
str(Data)
```r
Data <- read.csv("Hitters.csv")
str(Data)

## 'data.frame':    317 obs. of  20 variables:
##  $ AtBat    : int  293 315 479 496 321 594 185 298 323 574 ...
##  $ Hits     : int  66 81 130 141 87 169 37 73 81 159 ...
##  $ HmRun    : int  1 7 18 20 10 4 1 0 6 21 ...
##  $ Runs     : int  30 24 66 65 39 74 23 24 26 107 ...
##  $ RBI      : int  29 38 72 78 42 51 8 24 32 75 ...
##  $ Walks    : int  14 39 76 37 30 35 21 7 8 59 ...
##  $ Years    : int  1 14 3 11 2 11 2 3 2 10 ...
##  $ CAtBat   : int  293 3449 1624 5628 396 4408 214 509 341 4631 ...
##  $ CHits    : int  66 835 457 1575 101 1133 42 108 86 1300 ...
##  $ CHmRun   : int  1 69 63 225 12 19 1 0 6 90 ...
##  $ CRuns    : int  30 321 224 828 48 501 30 41 32 702 ...
##  $ CRBI     : int  29 414 266 838 46 336 9 37 34 504 ...
##  $ CWalks   : int  14 375 263 354 33 194 24 12 8 488 ...
##  $ League   : chr  "A" "N" "A" "N" ...
##  $ Division : chr  "E" "W" "W" "E" ...
##  $ PutOuts  : int  446 632 880 200 805 282 76 121 143 238 ...
##  $ Assists  : int  33 43 82 11 40 421 127 283 290 445 ...
##  $ Errors   : int  20 10 14 3 4 25 7 9 19 22 ...
##  $ Salary   : num  NA 475 480 500 91.5 ...
##  $ NewLeague: chr  "A" "N" "A" "N" ...
```r
Data <- read.csv("Hitters.csv")
str(Data)

summary(Data)
##      AtBat            Hits           HmRun            Runs       
##  Min.   : 16.0   Min.   :  1.0   Min.   : 0.00   Min.   :  0.00  
##  1st Qu.:256.0   1st Qu.: 64.0   1st Qu.: 4.00   1st Qu.: 30.00  
##  Median :379.0   Median : 96.0   Median : 8.00   Median : 48.00  
##  Mean   :381.2   Mean   :101.1   Mean   :10.75   Mean   : 50.89  
##  3rd Qu.:512.0   3rd Qu.:137.0   3rd Qu.:16.00   3rd Qu.: 69.00  
##  Max.   :687.0   Max.   :238.0   Max.   :40.00   Max.   :130.00  
##                                                                  
##       RBI             Walks            Years            CAtBat     
##  Min.   :  0.00   Min.   :  0.00   Min.   : 1.000   Min.   :   19  
##  1st Qu.: 28.00   1st Qu.: 22.00   1st Qu.: 4.000   1st Qu.:  822  
##  Median : 44.00   Median : 35.00   Median : 6.000   Median : 1928  
##  Mean   : 47.95   Mean   : 38.51   Mean   : 7.429   Mean   : 2646  
##  3rd Qu.: 63.00   3rd Qu.: 53.00   3rd Qu.:11.000   3rd Qu.: 3919  
##  Max.   :121.00   Max.   :105.00   Max.   :24.000   Max.   :14053  
##                                                                    
##      CHits            CHmRun           CRuns             CRBI       
##  Min.   :   4.0   Min.   :  0.00   Min.   :   1.0   Min.   :   0.0  
##  1st Qu.: 209.0   1st Qu.: 14.00   1st Qu.: 101.0   1st Qu.:  91.0  
##  Median : 506.0   Median : 37.00   Median : 247.0   Median : 219.0  
##  Mean   : 717.3   Mean   : 68.94   Mean   : 358.1   Mean   : 328.8  
##  3rd Qu.:1051.0   3rd Qu.: 90.00   3rd Qu.: 518.0   3rd Qu.: 421.0  
##  Max.   :4256.0   Max.   :548.00   Max.   :2165.0   Max.   :1659.0  
##                                                                     
##      CWalks        League            Division            PutOuts      
##  Min.   :   0   Length:317         Length:317         Min.   :   0.0  
##  1st Qu.:  68   Class :character   Class :character   1st Qu.: 109.0  
##  Median : 170   Mode  :character   Mode  :character   Median : 212.0  
##  Mean   : 258                                         Mean   : 290.5  
##  3rd Qu.: 337                                         3rd Qu.: 325.0  
##  Max.   :1566                                         Max.   :1378.0  
##                                                                       
##     Assists          Errors           Salary        NewLeague        
##  Min.   :  0.0   Min.   : 0.000   Min.   :  67.5   Length:317        
##  1st Qu.:  7.0   1st Qu.: 3.000   1st Qu.: 190.0   Class :character  
##  Median : 40.0   Median : 6.000   Median : 420.0   Mode  :character  
##  Mean   :107.9   Mean   : 8.091   Mean   : 532.8                     
##  3rd Qu.:166.0   3rd Qu.:11.000   3rd Qu.: 750.0                     
##  Max.   :492.0   Max.   :32.000   Max.   :2460.0                     
##                                   NA's   :58
dim(Data)
## [1] 317  20
anyDuplicated(Data)
## [1] 0
colSums(is.na(Data))
##     AtBat      Hits     HmRun      Runs       RBI     Walks     Years    CAtBat 
##         0         0         0         0         0         0         0         0 
##     CHits    CHmRun     CRuns      CRBI    CWalks    League  Division   PutOuts 
##         0         0         0         0         0         0         0         0 
##   Assists    Errors    Salary NewLeague 
##         0         0        58         0
original_n <- nrow(Data)
Data <- na.omit(Data)
cleaned_n <- nrow(Data)
Defining Categories
After cleaning our data, since Salary as a response variable was quantitative, we focused on making sure that it was a categorical variable. To make it more fun, while we were selecting the quartiles, we gave them creative names. Such as, RookieBudget, SolidStarter, BeenThereDoneThat, and TooRichToPitch. Since we divided SalaryLevel into 4 categories, we binned it based on the quartiles in 0.25 of percentages. We used the cut() function to calculate the cut-off points.

Below, is the cut-off points assigned to each category:

“RookieBudget”: bottom 25%

“SolidStarter”: 25–50%

“BeenThereDoneThat”: 50–75%

“TooRichToPitch”: top 25%

By using include.lowest = TRUE, we guaranteed that the lowest value is included in the first bin. Finally, we demonstrated a table for the count of players in each salary level category.

Data$SalaryLevel <- cut(Data$Salary,
                        breaks = quantile(Data$Salary, probs = c(0, 0.25, 0.5, 0.75, 1), na.rm = TRUE),
                        labels = c("RookieBudget", "SolidStarter", "BeenThereDoneThat", "TooRichToPitch"),
                        include.lowest = TRUE)

table(Data$SalaryLevel)
## 
##      RookieBudget      SolidStarter BeenThereDoneThat    TooRichToPitch 
##                66                64                70                59
EDA
Through the project we have concluded an exploratory data analysis for almost all variables, but we chose to report the ones that we found which had the strongest relationships with salary levels, while staying consistent with our lab sessions.

Primarily, we decided to use boxplot to visualize the strongest relationships with our salary levels. One of the reasons why we chose boxplot was due to its effectiveness in summarizing and comparing the distribution of numeric variables across all four salary levels. It allowed us to gain important statistical features in a single visual such as the median, interquartile range, and outliers.

Our other purpose of choosing boxplot was because, compared to histograms or scatterplots, it was more efficient in a classification problem like ours where we had a dependent variable which is categorical, and the independent variables being numerical.In addition, we were able to interpret how spread and variance changed across our salary levels.

By using boxplots, we were also able to analyze whether higher salary levels are aligned with stronger performances, which helped us understanding the variables that are the most predictive during our modeling.

Boxplot 1: Hits Across Salary Levels
ggplot(Data, aes(x = SalaryLevel, y = Hits, fill = SalaryLevel)) +
  geom_boxplot() +
  scale_fill_manual(values = c(
    "RookieBudget" = "lightpink",
    "SolidStarter" = "lightyellow",
    "BeenThereDoneThat" = "lightblue",
    "TooRichToPitch" = "lightgreen"
  )) +
  labs(
    title = "Hits Across Salary Levels",
    x = "Salary Level",
    y = "Hits"
  ) +
  theme_minimal() +
  theme(legend.position = "top")


Boxplot 2: PutOuts Across Salary Levels
ggplot(Data, aes(x = SalaryLevel, y = PutOuts, fill = SalaryLevel)) +
  geom_boxplot() +
  scale_fill_manual(values = c(
    "RookieBudget" = "lightpink",
    "SolidStarter" = "lightyellow",
    "BeenThereDoneThat" = "lightblue",
    "TooRichToPitch" = "lightgreen"
  )) +
  labs(title = "PutOuts Across Salary Levels", x = "Salary Level", y = "PutOuts") +
  theme_minimal() +
  theme(legend.position = "top")


Boxplot 3: CAtBat Across Salary Levels
ggplot(Data, aes(x = SalaryLevel, y = CAtBat, fill = SalaryLevel)) +
  geom_boxplot() +
  scale_fill_manual(values = c(
    "RookieBudget" = "lightpink",
    "SolidStarter" = "lightyellow",
    "BeenThereDoneThat" = "lightblue",
    "TooRichToPitch" = "lightgreen"
  )) +
  labs(title = "CAtBat Across Salary Levels", x = "Salary Level", y = "CAtBat") +
  theme_minimal() +
  theme(legend.position = "top")


Boxplot 4: AtBat Across Salary Levels
ggplot(Data, aes(x = SalaryLevel, y = AtBat, fill = SalaryLevel)) +
  geom_boxplot() +
  scale_fill_manual(values = c(
    "RookieBudget" = "lightpink",
    "SolidStarter" = "lightyellow",
    "BeenThereDoneThat" = "lightblue",
    "TooRichToPitch" = "lightgreen"
  )) +
  labs(title = "AtBat Across Salary Levels", x = "Salary Level", y = "AtBat") +
  theme_minimal() +
  theme(legend.position = "top")


Boxplot 5: Walks Across Salary Levels
ggplot(Data, aes(x = SalaryLevel, y = Walks, fill = SalaryLevel)) +
  geom_boxplot() +
  scale_fill_manual(values = c(
    "RookieBudget" = "lightpink",
    "SolidStarter" = "lightyellow",
    "BeenThereDoneThat" = "lightblue",
    "TooRichToPitch" = "lightgreen"
  )) +
  labs(title = "Walks Across Salary Levels", x = "Salary Level", y = "Walks") +
  theme_minimal() +
  theme(legend.position = "top")


Interpretation of EDA
Hits: We observed an upward trend in Hits as salary level increases. Players in the TooRichToPitch salary level had both a higher median and a wider spread of Hits compared to lower salary levels. Importantly, there were some extreme outliers (especially around 200+ Hits). We deducted that the upward shift in medians indicates that hitting performance is an important variable in salary prediction.

PutOuts: We saw that PutOuts had a much more scattered and less consistent pattern across salary levels. As we assumed, TooRitchToPitch salary group had a slightly higher median. However, interestingly, all salary levels demonstrated a large number of high outliers (around 1000+ PutOuts). From this result, we commented as that PutOuts may be more position-dependent rather than strictly related to salary.

CAtBats: Career AtBats showed a general increasing trend with the increase of salary. We observed that players in higher salary categories, specifically BeenThereDoneThat and TooRichToPitch. However, an important detail we obtined was the very wide spread of Career AtBats within each salary level. One other detail to mention is, especially in the TooRichToPitch level, we observed several extreme outliers with Career AtBat counts exceeding 10,000. From these results, we made the suggestion that due to the overlap and wide distribution, while experience matters, seasonal performance metrics may provide sharper separation.

AtBats: This variable actually showed a much cleaner pattern than CAtBat. AtBats median increased consistently with salary level increase. In addition, the spread of the boxes were moderate. We caught a few outliers that were extreme in the highest salary group. For example, there were players with almost no at-bats. We believe that these might be because of injured players.

Walks: The variable Walks conveyed us a clear increasing trend across the salary levels. As we moved from RookieBudget to TooRichToPitch, the median number of Walks rose as well. Moreover, we detected that there were some outliers in the TooRichToPitch level, where players had extremely high Walks compared to the rest. We suggested that these represent elite players who are exceptionally skilled.

Running the Model
In order to perform the multinomial logistic regression, we used the multinom() function from the nnet package, as we have learned in our lab sessions. Main reason why we used multinom() as a function, is due to its handling of categorical response variables with more than two outcomes. Since our response variable is SalaryLevel, and has 4 different categories, it was an efficient function to run our model.

In our model, we only wanted to include performance metrics. Therefore, we purposefully removed 3 variables from our fitting: League, Division, and New League, which are non-performance metrics, and factors. In other terms, they are categorical indicators of team or league affiliation, not direct measures of a player’s on-field performance. If we had decided to include them, salary differences could have reflected organizational or regional disparities rather than individual success.

During the fitting, we involved all relevant performance-related predictors: both season-based (AtBat, Hits, HmRun, Runs, RBI, Walks, etc.) and career-based statistics (CAtBat, CHits, CHmRun, CRuns, CRBI, CWalks). We also included field-related metrics such as PutOuts, Assists, and Errors. We did not include Salary as a predictor in our model because Salary was used to create the response variable SalaryLevel. If we have included Salary, it would have caused a data leakage, where the response variable would “leak” information to the other predictors, which might have caused a biased and an invalid conclusion, also it would have destroyed the purpose of our analysis.

As we previously mentioned, here is our 4 different salary levels: RookieBudget, SolidStarter, BeenThereDoneThat, TooRichToPitch. Our model estimated the log-odds of being in each non-baseline category in comparison to the baseline category. In our case, RookieBudget was the baseline/reference category. Overall, what our output is an indication of how a one-unit increase in a predictor affects the log-odds of a player in a specific salary level compared to RookieBudget, while all other variables are held constant.

Lastly, in the last lines of our code, you can see that we have chosen accuracy as an error measure. Even though we are going to explain it in depth after the cross-validation why we have chosen, briefly, we can say that we chose it because our case is a multinomial classification problem, not just binary. Regarding this, accuracy has a great interpretability which allows us to make comments and comparison on our model. Since test error = 1 - accuracy, accuracy directly helps us address minimizing prediction error.

model <- multinom(SalaryLevel ~ AtBat + Hits + HmRun + Runs + RBI + Walks +
                    Years + CAtBat + CHits + CHmRun + CRuns + CRBI + CWalks +
                    PutOuts + Assists + Errors,
                  data = Data) 
## # weights:  72 (51 variable)
## initial  value 359.050240 
## iter  10 value 267.646126
## iter  20 value 257.264837
## iter  30 value 238.799496
## iter  40 value 235.420026
## iter  50 value 220.213024
## iter  60 value 203.075080
## final  value 203.062174 
## converged
summary(model)
## Call:
## multinom(formula = SalaryLevel ~ AtBat + Hits + HmRun + Runs + 
##     RBI + Walks + Years + CAtBat + CHits + CHmRun + CRuns + CRBI + 
##     CWalks + PutOuts + Assists + Errors, data = Data)
## 
## Coefficients:
##                   (Intercept)       AtBat       Hits       HmRun       Runs
## SolidStarter        -2.119442 -0.03593779 0.02194497 -0.14307055 0.15500611
## BeenThereDoneThat   -3.589896 -0.04048693 0.07329386 -0.09775665 0.06661686
## TooRichToPitch      -5.681062 -0.03447630 0.06010758 -0.05073335 0.06707451
##                          RBI        Walks      Years      CAtBat       CHits
## SolidStarter      0.07361810 -0.007672958  0.2314250 0.011938334 -0.02148513
## BeenThereDoneThat 0.05333681  0.044999596  0.1013892 0.009384121 -0.01465302
## TooRichToPitch    0.03888834  0.052997948 -0.1258456 0.008701673 -0.01464651
##                       CHmRun        CRuns         CRBI      CWalks
## SolidStarter      0.03507606 -0.023852861 -0.017924048 0.011663216
## BeenThereDoneThat 0.03284238 -0.011127212 -0.015857585 0.006668275
## TooRichToPitch    0.02077034 -0.005058175 -0.008229895 0.004192038
##                         PutOuts     Assists      Errors
## SolidStarter      -0.0001852251 0.007906414 -0.14984796
## BeenThereDoneThat  0.0004663703 0.007587667 -0.10888977
## TooRichToPitch     0.0017017788 0.006098294 -0.07424669
## 
## Std. Errors:
##                   (Intercept)      AtBat       Hits    HmRun       Runs
## SolidStarter      0.006417864 0.01289570 0.04538321 0.118291 0.05791268
## BeenThereDoneThat 0.004682597 0.01318963 0.04637865 0.120964 0.05936727
## TooRichToPitch    0.005952914 0.01350297 0.04777750 0.124539 0.06115663
##                          RBI      Walks      Years      CAtBat      CHits
## SolidStarter      0.04804173 0.03878508 0.07028187 0.005863058 0.02425990
## BeenThereDoneThat 0.04975797 0.03977985 0.06205622 0.005830970 0.02431348
## TooRichToPitch    0.05128045 0.04127905 0.08223463 0.005838920 0.02454698
##                       CHmRun      CRuns       CRBI     CWalks     PutOuts
## SolidStarter      0.05413351 0.02243750 0.02460077 0.01654191 0.001272324
## BeenThereDoneThat 0.05393460 0.02254007 0.02454886 0.01654104 0.001168016
## TooRichToPitch    0.05441433 0.02289911 0.02477038 0.01664024 0.001144123
##                       Assists     Errors
## SolidStarter      0.003832414 0.07304611
## BeenThereDoneThat 0.003876996 0.07404536
## TooRichToPitch    0.004030371 0.07814092
## 
## Residual Deviance: 406.1243 
## AIC: 508.1243
predicted_classes <- predict(model, newdata = Data)
actual_classes <- Data$SalaryLevel
accuracy <- mean(predicted_classes == actual_classes)
paste("Training Accuracy:", round(accuracy * 100, 2), "%")
## [1] "Training Accuracy: 69.88 %"
In-Depth Interpretation of Our Predictors
To interpret our predictors’ results, we decided to divide them into two for a more compact overview: strong/positive predictors and weak/inconsistent predictors.

Strong/Positive Predictors:
Years: The coefficients for Years were positive and relatively large, specifically for the lower salary levels. However, it becomes negative for TooRichToPitch. This might suggest that, experience even though is a strong factor in reaching middle-high salary levels, is not enough for being in the top tier salary level.

Hits: It had strong positive and consistent coefficients, especially for BeenThereDoneThat and TooRichToPitch, showing that players with higher hitting performance is linked with higher salary levels. To give an example, a coefficient of 0.093 for BeenThereDoneThat means that, more hits increase the probability of being in the mid-to-high salary levels.

Walks: Statistically, the significance of Walks increase as we move to the higher salary levels. While it’s slightly negative for SolidStarter level, it becomes significantly positive for the top two salary levels.

RBI: RBI also conveys a positive relationship within the higher salary levels. Even though it has smaller coefficients compared to Hits, it still is statistically meaningful. An important interpretation from the results of RBI is that, coefficient size decreases from SolidStarter to TooRichToPitch. This can be an indication of having more discriminative power in separating low from mid-tier players, but is less important among higher salary levels.

AtBat & Runs: These two performance metrics tend to have smaller but yet positive influence. They show statistical significance, however, not strongly as Hits, RBIs, or Walks. We believe that this might be because of overlapping with other variables.

Weak/Inconsistent Predictors
Errors: From the results we have obtained, the coefficients of Errors are negative, which may suggest that players who make more fielding mistakes tend to be categorized within lower salary levels. In addition, the coefficients are small and the standard errors are significantly large, which makes Errors as a predictor, weak.

PutOuts & Assists: For both variables, they have a fluctuating and small coefficients. They have some cases where the standard errors are as large as or larger than their effect sizes. Additionally, since these variables are position-dependent, they add noise to the model. Overall, they have a minimal explanatory value.

Career Metrics (CAtBat, CHits, CHmRun, CRuns, CRBI, CWalks): Variables within the career metrics represent accumulated performance. This total performance may overlap with seasonal performance metrics that are already in the model. They tend to have low coefficient values and large standard errors. Therefore, despite their relevance conceptually, they do not show clear predictive value.

On the other hand, CHits & CHmRun variables have been, sometimes, statistically meaningful. They had small coefficients like the rest of the career metrics, and they were inconsistent. This demonstrated inconsistency, and weak relationship.

Residual Deviance and AIC
Similar to what we had in linear regression as Residual Sum of Squares, in logistic regression Residual Deviance is a measurement that is used to understand how well the model fits the observed data. Lower the residual deviance, better the model, and the better explanation of the variation in the response variable.

After running our model, we obtained 406.1243 from Residual Deviance. We interpreted as, our model provides a reasonable level of fit to the data. It can be said that, there is still some error in how well our model can predict which salary level each player belongs to. However, we can also say that our model is not random, the deviance is quite moderate, suggesting a statistically meaningful information about our results.

On the other hand, AIC (Akaike Information Criterion), is a measure for finding the right model which fits the data well, also ensuring that the model is not overly complex by penalizing, i.e. our model does not use too many predictors unnecessarily. Therefore, the lower the AIC, better the model fit.

In our results, we obtained an AIC of 508.1243. While we cannot compare to any previous model fittings, we can deduct that our model has been penalized for complexity since the score is not that low, probably due to unnecessary predictors. However, it also indicates that our model is not too much of an overfit.

Overall, even though our model has a large number of predictors, it still has a reasonable performance. In addition, the model fit is not poor, while demonstrating a moderate predictive ability. However, there is a clear room for improvement. Hence, we can say that our model could benefit from reducing less informative predictors and/or tuning, to enhance both its fit and predictive accuracy.

Training Accuracy
From running our full model, we have received a very optimistic accuracy, %69.88. As we know, training accuracy measures how well the model is able to classify the same data it was trained on. In other words, almost 70% of the players were correctly classified into their correct salary levels using the predictors.

We can definitely say that approximately 70% accuracy in a multi-class classification problem with four salary levels is actually pretty reasonable, because random guessing would only give us about 25% accuracy (since 4 categories => 1/4 = 25%). Also, this result indicates that our full model has learned the patterns and is able to perform on the test data with the understanding of important relationships. In addition, our training accuracy also doesn’t show too much overfitting, since it is not too close to 100 percent.

However, we should not be tricked by this since this is the training accuracy, what actually matters is the test error. Therefore, we have to perform cross-validation to gather more information how our model performs on the test set.

Cross-Validation with LOO-CV
Just like we mentioned above, to understand whether our model is performing well with little test error on the test set, we have to carry out cross-validation. The main reason why we are conducting this is because training accuracy alone is not a reliable measure of how well a model will perform on new, unseen data. Also, by applying cross validation to both the full model and the reduced model, we can fairly compare their performance and choose the model that generalizes better.

As we were given in the project guidelines, we had to make the code generate that selects a random cross validation method out of those 5 that are stated. In the beginning, since our group consists of 4 people, we started by determining the group member who had the closest birthday to the 21st of August.

The group member who had the closest birthday to 21st August was our dearest member: Deniz Yakici with her birthday being 05/07/2004. Therefore, we set the seed to her birthday. We selected the method randomly and it picked LOO-CV.

To give a general description, LOO-CV is a type of cross-validation where the model is trained n times (n being the number of observations), and each time the model leaves out only one observation as the test set, and uses the rest as the training set.

cv_methods <- c("Validation Set", "LOOCV", "5-Fold CV", "10-Fold CV")
set.seed(05072004)
selected_method <- sample(cv_methods, 1)
selected_method
## [1] "LOOCV"
Beginning the LOO-CV Loop
This code chunk begins with initializing an empty vector to store the predicted salary levels for each observation in the dataset. Then, we manually created a for loop to iterate through every row in the dataset.

Within the loop:

train_data <- Data[-i, ]

test_data <- Data[i, ] is used to split our data into train and test. Our train data, which we used to fit the model, is all the rows except the i-th one. Our test data is only the i-th row, which we used it for testing and prediction, normally so-called “unseen data”.

Then, we trained a multinomial logistic regression, similar to what we have done without a cross-validation. We continued using the same variables, like before.

After the loop, we converted the actual SalaryLevel values to characters for comparison. We built a confusion matrix to also compare predictions to the actual values. By this matrix, we were able to understand how many predictions were correct or misclassified.

Lastly, we calculated the prediction accuracy, as the percentage of correct predictions over the total predictions.

loocv_preds <- c()

for (i in 1:nrow(Data)) {
  train_data <- Data[-i, ]
  test_data <- Data[i, ]
  model <- multinom(SalaryLevel ~ AtBat + Hits + HmRun + Runs + RBI + Walks +
                    Years + CAtBat + CHits + CHmRun + CRuns + CRBI + CWalks +
                    PutOuts + Assists + Errors, data = train_data, trace = FALSE)
  pred <- predict(model, newdata = test_data)
  loocv_preds <- c(loocv_preds, as.character(pred))
}

actual <- as.character(Data$SalaryLevel)
conf_matrix <- table(Predicted = loocv_preds, Actual = actual)
print(conf_matrix)
##                    Actual
## Predicted           BeenThereDoneThat RookieBudget SolidStarter TooRichToPitch
##   BeenThereDoneThat                28            1           11             23
##   RookieBudget                      2           58           12              2
##   SolidStarter                     22            5           38              3
##   TooRichToPitch                   18            2            3             31
accuracy <- mean(loocv_preds == actual)
paste("LOOCV Accuracy:", round(accuracy * 100, 2), "%")
## [1] "LOOCV Accuracy: 59.85 %"
Interpretation
We read the results of confusion matrix, as diagonal entries being the correct predictions, whereas the off-diagonal ones being the misclassifications.

BeenThereDoneThat: 28 players were correctly classified.
RookieBudget: 58 players were correctly classified.
SolidStarter: 38 players were correctly classified.
TooRichToPitch: 31 players were correctly classified.
From this we made the assumption that the model has trouble distinguishing between lower-mid and lowest salary tiers. We believe that the reason is because their performance metrics might overlap heavily.

Overall, our model reached % 59.85 of LOO-CV accuracy. From this result, we made the deduction that almost 60 percent of the salary level predictions were actually the same as the true salary level categories. This accuracy can be said to be moderate, and the overall performance is reasonable but not the best to generalize on an unseen data.

Second Cross-Validation
Within our project, since we are 4 people, we conducted a second cross-validation. Rather than choosing randomly another one ourselves, we decided to be consistent with the first method of choice, but this time we chose the second closest birthday to 21st of August. First, we used the same code structure in LOO-CV to sample the cross-validation method, however, it gave us, again, LOO-CV. Therefore, we used ChatGPT to modify the code to make sure that the CV method was different from the previous one. ChatGPT provided us with this setdiff(cv_methods, “LOOCV”) code snippet. Our group member, who had the second closest birthday to 21st of August, was Defne Turcan with her birthday being 02/12/2005. Therefore, we set the seed accordingly, and proceeded with the same structure as before.

Our code picked the second cross-validation method: 5-Fold CV.

5-Fold Cross-Validation method is trained and tested 5 times. In each time, 4 folds are used for training, and the remaining 1 is used for testing. The whole process rotates. This ensures that every fold goes through the test set once. After the runs, 5-Fold takes the average all errors from each fold.

second_cv_options <- setdiff(cv_methods, "LOOCV")
set.seed(02122005)
second_method <- sample(second_cv_options, 1)
second_method
## [1] "5-Fold CV"
Beginning of 5-Fold Loop
With the first line of code, we split the data into 5 groups (folds) while using sampling based on the SalaryLevel. We created an empty vector to store the predicted salary levels, similar to before, from each test fold. We manually initiated a for loop to train and test the model with each fold.

Within the loop:

We wrote a test_index which holds the row numbers for the i-th fold. Then split the data into train and test. Our train_data contains the other 4 folds which we used to fit the model, whereas the test_data is used to evaluate our model performance.

Lastly, we predicted the salary level categories on our unseen test data, and stored the results in appropriate indices of kfold_preds. Then, we stored the true class labels with using actual <- as.character(Data$SalaryLevel).

To compare, we used the confusion matrix. It allowed us to compare the predicted values and the actual values. Eventually, we checked how many predictions were actually matching their correct categories.

set.seed(02122005)
folds <- createFolds(Data$SalaryLevel, k = 5, list = TRUE)

kfold_preds <- rep(NA, nrow(Data))

for (i in 1:5) {
  test_index <- folds[[i]]
  train_data <- Data[-test_index, ]
  test_data <- Data[test_index, ]
  
  model <- multinom(SalaryLevel ~ AtBat + Hits + HmRun + Runs + RBI + Walks +
                      Years + CAtBat + CHits + CHmRun + CRuns + CRBI + CWalks +
                      PutOuts + Assists + Errors, 
                    data = train_data, trace = FALSE)
  
  preds <- predict(model, newdata = test_data)
  kfold_preds[test_index] <- as.character(preds)
}


actual <- as.character(Data$SalaryLevel)
kfold_conf_matrix <- table(Predicted = kfold_preds, Actual = actual)
print(kfold_conf_matrix)
##                    Actual
## Predicted           BeenThereDoneThat RookieBudget SolidStarter TooRichToPitch
##   BeenThereDoneThat                29            1           11             18
##   RookieBudget                      0           59           11              2
##   SolidStarter                     25            5           37              6
##   TooRichToPitch                   16            1            5             33
kfold_accuracy <- mean(kfold_preds == actual)
paste("5-Fold CV Accuracy:", round(kfold_accuracy * 100, 2), "%")
## [1] "5-Fold CV Accuracy: 61 %"
Interpretation
Same as LOO-CV results, we received the confusion matrix. It gave us the statistics of how many times our model correctly predicted our salary levels, and how many times it misclassified them. To read the matrix, we followed each row being predicted values, and each column being the actual values.

RookieBudget: 59 players were correctly classified.
SolidStarter: 29 players were correctly classified.
BeenThereDoneThat: 37 players were correctly classified.
TooRichToPitch: 33 players were correctly classified.
From these results we made the assumption that the model struggles to differentiate between mid-high performers and top performers. Moreover, SolidStarter and BeenThereDoneThat probably have similar seasonal and career statistics, that’s why they overlap and cause misclassification.

Since we obtained an accuracy of 61 %, we made the claim that it has increased in accuracy compared to LOO-CV, but still has a moderate test error. This can be explained by since 5-Fold cross-validation usually provides a more stable and realistic estimate of model performance compared to LOO-CV which tends to have higher variance.

Better Model
In this part of the project, we had to choose a better model that had the lowest test error. First, we tried all the relevant methods that we learned in our lab sessions. We even tried to remove the variables with less signifcance manually, but then we consulted ChatGPT to gather information on how we can use a method that would have the lowest test error, hence a better model fit. We obtained the function stepAIC() to conduct backward selection. First, we set the full model (excluding non-performance metrics) to the multinomial logistic regression. Since we chose backward selection, we first loaded the full model in order to reduce it later. In our reduced model, we applied stepAIC function. The stepAIC model first takes all the predictors, then chooses the predictor that increases the AIC minimally and removes it. This model continues to remove predictors until the AIC is increased and our model is worsened. AIC here is a great method for model selection since it balances the model complexity and model fit. Specifically, AIC rewards models that fit the data well, while penalizing those that become too complex.

full_model <- multinom(SalaryLevel ~ AtBat + Hits + HmRun + Runs + RBI + Walks +
                         Years + CAtBat + CHits + CHmRun + CRuns + CRBI + CWalks +
                         PutOuts + Assists + Errors, data = Data, trace = FALSE)


reduced_model <- stepAIC(full_model, direction = "backward", trace = TRUE)
## Start:  AIC=508.12
## SalaryLevel ~ AtBat + Hits + HmRun + Runs + RBI + Walks + Years + 
##     CAtBat + CHits + CHmRun + CRuns + CRBI + CWalks + PutOuts + 
##     Assists + Errors
## 
##           Df    AIC
## - CHmRun   3 503.32
## - CHits    3 503.34
## - CRBI     3 504.33
## - HmRun    3 504.57
## - RBI      3 505.06
## - CWalks   3 505.38
## - Hits     3 506.44
## - CRuns    3 506.73
## - Assists  3 507.01
## - PutOuts  3 507.12
## - Years    3 507.18
## - Errors   3 507.24
## - CAtBat   3 507.38
## <none>       508.12
## - Walks    3 511.00
## - AtBat    3 513.10
## - Runs     3 514.16
## 
## Step:  AIC=503.32
## SalaryLevel ~ AtBat + Hits + HmRun + Runs + RBI + Walks + Years + 
##     CAtBat + CHits + CRuns + CRBI + CWalks + PutOuts + Assists + 
##     Errors
## 
##           Df    AIC
## - CRBI     3 498.94
## - HmRun    3 498.95
## - RBI      3 499.63
## - CWalks   3 500.17
## - CHits    3 501.00
## - CRuns    3 501.98
## - Assists  3 501.99
## - Years    3 502.16
## - Hits     3 502.25
## - Errors   3 502.43
## - PutOuts  3 502.68
## <none>       503.32
## - CAtBat   3 503.90
## - Walks    3 506.07
## - AtBat    3 508.69
## - Runs     3 509.42
## 
## Step:  AIC=498.94
## SalaryLevel ~ AtBat + Hits + HmRun + Runs + RBI + Walks + Years + 
##     CAtBat + CHits + CRuns + CWalks + PutOuts + Assists + Errors
## 
##           Df    AIC
## - RBI      3 494.92
## - HmRun    3 494.93
## - CWalks   3 495.30
## - CHits    3 497.24
## - CRuns    3 497.64
## - Hits     3 497.73
## - Assists  3 497.86
## - Years    3 497.90
## - Errors   3 498.02
## - PutOuts  3 498.58
## <none>       498.94
## - CAtBat   3 499.34
## - Walks    3 501.46
## - AtBat    3 504.06
## - Runs     3 505.85
## 
## Step:  AIC=494.92
## SalaryLevel ~ AtBat + Hits + HmRun + Runs + Walks + Years + CAtBat + 
##     CHits + CRuns + CWalks + PutOuts + Assists + Errors
## 
##           Df    AIC
## - HmRun    3 490.46
## - CWalks   3 491.16
## - CHits    3 493.11
## - Assists  3 493.39
## - Errors   3 493.75
## - CRuns    3 493.83
## - Hits     3 494.26
## - Years    3 494.43
## - PutOuts  3 494.78
## <none>       494.92
## - CAtBat   3 495.38
## - Walks    3 497.24
## - AtBat    3 499.42
## - Runs     3 500.45
## 
## Step:  AIC=490.46
## SalaryLevel ~ AtBat + Hits + Runs + Walks + Years + CAtBat + 
##     CHits + CRuns + CWalks + PutOuts + Assists + Errors
## 
##           Df    AIC
## - CWalks   3 486.70
## - CHits    3 488.42
## - CRuns    3 489.44
## - Errors   3 489.51
## - Assists  3 489.99
## - Years    3 490.05
## - Hits     3 490.18
## <none>       490.46
## - CAtBat   3 490.86
## - PutOuts  3 490.96
## - Walks    3 492.50
## - AtBat    3 495.48
## - Runs     3 496.35
## 
## Step:  AIC=486.7
## SalaryLevel ~ AtBat + Hits + Runs + Walks + Years + CAtBat + 
##     CHits + CRuns + PutOuts + Assists + Errors
## 
##           Df    AIC
## - CRuns    3 483.58
## - Errors   3 485.92
## - Assists  3 486.00
## - Years    3 486.52
## - PutOuts  3 486.64
## <none>       486.70
## - Hits     3 486.86
## - CHits    3 489.54
## - Walks    3 489.54
## - Runs     3 491.01
## - CAtBat   3 492.45
## - AtBat    3 494.58
## 
## Step:  AIC=483.58
## SalaryLevel ~ AtBat + Hits + Runs + Walks + Years + CAtBat + 
##     CHits + PutOuts + Assists + Errors
## 
##           Df    AIC
## - Errors   3 482.52
## - Assists  3 483.24
## - PutOuts  3 483.34
## <none>       483.58
## - Hits     3 483.95
## - Years    3 483.97
## - Runs     3 486.15
## - Walks    3 487.34
## - CAtBat   3 488.21
## - AtBat    3 491.14
## - CHits    3 491.57
## 
## Step:  AIC=482.52
## SalaryLevel ~ AtBat + Hits + Runs + Walks + Years + CAtBat + 
##     CHits + PutOuts + Assists
## 
##           Df    AIC
## - Assists  3 478.52
## <none>       482.52
## - Hits     3 482.84
## - PutOuts  3 482.94
## - Years    3 483.05
## - Runs     3 484.24
## - Walks    3 485.37
## - CAtBat   3 486.00
## - CHits    3 489.46
## - AtBat    3 490.21
## 
## Step:  AIC=478.52
## SalaryLevel ~ AtBat + Hits + Runs + Walks + Years + CAtBat + 
##     CHits + PutOuts
## 
##           Df    AIC
## - Years    3 478.46
## <none>       478.52
## - Hits     3 479.48
## - PutOuts  3 479.86
## - Runs     3 480.51
## - Walks    3 481.02
## - CAtBat   3 482.93
## - AtBat    3 484.91
## - CHits    3 486.56
## 
## Step:  AIC=478.46
## SalaryLevel ~ AtBat + Hits + Runs + Walks + CAtBat + CHits + 
##     PutOuts
## 
##           Df    AIC
## <none>       478.46
## - Runs     3 479.90
## - PutOuts  3 480.02
## - Hits     3 480.15
## - Walks    3 480.52
## - AtBat    3 488.48
## - CHits    3 489.93
## - CAtBat   3 490.00
predicted_reduced <- predict(reduced_model, newdata = Data)


accuracy_reduced <- mean(predicted_reduced == Data$SalaryLevel)
paste("Reduced Model Accuracy:", round(accuracy_reduced * 100, 2), "%")
## [1] "Reduced Model Accuracy: 67.95 %"
Performing Cross-Validations on the Reduced Model
One might ask, but isn’t AIC a criteria for model selection, it works on the training error not test error? Even though AIC is a metric for model selection and not evaluation, our goal is also to determine which predictors meaningfully contribute to explaining variation in the response variable. So, it is not necessarily maximizing accuracy on the training data. But using this function was essential to first reduce the variables and then look for the improvement in the test set. Therefore, we didn’t just use the stepAIC function and finished the model selection. After receiving the variables that are the most significant, we calculated the overall accuracy. We got %67.95 of an overall accuracy, which showed an extensive increase in the accuracy for the training error. But as we know, the accuracy that we calculated is the training error, which is not reliable. Because we are interested in the test error, since we want to know how good our model is generalizing in the unseen data. Therefore, we performed both cross-validation methods we used before again to check if our test error has decreased.

5-Fold CV
We used the same structure as we used before with the full model. But, obviously, with the reduced model which has fewer predictors that we obtained from the backward selection. Then, we selected the same seed to make sure that our folds are identical between the full model and reduced model cross-validation. This allowed us to do model comparison unbiased and fair. We received a test error of %61.39.

set.seed(02122005)
folds <- createFolds(Data$SalaryLevel, k = 5, list = TRUE)

kfold_preds <- rep(NA, nrow(Data))

for (i in 1:5) {
  test_index <- folds[[i]]
  train_data <- Data[-test_index, ]
  test_data <- Data[test_index, ]
  

  model_reduced_cv <- multinom(SalaryLevel ~ AtBat + Hits + Runs + Walks + CAtBat + CHits + 
    PutOuts, data = train_data, trace = FALSE)
  
  preds <- predict(model_reduced_cv, newdata = test_data)
  kfold_preds[test_index] <- as.character(preds)
}


actual <- as.character(Data$SalaryLevel)
conf_matrix <- table(Predicted = kfold_preds, Actual = actual)
print(conf_matrix)
##                    Actual
## Predicted           BeenThereDoneThat RookieBudget SolidStarter TooRichToPitch
##   BeenThereDoneThat                33            1           11             22
##   RookieBudget                      0           60           12              2
##   SolidStarter                     22            4           37              6
##   TooRichToPitch                   15            1            4             29
kfold_accuracy <- mean(kfold_preds == actual)
paste("5-Fold CV Accuracy (Reduced Model):", round(kfold_accuracy * 100, 2), "%")
## [1] "5-Fold CV Accuracy (Reduced Model): 61.39 %"
LOO-CV
To be even more consistent on calculating the test error. Since we used LOO-CV as a cross-validation method before, we wanted to run it also on the reduced model. We again set the same seed as the full model, and ran it with only using the variables that we had with stepAIC. Impressively, we received a significant increase in accuracy, almost %4. After conducting cross-validatio LOO-CV on our reduced model. We obtained %63.71 as the accuracy. Compared to our previous accuracy in the full model, which was %59.85. We can see a significant increase, suggesting our model being better than before with having lower test error.

set.seed(05072004) 
n <- nrow(Data)
loo_preds <- rep(NA, n)  

for (i in 1:n) {
 
  train_data <- Data[-i, ]
  test_data <- Data[i, , drop = FALSE] 
  
  model_reduced_loo <- multinom(SalaryLevel ~ AtBat + Hits + Runs + Walks + CAtBat + CHits + 
    PutOuts, data = train_data, trace = FALSE)
  
  pred <- predict(model_reduced_loo, newdata = test_data)
  loo_preds[i] <- as.character(pred)
}


actual_loo <- as.character(Data$SalaryLevel)
loo_conf_matrix <- table(Predicted = loo_preds, Actual = actual_loo)
print(loo_conf_matrix)
##                    Actual
## Predicted           BeenThereDoneThat RookieBudget SolidStarter TooRichToPitch
##   BeenThereDoneThat                35            1           10             22
##   RookieBudget                      1           61           13              2
##   SolidStarter                     20            3           37              3
##   TooRichToPitch                   14            1            4             32
loo_accuracy <- mean(loo_preds == actual_loo)
paste("LOOCV Accuracy (Reduced Model):", round(loo_accuracy * 100, 2), "%")
## [1] "LOOCV Accuracy (Reduced Model): 63.71 %"
Interpretation on the Reduced Model
In comparison with the full model, previously we received a LOO-CV accuracy of %59.85 on the full model. Whereas now, we obtained %63.71. On the other hand, for 5-Fold CV, in the full model we received %61, while in the reduced model we obtained %61.39. Overall, we can say that our model has improved from the full model, in both training and test sets. Since we are more interested in the test error than training, we still can say that our model has generally performed better in the reduced model. Even though %0.39 increase in the accuracy for 5-Fold CV seems not significant, we can explain the little increase because of having very little observations and hence a small dataset. In addition, we thought that another reason why it has increased very little might be due to cross-validation variance, since accuracy can fluctuate slightly due to the way folds are split, especially on small datasets. Also, we believe that this outcome is expected since we saw in our exploratory data analysis, the natural overlap between salary levels, the presence of noise in performance data, and the limited scope of available predictors.

We also looked for other methods to improve our model more, if our function wasn’t mulinom() but a random forest or decision trees we could have performed hyperparameter tuning, but since we don’t have any parameters to tune, this method was for us the best in having lower test error. Despite these, we saw that the reduced model achieved a balance between model complexity and predictive ability, successfully lowering the test error while enhancing model robustness.

Our Model With Including Factor Variables
Even though this part was not asked from us, to have a more compact and a better understanding of our model we decided to reconsider our approach. Therefore, we decided to complete the entire project again but this time with the factor variables. As we mentioned in the beginning, we purposefully removed 3 variables: League, Division, and NewLeague. However, we realised that what if they are actually significant. Because we understood that sometimes categorical attributes can carry important structural information about a player’s playing context, such as the league type or the division they belong to, which can ultimately influence their salary.

Below you can find the entire coding just with the difference of adding factor variables.

library(tidyverse)
library(ggplot2)
library(nnet)
library(caret)
library(MASS)
Data <- read.csv("Hitters.csv")
str(Data)
## 'data.frame':    317 obs. of  20 variables:
##  $ AtBat    : int  293 315 479 496 321 594 185 298 323 574 ...
##  $ Hits     : int  66 81 130 141 87 169 37 73 81 159 ...
##  $ HmRun    : int  1 7 18 20 10 4 1 0 6 21 ...
##  $ Runs     : int  30 24 66 65 39 74 23 24 26 107 ...
##  $ RBI      : int  29 38 72 78 42 51 8 24 32 75 ...
##  $ Walks    : int  14 39 76 37 30 35 21 7 8 59 ...
##  $ Years    : int  1 14 3 11 2 11 2 3 2 10 ...
##  $ CAtBat   : int  293 3449 1624 5628 396 4408 214 509 341 4631 ...
##  $ CHits    : int  66 835 457 1575 101 1133 42 108 86 1300 ...
##  $ CHmRun   : int  1 69 63 225 12 19 1 0 6 90 ...
##  $ CRuns    : int  30 321 224 828 48 501 30 41 32 702 ...
##  $ CRBI     : int  29 414 266 838 46 336 9 37 34 504 ...
##  $ CWalks   : int  14 375 263 354 33 194 24 12 8 488 ...
##  $ League   : chr  "A" "N" "A" "N" ...
##  $ Division : chr  "E" "W" "W" "E" ...
##  $ PutOuts  : int  446 632 880 200 805 282 76 121 143 238 ...
##  $ Assists  : int  33 43 82 11 40 421 127 283 290 445 ...
##  $ Errors   : int  20 10 14 3 4 25 7 9 19 22 ...
##  $ Salary   : num  NA 475 480 500 91.5 ...
##  $ NewLeague: chr  "A" "N" "A" "N" ...
summary(Data)
##      AtBat            Hits           HmRun            Runs       
##  Min.   : 16.0   Min.   :  1.0   Min.   : 0.00   Min.   :  0.00  
##  1st Qu.:256.0   1st Qu.: 64.0   1st Qu.: 4.00   1st Qu.: 30.00  
##  Median :379.0   Median : 96.0   Median : 8.00   Median : 48.00  
##  Mean   :381.2   Mean   :101.1   Mean   :10.75   Mean   : 50.89  
##  3rd Qu.:512.0   3rd Qu.:137.0   3rd Qu.:16.00   3rd Qu.: 69.00  
##  Max.   :687.0   Max.   :238.0   Max.   :40.00   Max.   :130.00  
##                                                                  
##       RBI             Walks            Years            CAtBat     
##  Min.   :  0.00   Min.   :  0.00   Min.   : 1.000   Min.   :   19  
##  1st Qu.: 28.00   1st Qu.: 22.00   1st Qu.: 4.000   1st Qu.:  822  
##  Median : 44.00   Median : 35.00   Median : 6.000   Median : 1928  
##  Mean   : 47.95   Mean   : 38.51   Mean   : 7.429   Mean   : 2646  
##  3rd Qu.: 63.00   3rd Qu.: 53.00   3rd Qu.:11.000   3rd Qu.: 3919  
##  Max.   :121.00   Max.   :105.00   Max.   :24.000   Max.   :14053  
##                                                                    
##      CHits            CHmRun           CRuns             CRBI       
##  Min.   :   4.0   Min.   :  0.00   Min.   :   1.0   Min.   :   0.0  
##  1st Qu.: 209.0   1st Qu.: 14.00   1st Qu.: 101.0   1st Qu.:  91.0  
##  Median : 506.0   Median : 37.00   Median : 247.0   Median : 219.0  
##  Mean   : 717.3   Mean   : 68.94   Mean   : 358.1   Mean   : 328.8  
##  3rd Qu.:1051.0   3rd Qu.: 90.00   3rd Qu.: 518.0   3rd Qu.: 421.0  
##  Max.   :4256.0   Max.   :548.00   Max.   :2165.0   Max.   :1659.0  
##                                                                     
##      CWalks        League            Division            PutOuts      
##  Min.   :   0   Length:317         Length:317         Min.   :   0.0  
##  1st Qu.:  68   Class :character   Class :character   1st Qu.: 109.0  
##  Median : 170   Mode  :character   Mode  :character   Median : 212.0  
##  Mean   : 258                                         Mean   : 290.5  
##  3rd Qu.: 337                                         3rd Qu.: 325.0  
##  Max.   :1566                                         Max.   :1378.0  
##                                                                       
##     Assists          Errors           Salary        NewLeague        
##  Min.   :  0.0   Min.   : 0.000   Min.   :  67.5   Length:317        
##  1st Qu.:  7.0   1st Qu.: 3.000   1st Qu.: 190.0   Class :character  
##  Median : 40.0   Median : 6.000   Median : 420.0   Mode  :character  
##  Mean   :107.9   Mean   : 8.091   Mean   : 532.8                     
##  3rd Qu.:166.0   3rd Qu.:11.000   3rd Qu.: 750.0                     
##  Max.   :492.0   Max.   :32.000   Max.   :2460.0                     
##                                   NA's   :58
dim(Data)
## [1] 317  20
anyDuplicated(Data)
## [1] 0
colSums(is.na(Data))
##     AtBat      Hits     HmRun      Runs       RBI     Walks     Years    CAtBat 
##         0         0         0         0         0         0         0         0 
##     CHits    CHmRun     CRuns      CRBI    CWalks    League  Division   PutOuts 
##         0         0         0         0         0         0         0         0 
##   Assists    Errors    Salary NewLeague 
##         0         0        58         0
original_n <- nrow(Data)
Data <- na.omit(Data)
cleaned_n <- nrow(Data)
Data$SalaryLevel <- cut(Data$Salary,
                        breaks = quantile(Data$Salary, probs = c(0, 0.25, 0.5, 0.75, 1), na.rm = TRUE),
                        labels = c("RookieBudget", "SolidStarter", "BeenThereDoneThat", "TooRichToPitch"),
                        include.lowest = TRUE)

table(Data$SalaryLevel)
## 
##      RookieBudget      SolidStarter BeenThereDoneThat    TooRichToPitch 
##                66                64                70                59
Data$League <- as.factor(Data$League)
Data$Division <- as.factor(Data$Division)
Data$NewLeague <- as.factor(Data$NewLeague)


model <- multinom(SalaryLevel ~ AtBat + Hits + HmRun + Runs + RBI + Walks +
                    Years + CAtBat + CHits + CHmRun + CRuns + CRBI + CWalks +
                    PutOuts + Assists + Errors + Division + League + NewLeague,
                  data = Data) 
## # weights:  84 (60 variable)
## initial  value 359.050240 
## iter  10 value 267.644210
## iter  20 value 257.213521
## iter  30 value 237.996687
## iter  40 value 232.416688
## iter  50 value 218.849292
## iter  60 value 196.767496
## iter  70 value 195.317812
## final  value 195.315421 
## converged
summary(model)
## Call:
## multinom(formula = SalaryLevel ~ AtBat + Hits + HmRun + Runs + 
##     RBI + Walks + Years + CAtBat + CHits + CHmRun + CRuns + CRBI + 
##     CWalks + PutOuts + Assists + Errors + Division + League + 
##     NewLeague, data = Data)
## 
## Coefficients:
##                   (Intercept)       AtBat       Hits       HmRun       Runs
## SolidStarter        -2.703014 -0.04024208 0.03770561 -0.15087782 0.15694549
## BeenThereDoneThat   -4.476484 -0.04641756 0.09341681 -0.10369610 0.07151012
## TooRichToPitch      -5.566354 -0.03789970 0.07758256 -0.03622496 0.06256441
##                          RBI        Walks      Years     CAtBat       CHits
## SolidStarter      0.08239505 -0.001403527  0.2406987 0.01536027 -0.02884436
## BeenThereDoneThat 0.06304987  0.049998102  0.1300238 0.01294420 -0.02352491
## TooRichToPitch    0.03364088  0.062803887 -0.1895948 0.01259362 -0.02251776
##                       CHmRun       CRuns        CRBI       CWalks       PutOuts
## SolidStarter      0.04594024 -0.03023788 -0.02237094 0.0079807355 -0.0002331066
## BeenThereDoneThat 0.04349788 -0.01571901 -0.02014732 0.0026580249  0.0005111301
## TooRichToPitch    0.03184823 -0.01314544 -0.01203831 0.0002506327  0.0016423656
##                       Assists      Errors  DivisionW  LeagueN NewLeagueN
## SolidStarter      0.008361676 -0.17266354 -0.5155852 3.422584  -3.046514
## BeenThereDoneThat 0.008382301 -0.13961087 -0.3811349 3.458222  -2.631971
## TooRichToPitch    0.005955501 -0.08671069 -1.4830170 2.297816  -2.019577
## 
## Std. Errors:
##                   (Intercept)      AtBat       Hits     HmRun       Runs
## SolidStarter      0.014814141 0.01335715 0.04692165 0.1234786 0.05790332
## BeenThereDoneThat 0.017412653 0.01364750 0.04794887 0.1255874 0.05916524
## TooRichToPitch    0.007962515 0.01384442 0.04926317 0.1279996 0.06112227
##                          RBI      Walks      Years      CAtBat      CHits
## SolidStarter      0.05007022 0.04027155 0.07140607 0.005773175 0.02349087
## BeenThereDoneThat 0.05163242 0.04119219 0.06234205 0.005736461 0.02351300
## TooRichToPitch    0.05335410 0.04273632 0.08341543 0.005749007 0.02379165
##                       CHmRun      CRuns       CRBI     CWalks     PutOuts
## SolidStarter      0.05435400 0.02223822 0.02413779 0.01646046 0.001238250
## BeenThereDoneThat 0.05407951 0.02226706 0.02404887 0.01643798 0.001158108
## TooRichToPitch    0.05449661 0.02260721 0.02421663 0.01654449 0.001155263
##                       Assists     Errors   DivisionW    LeagueN NewLeagueN
## SolidStarter      0.004116673 0.07871314 0.015122240 0.07324131 0.07154647
## BeenThereDoneThat 0.004157339 0.07986275 0.012945413 0.11321472 0.11210042
## TooRichToPitch    0.004305403 0.08345695 0.005838167 0.04451421 0.04481763
## 
## Residual Deviance: 390.6308 
## AIC: 510.6308
predicted_classes <- predict(model, newdata = Data)
actual_classes <- Data$SalaryLevel
accuracy <- mean(predicted_classes == actual_classes)
paste("Training Accuracy:", round(accuracy * 100, 2), "%")
## [1] "Training Accuracy: 68.73 %"
cv_methods <- c("Validation Set", "LOOCV", "5-Fold CV", "10-Fold CV")
set.seed(05072004)
selected_method <- sample(cv_methods, 1)
selected_method
## [1] "LOOCV"
loocv_preds <- c()

for (i in 1:nrow(Data)) {
  train_data <- Data[-i, ]
  test_data <- Data[i, ]
  model <- multinom(SalaryLevel ~ AtBat + Hits + HmRun + Runs + RBI + Walks +
                    Years + CAtBat + CHits + CHmRun + CRuns + CRBI + CWalks +
                    PutOuts + Assists + Errors + League + Division + NewLeague, data = train_data, trace = FALSE)
  pred <- predict(model, newdata = test_data)
  loocv_preds <- c(loocv_preds, as.character(pred))
}

actual <- as.character(Data$SalaryLevel)
conf_matrix <- table(Predicted = loocv_preds, Actual = actual)
print(conf_matrix)
##                    Actual
## Predicted           BeenThereDoneThat RookieBudget SolidStarter TooRichToPitch
##   BeenThereDoneThat                30            1           12             21
##   RookieBudget                      1           58           11              2
##   SolidStarter                     22            5           37              4
##   TooRichToPitch                   17            2            4             32
accuracy <- mean(loocv_preds == actual)
paste("LOOCV Accuracy:", round(accuracy * 100, 2), "%")
## [1] "LOOCV Accuracy: 60.62 %"
second_cv_options <- setdiff(cv_methods, "LOOCV")
set.seed(02122005)
second_method <- sample(second_cv_options, 1)
second_method
## [1] "5-Fold CV"
set.seed(02122005)
folds <- createFolds(Data$SalaryLevel, k = 5, list = TRUE)

kfold_preds <- rep(NA, nrow(Data))

for (i in 1:5) {
  test_index <- folds[[i]]
  train_data <- Data[-test_index, ]
  test_data <- Data[test_index, ]
  
  model <- multinom(SalaryLevel ~ AtBat + Hits + HmRun + Runs + RBI + Walks +
                      Years + CAtBat + CHits + CHmRun + CRuns + CRBI + CWalks +
                      PutOuts + Assists + Errors + League + Division + NewLeague, 
                    data = train_data, trace = FALSE)
  
  preds <- predict(model, newdata = test_data)
  kfold_preds[test_index] <- as.character(preds)
}


actual <- as.character(Data$SalaryLevel)
kfold_conf_matrix <- table(Predicted = kfold_preds, Actual = actual)
print(kfold_conf_matrix)
##                    Actual
## Predicted           BeenThereDoneThat RookieBudget SolidStarter TooRichToPitch
##   BeenThereDoneThat                27            1           14             17
##   RookieBudget                      0           56            9              2
##   SolidStarter                     25            6           36              5
##   TooRichToPitch                   18            3            5             35
kfold_accuracy <- mean(kfold_preds == actual)
paste("5-Fold CV Accuracy:", round(kfold_accuracy * 100, 2), "%")
## [1] "5-Fold CV Accuracy: 59.46 %"
full_model <- multinom(SalaryLevel ~ AtBat + Hits + HmRun + Runs + RBI + Walks +
                         Years + CAtBat + CHits + CHmRun + CRuns + CRBI + CWalks +
                         PutOuts + Assists + Errors  + Division + NewLeague + League, data = Data, trace = FALSE)


reduced_model <- stepAIC(full_model, direction = "backward", trace = TRUE)
## Start:  AIC=510.63
## SalaryLevel ~ AtBat + Hits + HmRun + Runs + RBI + Walks + Years + 
##     CAtBat + CHits + CHmRun + CRuns + CRBI + CWalks + PutOuts + 
##     Assists + Errors + Division + NewLeague + League
## 
##             Df    AIC
## - CHmRun     3 506.00
## - CHits      3 506.05
## - CRBI       3 507.31
## - HmRun      3 507.63
## - CWalks     3 507.67
## - RBI        3 508.81
## - PutOuts    3 508.92
## - CRuns      3 509.05
## - NewLeague  3 509.49
## - Assists    3 510.03
## - Hits       3 510.11
## <none>         510.63
## - CAtBat     3 510.80
## - Errors     3 510.90
## - Years      3 511.42
## - League     3 511.47
## - Division   3 511.51
## - Walks      3 513.45
## - Runs       3 516.28
## - AtBat      3 517.41
## 
## Step:  AIC=506
## SalaryLevel ~ AtBat + Hits + HmRun + Runs + RBI + Walks + Years + 
##     CAtBat + CHits + CRuns + CRBI + CWalks + PutOuts + Assists + 
##     Errors + Division + NewLeague + League
## 
##             Df    AIC
## - HmRun      3 501.90
## - CRBI       3 502.53
## - CWalks     3 502.70
## - RBI        3 503.10
## - NewLeague  3 504.68
## - PutOuts    3 504.72
## - CHits      3 504.87
## - CRuns      3 505.04
## - Assists    3 505.06
## <none>         506.00
## - Errors     3 506.25
## - Hits       3 506.40
## - League     3 506.57
## - Years      3 506.63
## - Division   3 507.02
## - CAtBat     3 507.87
## - Walks      3 508.75
## - Runs       3 511.48
## - AtBat      3 513.29
## 
## Step:  AIC=501.9
## SalaryLevel ~ AtBat + Hits + Runs + RBI + Walks + Years + CAtBat + 
##     CHits + CRuns + CRBI + CWalks + PutOuts + Assists + Errors + 
##     Division + NewLeague + League
## 
##             Df    AIC
## - RBI        3 497.44
## - CWalks     3 498.37
## - CRBI       3 498.85
## - CHits      3 500.44
## - NewLeague  3 500.58
## - CRuns      3 500.74
## - PutOuts    3 500.75
## - Assists    3 501.86
## <none>         501.90
## - Division   3 502.28
## - Errors     3 502.63
## - Hits       3 502.65
## - League     3 502.95
## - Years      3 503.05
## - CAtBat     3 503.70
## - Walks      3 504.01
## - Runs       3 506.19
## - AtBat      3 509.11
## 
## Step:  AIC=497.44
## SalaryLevel ~ AtBat + Hits + Runs + Walks + Years + CAtBat + 
##     CHits + CRuns + CRBI + CWalks + PutOuts + Assists + Errors + 
##     Division + NewLeague + League
## 
##             Df    AIC
## - CWalks     3 493.79
## - CRBI       3 494.51
## - NewLeague  3 495.94
## - PutOuts    3 496.43
## - CRuns      3 496.46
## - Assists    3 496.48
## - CHits      3 496.53
## - Errors     3 497.30
## <none>         497.44
## - Division   3 497.51
## - League     3 498.15
## - Hits       3 498.60
## - Years      3 498.62
## - CAtBat     3 499.45
## - Walks      3 499.52
## - Runs       3 502.25
## - AtBat      3 503.89
## 
## Step:  AIC=493.79
## SalaryLevel ~ AtBat + Hits + Runs + Walks + Years + CAtBat + 
##     CHits + CRuns + CRBI + PutOuts + Assists + Errors + Division + 
##     NewLeague + League
## 
##             Df    AIC
## - CRBI       3 490.53
## - CRuns      3 491.23
## - PutOuts    3 492.12
## - NewLeague  3 492.29
## - Assists    3 492.85
## <none>         493.79
## - Division   3 493.89
## - Errors     3 493.90
## - League     3 494.76
## - Hits       3 494.96
## - Years      3 495.26
## - Walks      3 495.53
## - Runs       3 497.44
## - CHits      3 497.61
## - CAtBat     3 500.62
## - AtBat      3 502.52
## 
## Step:  AIC=490.53
## SalaryLevel ~ AtBat + Hits + Runs + Walks + Years + CAtBat + 
##     CHits + CRuns + PutOuts + Assists + Errors + Division + NewLeague + 
##     League
## 
##             Df    AIC
## - CRuns      3 488.32
## - NewLeague  3 488.84
## - PutOuts    3 489.58
## - Errors     3 490.50
## <none>         490.53
## - Assists    3 490.56
## - Division   3 490.60
## - League     3 490.97
## - Years      3 491.96
## - Hits       3 492.28
## - Walks      3 492.88
## - CHits      3 494.31
## - Runs       3 494.35
## - CAtBat     3 497.72
## - AtBat      3 499.85
## 
## Step:  AIC=488.32
## SalaryLevel ~ AtBat + Hits + Runs + Walks + Years + CAtBat + 
##     CHits + PutOuts + Assists + Errors + Division + NewLeague + 
##     League
## 
##             Df    AIC
## - NewLeague  3 486.02
## - PutOuts    3 487.54
## - League     3 488.23
## - Division   3 488.24
## - Errors     3 488.27
## <none>         488.32
## - Assists    3 488.87
## - Runs       3 488.90
## - Hits       3 489.95
## - Years      3 490.21
## - Walks      3 491.39
## - CAtBat     3 493.88
## - AtBat      3 496.75
## - CHits      3 496.81
## 
## Step:  AIC=486.02
## SalaryLevel ~ AtBat + Hits + Runs + Walks + Years + CAtBat + 
##     CHits + PutOuts + Assists + Errors + Division + League
## 
##            Df    AIC
## - League    3 483.22
## - PutOuts   3 485.73
## - Errors    3 485.94
## <none>        486.02
## - Division  3 486.11
## - Assists   3 487.03
## - Hits      3 487.31
## - Years     3 487.55
## - Runs      3 487.58
## - Walks     3 488.62
## - CAtBat    3 491.69
## - CHits     3 494.69
## - AtBat     3 495.13
## 
## Step:  AIC=483.22
## SalaryLevel ~ AtBat + Hits + Runs + Walks + Years + CAtBat + 
##     CHits + PutOuts + Assists + Errors + Division
## 
##            Df    AIC
## - PutOuts   3 482.84
## - Errors    3 482.88
## <none>        483.22
## - Division  3 483.58
## - Assists   3 484.10
## - Years     3 484.22
## - Hits      3 484.58
## - Runs      3 485.24
## - Walks     3 486.94
## - CAtBat    3 488.34
## - CHits     3 491.40
## - AtBat     3 492.30
## 
## Step:  AIC=482.84
## SalaryLevel ~ AtBat + Hits + Runs + Walks + Years + CAtBat + 
##     CHits + Assists + Errors + Division
## 
##            Df    AIC
## <none>        482.84
## - Errors    3 482.90
## - Division  3 483.34
## - Years     3 484.15
## - Hits      3 484.57
## - Assists   3 485.19
## - Runs      3 485.27
## - CAtBat    3 487.86
## - Walks     3 489.42
## - CHits     3 491.03
## - AtBat     3 492.58
predicted_reduced <- predict(reduced_model, newdata = Data)


accuracy_reduced <- mean(predicted_reduced == Data$SalaryLevel)
paste("Reduced Model Accuracy:", round(accuracy_reduced * 100, 2), "%")
## [1] "Reduced Model Accuracy: 69.5 %"
set.seed(02122005)
folds <- createFolds(Data$SalaryLevel, k = 5, list = TRUE)

kfold_preds <- rep(NA, nrow(Data))

for (i in 1:5) {
  test_index <- folds[[i]]
  train_data <- Data[-test_index, ]
  test_data <- Data[test_index, ]
  

  model_reduced_cv <- multinom(SalaryLevel ~  AtBat + Hits + Runs + Walks + Years + CAtBat + 
    CHits + Assists + Errors + Division, data = train_data, trace = FALSE)
  
  preds <- predict(model_reduced_cv, newdata = test_data)
  kfold_preds[test_index] <- as.character(preds)
}


actual <- as.character(Data$SalaryLevel)
conf_matrix <- table(Predicted = kfold_preds, Actual = actual)
print(conf_matrix)
##                    Actual
## Predicted           BeenThereDoneThat RookieBudget SolidStarter TooRichToPitch
##   BeenThereDoneThat                34            1           13             15
##   RookieBudget                      0           59           12              2
##   SolidStarter                     21            5           36              4
##   TooRichToPitch                   15            1            3             38
kfold_accuracy <- mean(kfold_preds == actual)
paste("5-Fold CV Accuracy (Reduced Model):", round(kfold_accuracy * 100, 2), "%")
## [1] "5-Fold CV Accuracy (Reduced Model): 64.48 %"
set.seed(05072004) 
n <- nrow(Data)
loo_preds <- rep(NA, n)  

for (i in 1:n) {
 
  train_data <- Data[-i, ]
  test_data <- Data[i, , drop = FALSE] 
  
  model_reduced_loo <- multinom(SalaryLevel ~ AtBat + Hits + Runs + Walks + Years + CAtBat + 
    CHits + Assists + Errors + Division, data = train_data, trace = FALSE)
  
  pred <- predict(model_reduced_loo, newdata = test_data)
  loo_preds[i] <- as.character(pred)
}


actual_loo <- as.character(Data$SalaryLevel)
loo_conf_matrix <- table(Predicted = loo_preds, Actual = actual_loo)
print(loo_conf_matrix)
##                    Actual
## Predicted           BeenThereDoneThat RookieBudget SolidStarter TooRichToPitch
##   BeenThereDoneThat                30            1           10             18
##   RookieBudget                      1           59           13              2
##   SolidStarter                     23            5           39              2
##   TooRichToPitch                   16            1            2             37
loo_accuracy <- mean(loo_preds == actual_loo)
paste("LOOCV Accuracy (Reduced Model):", round(loo_accuracy * 100, 2), "%")
## [1] "LOOCV Accuracy (Reduced Model): 63.71 %"
Overview of Changes With the Inclusion of League, Division, and NewLeague
After reviewing the results and their changes, we realised that it was a bad idea to remove these variables manually, since including the factor variables was better. These factors captured important structural differences among players that only performance metrics could not explain alone. We saw that While they introduced some noise when all variables were included in the full model, after reduction, they increased the model’s stability, and predictive accuracy. This process helped our model take into account of external salary-influencing factors, which reduced unexplained variability and improved prediction accuracy. With doing the project including both sides, we learned from excluding them in the beginning, and we identified an opportunity for improvement, therefore, eventually received a better model with the lowest test error.

Full Model

Training Accuracy
Without Factors: 69.88%
With Factors: 68.73%
AIC
Without Factors: 508.1243
With Factors: 510.6308
Residual Deviance
Without Factors: 406.1243
With Factors: 390.6308
LOO-CV Accuracy
Without Factors: 59.85%
With Factors: 60.62%
5-Fold Accuracy
Without Factors: 61%
With Factors: 59.46%
Reduced Model

Training Accuracy
Without Factors: 67.95%
With Factors: 69.5%
AIC
Without Factors: 478.46
With Factors: 482.84
Residual Deviance
Without Factors: 478.46
With Factors: 482.84
LOO-CV Accuracy
Without Factors: 63.71%
With Factors: 63.71%
5-Fold Accuracy
Without Factors: 61.39%
With Factors: 64.48%
Summary
In this project, our aim was to build a predictive multinomial logistic regression model to classify baseball players into salary levels based on performance data from the Hitters dataset. After pre-processing the dataset by handling missing values and creating four salary level categories (RookieBudget, SolidStarter, BeenThereDoneThat, TooRichToPitch) based on salary quartiles, we conducted exploratory data analysis (EDA) to identify variables most strongly associated with salary differences.

We fitted a full multinomial logistic regression model using all available performance predictors and evaluated their performance by first a full model accuracy and then with 2 Cross-Validation methods for test errors. After receiving the moderate accuracy, as we were required to do so, we looked for methods to improve the model. We chose doing it so by applying backward stepwise selection based on the Akaike Information Criterion (AIC).

After selecting a reduced model containing only the most impactful predictors, we performed 5-Fold and LOO-CV to evaluate its performance, while we made sure the consistency by using the same random seed for data splitting. After the observation of the results we realised that the reduced model achieved a modest but meaningful improvement in predictive accuracy compared to the full model.

Although the increase in accuracy was not huge, we believe that this outcome is consistent with the dataset’s size and the limitations of the available predictors.

On the other hand, we did the project again separately while including the categorical variables, to make sure that we were not ignoring valuable significant variables during prediction. We recognized the importance of these factors during the modeling process which gave us a valuable learning experience. Initially, we overlooked them, but after careful analysis of model performance, we realized that including relevant categorical variables leads to more robust and interpretable models. Therefore, we chose to redo our coding to explicitly include these variables, which allowed us to produce a more complete and statistically better final model.
