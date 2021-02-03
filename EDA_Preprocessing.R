library(tidyverse)
library(data.table)
library(ggplot2)
library(VIM)
library(mice)
library(caret)
library(dplyr)
library(DMwR)
library(e1071)
library(plotly)
library(corrplot)
library(colorspace)
#devtools::install_github("alastairrushworth/inspectdf")
library(inspectdf)
library(treemapify)
library(vcd)

##### Read and Analyze the data #####

# Importing the dataset
bankdata = read.table(
  'data/bank-additional-full.csv',
  sep = ';',
  header = T,
  stringsAsFactors = TRUE
)


# summary of data
dim(bankdata)  #~41 K people's data
# column names
names(bankdata)
# datatypes
str(bankdata) #y is the dependent variable here (term deposit)

# On summarizing we get following insights about the underlying data:
# Median age is 38, and this dataset has no children in it which makes sense.
# Most of the people in this dataset have admin/blue-collar jobs
# Most people have not defaulted or have no values present in the default column, will be best to drop this column.
# p days 999 means client was not previously contacted
# Missing values are present as unknown
summary(bankdata)

##### Data Cleaning #####

# Replace unknown with NA
bankdata[bankdata == "unknown"] <- NA

# As unknown was appearing as a level in some columns, dropping unused levels.
bankdata$marital = droplevels(bankdata$marital)
bankdata$housing = droplevels(bankdata$housing)
bankdata$loan = droplevels(bankdata$loan)
bankdata$education = droplevels(bankdata$education)
bankdata$default = droplevels(bankdata$default)

# Removing default variable as it has 99.9% same or NA observations
table(bankdata$default)
bankdata <- select(bankdata, -c(default))

summary(bankdata)

# Divide age into different brackets - discretize it
bankdata <- bankdata %>% mutate(age_group = cut(
  age,
  breaks = c(17, 19, 34, 59, 99),
  labels = c("Teenagers", "Young Adults", "Adults", "Senior Citizens")
))

# Replace target variables with yes with 1 and no with 0
bankdata$y <- ifelse(bankdata$y == 'yes', 1, 0)
bankdata$y <- as.factor(bankdata$y)
table(bankdata$y)

# pdays - how many days maximum do they keep records of having prior contact with a customer?

plot_ly(alpha = 0.6) %>% add_histogram(x = bankdata$pdays) %>% layout(title = "How many prior contacts")

# It seems they don't maintain contact or perhaps track user after 19 days. Changing this variable below

# Transforming Pdays variable
bankdata$pdays <-
  cut(
    bankdata$pdays,
    breaks = c(0, 200, Inf),
    labels = c('priorContact', 'NoPriorContact')
  )
bankdata$pdays <- as.factor(bankdata$pdays)
table(bankdata$pdays)

# 36548 didn't respond to the campaign and 4640 responded

# Attribute DURATION highly affects the output target (e.g., if duration=0 then y='no').
# Also, the duration is not known before a call is performed. Also, after the end of the call y is
# obviously known. Thus, this input should only be included for benchmark purposes and should be
# discarded if the intention is to have a realistic predictive model.


# Check for Duplicate Rows and remove them
sum(duplicated(bankdata))
bankdata = bankdata %>% distinct


##### Exploratory Data Analysis #####

# Variable Frequency Bar charts
inspect_num(bankdata) %>% show_plot()
# Most common level in categorical columns
inspect_imb(bankdata) %>% show_plot()

# Full distribution of categorical columns
inspect_cat(bankdata) %>% show_plot()


# Correlations

numeric_df <- bankdata %>% dplyr::select(where(is.numeric))
inspect_cor(numeric_df) %>% show_plot()

##### Hypothesis Testing #####

## Do people in certain jobs subscribe more?
# Most people are from admin or blue collar fields
plotdata <-
  bankdata %>% count(y, job) %>%  group_by(job) %>% mutate(freq = n / sum(n)) %>% filter(y ==
                                                                                         1)

ggplot(plotdata,
       aes(fill = job,
           area = n,
           label = job)) +
  geom_treemap() +
  geom_treemap_text(colour = "white",
                    place = "centre") +
  labs(title = "Bank Subscription by Job Title") +
  theme(legend.position = "none")

## Are there certain months /days when people tend to subscribe more?

# As we have seen earlier May has the highest number of calls but it can be seen from the
# graphic below, that in may number of people subscribing is quite less.The highest subscriptions are
# (percentage are in December/October/September)
ggplot(bankdata,
       aes(x = month,
           fill = y)) +
  geom_bar(position = "fill") +
  scale_fill_brewer(palette = "Set2") +
  labs(y = "Percent",
       fill = "Subscription",
       x = "Month",
       title = "Subscription by Month") +
  theme_minimal()

# Thursday sees highest conversions.
ggplot(bankdata,
       aes(x = day_of_week,
           fill = y)) +
  geom_bar(position = "fill") +
  scale_fill_brewer(palette = "Set2") +
  labs(y = "Percent",
       fill = "Subscription",
       x = "Day",
       title = "Subscription by Day") +
  theme_minimal()

## Age group by subscription
# It seems that overall, teenagers and senior citizens have a higher proportion of subscribers
ggplot(bankdata,
       aes(x = age_group,
           fill = y)) +
  geom_bar(position = "fill") +
  scale_fill_brewer(palette = "Set2") +
  labs(y = "Percent",
       fill = "Subscription",
       x = "Age Group",
       title = "Subscription by Age Group") 

## Subscription based on Number of Contacts during the Campaign

ggplot(data=bankdata, aes(x=campaign, fill=y))+geom_histogram(bins=30)+
  ggtitle("Subscription based on Number of Contact during the Campaign")+
  xlab("Number of Contact during the Campaign")+xlim(c(min=1,max=30)) +
  guides(fill=guide_legend(title="Subscription of Term Deposit"))

## Duration vs Subscription
ggplot(bankdata, 
       aes(x = y, 
           y = duration)) +
  geom_violin(fill = "cornflowerblue") +
  geom_boxplot(width = .2, 
               fill = "orange",
               outlier.color = "orange",
               outlier.size = 2) + 
  labs(title = "Duration distribution by subscription")

## Previous Campaign outcome vs Subscription
# Strangely people out of the people who subscribed to the previous campaign few have opted for this one.
# And out of those the previous campaign, success rate for this campaign is higher.
tbl <- xtabs(~poutcome + y , bankdata)

mosaic(tbl, shade = TRUE,
       legend = TRUE,
       labeling_args = list(set_varnames = c(poutcome = "Outcome of Previous Campaign",
                                             y = "Subscribed")),
       set_labels = list(poutcome = c("Success", "Non-Existent",'Failure'),
                         y = c("No", "Yes")),
       main = " Previous Campaign outcome vs Subscription")

##### Data Imputation #####

# How Many Rows Are Completely Missing Values In All Columns - None
all.empty = rowSums(is.na(bankdata)) == ncol(bankdata)
sum(all.empty)

# How Many Rows Contain Missing Data - ~3K rows
sum(!complete.cases(bankdata))

# Missing Value By Variable - education has the highest missing values, followed by loan and marital status

inspect_na(bankdata) %>% show_plot()

## MICE Imputation
init=mice(bankdata,maxit=0)
meth = init$method
predM = init$predictorMatrix
meth[c("housing","loan")]="logreg" 
meth[c("job","marital","education")]="polyreg"
imp <- mice(bankdata, method = meth,  predictorMatrix=predM , m = 2) # Impute data
bank_clean <- complete(imp)
summary(bank_clean)
write.csv(bank_clean, "data/Imputed_data.csv")

