
#ubiqum module 2 task 2 predicting brand preferences
#complete customer survey will be used to predict brand preference of the incomplete survey

# LIBRARIES ####
library(readr)
library(dplyr)
library(caret)
library(ggplot2)
library(data.table)
library(plotly)
library(stats)

# IMPORT DATA ####
mydata <- fread("Ubiqum m2 CompleteResponses.csv")
incomplete <- fread("Ubiqum m2 SurveyIncomplete.csv")
survey.key <- fread("ubiqum module2 2 survey key.csv")

# PREPROCESS DATA ####
#change from 0/1 to band names
mydata$brand[mydata$brand == "0"] <- "Acer"
mydata$brand[mydata$brand == "1"] <- "Sony"

#delete incomplete brand row and factorize car/zipcode
incomplete$brand <- NULL
incomplete$car <- as.factor(incomplete$car)
incomplete$zipcode <- as.factor(incomplete$zipcode)

#factorize car/zipcode/brand
mydata$brand <- as.factor(mydata$brand)
mydata$car <- as.factor(mydata$car)
mydata$zipcode <- as.factor(mydata$zipcode)

# VISUALIZE DATA ####
#histogram salary
hist.salary <- ggplot(mydata, aes(x= salary, fill=brand )) +
  geom_histogram(bins = 30 )
hist.salary + 
  geom_vline(aes(xintercept=mean(salary))) +
  ggtitle("Complete Response Brand - Salary") +
  ylab("Count") +
  xlab("Salary")+
  scale_y_continuous(labels = format_format(scientific = FALSE, big.mark = ".", decimal.mark = ",")) +
  scale_x_continuous(labels = format_format(scientific = FALSE, big.mark = ".", decimal.mark = ","))

#histogram credit
hist.credit <- ggplot(mydata, aes(x= credit, fill=brand )) +
  geom_histogram(binwidth = 20000 ) 
hist.credit + 
  ggtitle("Complete Response Brand - Credit") +
  ylab("Count") +
  xlab("Credit")+
  geom_vline(aes(xintercept=mean(credit))) +
  scale_y_continuous(labels = format_format(scientific = FALSE, big.mark = ".", decimal.mark = ",")) +
  scale_x_continuous(labels = format_format(scientific = FALSE, big.mark = ".", decimal.mark = ","))

#histogram age
hist.age <- ggplot(mydata, aes(x= age, fill=brand )) + 
  geom_histogram(binwidth = 5)
hist.age + 
  ggtitle("Complete Response Brand - Age") +
  ylab("Count") +
  xlab("Age")+
  geom_vline(aes(xintercept=mean(age)))

#barchart car
bar.car <- ggplot(mydata, aes(x= car, fill=brand )) +
  geom_bar() 
bar.car +
  ggtitle("Complete Response Brand - Car") +
  ylab("Count") +
  xlab("Car")

#barchart education level
bar.elevel <- ggplot(mydata, aes(x= elevel, fill=brand )) +
  geom_bar() 
bar.elevel +
  ggtitle("Complete Response Brand - Education Level") +
  ylab("Count") +
  xlab("Education Level")

#barchart zip code
bar.zipcode <- ggplot(mydata, aes(x= zipcode, fill=brand )) +
  geom_bar()
bar.zipcode +
  ggtitle("Complete Response Brand - ZIP Code") +
  ylab("Count") +
  xlab("ZIP Code")

#bar chart brand
bar.brand <- ggplot(mydata, aes(x= brand, fill=brand)) +
  geom_bar() 
bar.brand +
  ggtitle("Complete Response Brands") +
  ylab("Count") +
  xlab("Brand")

#scatter plot brand with age and salary
scatter.age.salary <- ggplot(mydata, aes(x= age, y=salary, col = brand)) +
  geom_point()
scatter.age.salary +
  ggtitle("Complete Response Brand by Age and Salary") +
  ylab("Salary") +
  xlab("Age") +
  scale_y_continuous(labels = format_format(scientific = FALSE, big.mark = ".", decimal.mark = ",")) +
  scale_x_continuous(labels = format_format(scientific = FALSE, big.mark = ".", decimal.mark = ","))
ggplotly(scatter.age.salary)

#scatter plot brand with age and credit
scatter.age.credit <-
  ggplot(mydata, aes(x = age, y = credit, col = brand)) +
  geom_point()
scatter.age.credit +
  ggtitle("Complete Response Brand by Age and Credit") +
  ylab("Credit") +
  xlab("Age") +
  scale_y_continuous(labels = format_format(scientific = FALSE, big.mark = ".", decimal.mark = ",")) +
  scale_x_continuous(labels = format_format(scientific = FALSE, big.mark = ".", decimal.mark = ","))
ggplotly(scatter.age.credit)

# TRAIN TEST SETS ####
set.seed(134)
train.index <- createDataPartition(mydata$brand, p = 0.75, list = FALSE)
train <- mydata[train.index, ]
test <- mydata[-train.index, ]
trctrl <- trainControl(method = "cv", number = 5)

# DECISION TREE C5.0 ####
model.c5 <- train(brand ~., 
                  data = train,
                  method = "C5.0",
                  trControl=trctrl,
                  preProcess = c("center", "scale"),
                  tuneLength=2)

prediction.c5 <- predict(model.c5, test)
test$prediction.c5 <- prediction.c5
validation.c5 <- confusionMatrix(table(test$prediction.c5, test$brand))
validation.c5$overall["Accuracy"]


# RANDOM FOREST ####
model.rf <- train(brand ~., 
                  data = train,
                  method = "rf",
                  trControl=trctrl,
                  tuneGrid=expand.grid(mtry=c(5)),
                  preProcess = c("center", "scale"))

prediction.rf <- predict(model.rf, test, type="raw")
test$prediction.rf <- prediction.rf
validation.rf <- confusionMatrix(table(test$prediction.rf, test$brand))
validation.rf$overall["Accuracy"]


# MODEL ANALYSIS ####
varImp(model.c5)
varImp(model.rf)
postResample(test$prediction.c5,test$brand)
postResample(test$prediction.rf,test$brand)
model.c5
model.rf

# RE-TRAIN MODELS with only salary and age ####
#c5
re.model.c5 <- train(brand ~ salary + age, 
                     data = train,
                     method = "C5.0",
                     trControl=trctrl,
                     preProcess = c("center", "scale"),
                     tuneLength=2)

re.prediction.c5 <- predict(re.model.c5, test)
test$re.prediction.c5 <- re.prediction.c5
re.validation.c5 <- confusionMatrix(table(test$re.prediction.c5, test$brand))
re.validation.c5$overall["Accuracy"]

#rf
re.model.rf <- train(brand ~salary + age, 
                     data = train,
                     method = "rf",
                     trControl=trctrl,
                     tuneGrid=expand.grid(mtry=5),
                     preProcess = c("center", "scale"))

re.prediction.rf <- predict(re.model.rf, test, type="raw")
test$re.prediction.rf <- re.prediction.rf
re.validation.rf <- confusionMatrix(table(test$re.prediction.rf, test$brand))
re.validation.rf$overall["Accuracy"]

# REDO MODEL ANALYSIS ####
postResample(test$prediction.c5,test$brand)
postResample(test$re.prediction.c5,test$brand)

postResample(test$prediction.rf,test$brand)
postResample(test$re.prediction.rf,test$brand)

# PREDICTION ON INCOMPLETE SURVEY ####
predicted.brand.c5 <- predict(re.model.c5, incomplete)
incomplete$brand.c5 <- predicted.brand.c5
summary(incomplete$brand.c5)

predicted.brand.rf <- predict(re.model.rf, incomplete)
incomplete$brand.rf <- predicted.brand.rf
summary(incomplete$brand.rf)

# COMBINE COMPLETE AND INCOMPETE SURVEYS ####

incomplete2 <- incomplete
incomplete2$brand.rf<-NULL
incomplete2$brand<-incomplete2$brand.c5
incomplete2$brand.c5<-NULL

alldata<-data.frame()
alldata<-rbind(mydata,incomplete2)

#scatterplot brand with age and salary on combined survey
scatter.age.salary.combined <- ggplot(alldata, aes(x= age, y=salary, col = brand)) +
  geom_point()
scatter.age.salary.combined +
  ggtitle("Combined Data Brand by Age and Salary") +
  ylab("Salary") +
  xlab("Age") +
  scale_y_continuous(labels = format_format(scientific = FALSE, big.mark = ".", decimal.mark = ",")) +
  scale_x_continuous(labels = format_format(scientific = FALSE, big.mark = ".", decimal.mark = ","))
ggplotly(scatter.age.salary)

#histogram age brand 
hist.age <- ggplot(alldata, aes(x= age, fill=brand )) +
  geom_histogram(binwidth = 5) #(binwidth = 1000), color=I("red")
hist.age + 
  ggtitle("Combined Data Brand - Age") +
  ylab("Salary") +
  xlab("Age") +
  geom_vline(aes(xintercept=mean(age)))


# ERROR VISUALISATION #####
test$error.c5[test$re.prediction.c5 != test$brand] <- "Test Error"
df <- test[test$error.c5 == "Test Error"]

#scatterplot test prediction error with age and salary
scatter.error.age.salary <- ggplot(df, aes(x = age, y = salary, col = error.c5)) +
  geom_point()
scatter.error.age.salary +
  ggtitle("Test Error by Age and Salary") +
  ylab("Salary") +
  xlab("Age") +
  scale_y_continuous(labels = format_format(scientific = FALSE, big.mark = ".", decimal.mark = ",")) +
  scale_x_continuous(labels = format_format(scientific = FALSE, big.mark = ".", decimal.mark = ","))
ggplotly(scatter.error.age.salary)

# END ####



