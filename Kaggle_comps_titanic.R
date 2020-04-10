################################
# Kaggle - Titanic competition #
################################

#load libraries
library(dplyr)
library(summarytools)
library('summarytools')
library(stringr)
library(car)
library('caret')
library(imputeTS)


#read data
setwd('C:/Users/jaket/Dropbox/Kaggle/Titanic')
#setwd('~/Dropbox/Kaggle/Titanic')
train_df<-read.csv('train.csv')%>%mutate(set='train')
test_df<-read.csv('test.csv')%>%mutate(Survived=NA)%>%mutate(set='test')
gender_df<-read.csv('gender_submission.csv')

all_df<-bind_rows(train_df, test_df)

#examine data
str(all_df)
summary(all_df)
view(dfSummary(all_df)) #extensive summary table with distribution plots

#note: 40.9% survived

#count NAs
all_df %>%
  select(everything()) %>%  
  summarise_all(funs(sum(is.na(.)))) #only age has missingness

#dealing with normality, skewness etc
# from dfSummary, fare is highly skewed, take log..
all_df$log_fare=log(all_df$Fare)
hist(all_df$log_fare) #now little skew
#all other vars are normal.


##################
# pre-processing #
##################

#### IMPUTATION
#impute age before we begin to generate new features so age can only be a function of certain data
library('mice')
md.pattern(all_df) #visualise missingness

#default imputation from mice using random forest
age_imputation_rf<- mice(all_df[,c(1,3,5:8,10)],m=5,maxit=50, seed=500)

#examine the imputation vs observed data
densityplot(age_imputation_rf) #observed = blue, red= each imputation
stripplot(age_imputation_rf, pch = 20, cex = 1.2)

#take mean of all imputations and round
complete_age<-mice::complete(age_imputation_rf, action='all')%>%bind_rows(.)%>%
  group_by(PassengerId)%>%
  dplyr::summarise(Age=round(mean(Age)))

#merge imputed weight back into train df
all_df$Age<-complete_age$Age

all_df$Fare<-na_mean(all_df$Fare) #one missing cell here, replace with mean

#fill gaps on other vars
all_df$Embarked <- replace(all_df$Embarked, which(is.na(all_df$Embarked)), 'S')


##### FEATURE GENERATION

#age groups generation
all_df$Age_group<-as.factor(ifelse(all_df$Age >0 & all_df$Age <12, 1,
                                     ifelse(all_df$Age >= 12 & all_df$Age <18,2,
                                            ifelse(all_df$Age >= 18 & all_df$Age <60, 3, 4))))
summary                              
#cabin details
all_df$Cabin<-as.character(all_df$Cabin)
all_df$cabin_letter_sort<-gsub("\\d+", "", all_df$Cabin) #this var contains letter and n(letter)
all_df$number_of_cabins<-str_count(all_df$cabin_letter_sort, all_df$cabin_letter) #extract n(letter)

#NA = no cabis, replace NA with 0
all_df$number_of_cabins[is.na(all_df$number_of_cabins)] <- 0
all_df$Survived<-as.factor(all_df$Survived)

#interactions between age, sex and class are likely to impact saviour
all_df<-all_df%>%mutate(Sex_no=ifelse(Sex=='male', 1, 2),
                            age_class=Age * Pclass,
                            age_sex=Sex_no*Age,
                            sex_class=Sex_no*Pclass,
                            age_class_sex=Sex_no *Age*Pclass)

#extract info from name
all_df$title <- gsub("^.*, (.*?)\\..*$", "\\1", all_df$Name)

#we can get an idea of family
all_df$NFamMembers <-as.numeric(all_df$SibSp + all_df$Parch + 1)
all_df$FamSize<-as.factor(ifelse(all_df$NFamMembers==1, 'Single',
                                 ifelse(all_df$NFamMembers >= 2 & all_df$NFamMembers < 5, 'Small', 'Big')))

#manual sort these where there's small n
all_df$title[all_df$title == 'Mlle']        <- 'Miss' 
all_df$title[all_df$title == 'Ms']          <- 'Miss'
all_df$title[all_df$title == 'Mme']         <- 'Mrs' 
all_df$title[all_df$title == 'Lady']          <- 'Miss'
all_df$title[all_df$title == 'Dona']          <- 'Miss'
all_df$title[all_df$title == 'Capt']        <- 'UpperTitle' #upper class sounding titles
all_df$title[all_df$title == 'Col']        <- 'UpperTitle' 
all_df$title[all_df$title == 'Major']   <- 'UpperTitle'
all_df$title[all_df$title == 'Dr']   <- 'UpperTitle'
all_df$title[all_df$title == 'Rev']   <- 'UpperTitle'
all_df$title[all_df$title == 'Don']   <- 'UpperTitle'
all_df$title[all_df$title == 'Sir']   <- 'UpperTitle'
all_df$title[all_df$title == 'the Countess']   <- 'UpperTitle'
all_df$title[all_df$title == 'Jonkheer']   <- 'UpperTitle'

#extract information from ticket numbers
ticket.unique <- rep(0, nrow(all_df))
tickets <- unique(all_df$Ticket)

for (i in 1:length(tickets)) {
  current.ticket <- tickets[i]
  party.indexes <- which(all_df$Ticket == current.ticket)
  
  for (k in 1:length(party.indexes)) {
    ticket.unique[party.indexes[k]] <- length(party.indexes)
  }
}

all_df$ticket.unique <- ticket.unique
all_df$ticket.size[all_df$ticket.unique == 1]   <- 'Single'
all_df$ticket.size[all_df$ticket.unique < 5 & all_df$ticket.unique>= 2]   <- 'Small'
all_df$ticket.size[all_df$ticket.unique >= 5]   <- 'Big'


#############################
# exploratory data analysis #
#############################
#survival by age group
ggplot(data=filter(all_df, !is.na(Survived)), aes(Age_group, fill=Survived))+
  geom_bar(position="fill")+theme_linedraw()

#survial by social class
ggplot(data=filter(all_df, !is.na(Survived)), aes(Pclass, fill=Survived))+
  geom_bar(position="fill")+theme_linedraw()

#survial by Sex
ggplot(data=filter(all_df, !is.na(Survived)), aes(Sex, fill=Survived))+
  geom_bar(position="fill")+theme_linedraw()

#check out some generated variables - title
ggplot(data=filter(all_df, !is.na(Survived)), aes(title, fill=Survived))+
  geom_bar(position="fill")+theme_linedraw()

#family size
ggplot(data=filter(all_df, !is.na(Survived)), aes(FamSize, fill=Survived))+
  geom_bar(position="fill")+theme_linedraw()


#mosiac plots
tbl_mosaic <- all_df %>%
  filter(!is.na(Survived)) %>%
  select(Survived, Pclass, Sex, Age_group, title, Embarked, 'FamSize') %>%
  mutate_all(as.factor)

library(vcd)
mosaic(~Pclass+Sex+Survived, data=tbl_mosaic, shade=TRUE, legend=TRUE)
mosaic(~Pclass+Age_group+Survived, data=tbl_mosaic, shade=TRUE, legend=TRUE)



####################
# model generation #
####################
#ensure correct str
all_df$Sex_no<-as.numeric(all_df$Sex_no)
all_df$Pclass<-as.numeric(all_df$Pclass)
all_df$Parch<-as.numeric(all_df$Parch)
all_df$Age_group<-as.numeric(all_df$Age_group)
all_df$SibSp<-as.numeric(all_df$SibSp)
all_df$ticket.unique<-as.numeric(all_df$ticket.unique)
all_df$ticket.size<-as.factor(all_df$ticket.size)

#ensure correct levels
levels(all_df$Survived) <- c("No", "Yes")

#select potentially useful vars
train_df<-all_df%>%filter(set=='train')%>%
  subset(., 
                 select=c(-PassengerId, -Ticket, -Cabin,  -Name, -Sex,-cabin_letter_sort,
                          -log_fare, -title, -set, -Embarked))

#test set
test_df2<-filter(all_df, set=='test')%>%
  subset(select=c(-PassengerId, -Ticket, -Cabin,  -Name, -Sex, -cabin_letter_sort,
                  -log_fare, -title, -set, -Embarked))

#check str
str(train_df)

#we'll test some basic ML models.

###################
# random forest  #
##################
library('randomForest')
set.seed(500)
objControl <- trainControl(method='repeatedcv', number=3, 
                           returnResamp='none', 
                           classProbs = TRUE,
                           search='random')
grid <- expand.grid(.mtry=c(1:15))
RF1 <- train(Survived~., data=train_df, seed=500,
                  method='rf',
                  trControl=objControl,
             tuneGrid=grid,
             metric='Accuracy',
                  preProc = c("center", "scale"))

RF1
plot(RF1)
plot(varImp(RF1))

#test on testing data
RF.pred=predict(RF1,newdata = test_df2)
RF.pred2=cbind.data.frame(as.numeric(RF.pred), test_df$PassengerId)

RF.pred2$Survived=ifelse(RF.pred2[1]==1, 0, 1)
RF.pred2<-RF.pred2[-1]
colnames(RF.pred2)[1]<-'PassengerId'
write.csv(RF.pred2, 'Kaggle_titanic_RF_V2.csv', row.names = F)

#######################
# logistic regression #
#######################


summary(GLM_logit <- glm(Survived ~ ., family = binomial(link=logit),
               data = train_df))
confint(GLM_logit)

#predict train
pred.train.glm <- predict(GLM_logit, newdata=train_df, type='response')
table(train_df$Survived, pred.test.glm>0.5)
(490+247)/(490+58+95+247) #82.8% on train

#predict test
pred.test.glm <- predict(GLM_logit, newdata=test_df, type='response')
predictions_glm<-round(pred.test.glm)
table(gender_df$Survived, pred.test.glm>0.5)
(252+140)/(252+140+12+13) #94% accuracy on test

#sort predictions for kaggle submit
predictions_glm<-data.frame(predictions_glm)%>%mutate(PassengerId = test_df$PassengerId)
predictions_glm<-predictions_glm[,c(2,1)]
colnames(predictions_glm)[2]<-'Survived'
write.csv(predictions_glm, 'Kaggle_titanic_V1.csv', row.names = F)

