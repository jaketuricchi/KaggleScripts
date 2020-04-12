################################
# Kaggle - Covid19 competition #
################################
#thanks to Dinesh Kumar Talaicha for forecast loop help

#load libraries
library(dplyr)
library(summarytools)
library(stringr)
library(car)
library('caret')
library(anytime)
library(reshape2)
library('ggrepel')
library('directlabels')
library(lubridate)
library(forecast)
library(arm)
library(Metrics)

options(scipen = 999)

#read data
setwd('C:/Users/jaket/Dropbox/Kaggle/Covid19_week4')

train_df<-read.csv('train.csv', stringsAsFactors = F)%>%mutate(set=as.factor('train'))
test_df<-read.csv('test.csv', stringsAsFactors = F)%>%mutate(set=as.factor('test'))
submission <-read.csv('submission.csv')

#check NA and strcuture
str(train_df)
colSums(is.na(train_df))

str(test_df)
colSums(is.na(test_df))

# change formats where needed
train_df$Province_State <- as.character(train_df$Province_State )
train_df$Country_Region <- as.character(train_df$Country_Region )

test_df$Province_State <- as.character(test_df$Province_State)
test_df$Country_Region <- as.character(test_df$Country_Region)

train_df$Date<-anydate(train_df$Date)
test_df$Date<-anydate(test_df$Date)

#replace gaps in provuince state with country
train_df$Province_State <- ifelse(train_df$Province_State == "", 
                                               train_df$Country_Region, train_df$Province_State)
test_df$Province_State<- ifelse(test_df$Province_State == "", 
                                              test_df$Country_Region, test_df$Province_State)

################
#visualisation #
#################

## all plotted together
plotting_df<-train_df%>%
  group_by(Date)%>%dplyr::summarise(total_cases=sum(ConfirmedCases, na.rm = T), 
                                    total_fatalities=sum(Fatalities, na.rm=T))%>%
  melt(., id.var='Date')
ggplot(plotting_df,
       aes(x=Date, y=value, color=variable, group=variable))+geom_point()+geom_line()+theme_minimal()+
  ylab('Individuals')+ ggtitle("Total cases and fatalities worldwide")

## plot all cases by country, can only label a few due to space - lets pick the top 10
top10<-train_df%>%group_by(Country_Region)%>%dplyr::summarise(all_cases=sum(ConfirmedCases, na.rm=T))%>%arrange(desc(all_cases))%>%
  slice(1:10)

plotting_df_country<-train_df%>%subset(Country_Region %in% top10$Country_Region)%>%
  group_by(Date, Country_Region)%>%dplyr::summarise(total_cases=sum(ConfirmedCases, na.rm = T), 
                                                    total_fatalities=sum(Fatalities, na.rm=T))%>%
  melt(., id.var=c('Date', 'Country_Region'))

#total cases by country
top10_cases<-ggplot(filter(plotting_df_country, variable=='total_cases'),
                    aes(x=Date, y=value, color=Country_Region, group=Country_Region))+geom_point()+geom_line()+theme_minimal()+ 
  theme(legend.position = "none") +
  ylab('Individuals')+ ggtitle("Total cases by country")+
  geom_dl(aes(label = Country_Region), method = list(dl.trans(x = x + .2), "last.points"))

#total deaths by country
top10_deaths<-ggplot(filter(plotting_df_country, variable=='total_fatalities'),
                     aes(x=Date, y=value, color=Country_Region, group=Country_Region))+geom_point()+geom_line()+theme_minimal()+ 
  theme(legend.position = "none") +
  ylab('Individuals')+ ggtitle("Total fatalities by country")+
  geom_dl(aes(label = Country_Region), method = list(dl.trans(x = x + .2), "last.points"))

library(gridExtra)
grid.arrange(top10_cases, top10_deaths)

### compare cases vs fatalities in top countries

ggplot(plotting_df_country, aes(x=Country_Region, y=value, color=variable, fill=variable))+
  geom_bar(stat = 'identity', position = 'dodge')+labs(x='Country/Region', y='Individuals',
                                                       title='Cases vs Fatalities in top 10 counties')


unique_state_train <- train_df %>% distinct(Province_State)
state_len <- length(unique_state_train$Province_State)

for(states in 1:state_len){
  
  cat(states, "/", state_len, "Province:", unique_state_train$Province_State[states], "\n")
  
  train <- train_df %>% dplyr::filter(train_df$Province_State == unique_state_train$Province_State[states] ) %>%
    arrange(Date) 
  
  test <- test_df %>% dplyr::filter(test_df$Province_State == unique_state_train$Province_State[states]) %>%
    arrange(Date)
  
  
  ## Confirm Case Forecasting
  
  if (all(train$ConfirmedCases == 0)) {
    
    submission$ConfirmedCases[submission$ForecastId %in% test$ForecastId] <- 0
    submission$Fatalities[submission$ForecastId %in% test$ForecastId] <- 0
    
    next()
    
  } else {
    
    ts.cc.train <- ts(train$ConfirmedCases, start = decimal_date(as.Date("2020-01-22")), frequency = 365.25)
    
    fit.cc <-  Arima(ts.cc.train, order = c(2,2,2), seasonal = list(order = c(1,1,0), period = 12), method = "ML",
                     optim.method = "BFGS")
    
    
    forecast.cc <- forecast(fit.cc, h=43, level=c(99.5))
    
    for(i in 1:43){
      test$ConfirmedCases[i] <- ifelse(forecast.cc$upper[i] > 0, as.numeric(round(forecast.cc$upper[i])), 0)
    }
    
    submission$ConfirmedCases[submission$ForecastId %in% test$ForecastId] <-
      ifelse(forecast.cc[["upper"]] > 0, round(forecast.cc[["upper"]]), 0)
    
    rm(forecast.cc)
  }
  
  ### Fatalities Forecasting
  
  if (all(train$Fatalities == 0)) {
    
    submission$Fatalities[submission$ForecastId %in% test$ForecastId] <- 0
    next()
    
  } else {
    
    fit.fat <- train(form=as.formula("Fatalities ~ ConfirmedCases"),
                     data = train,
                     method = "bayesglm",
                     trControl=trainControl(method="repeatedcv", number=8, repeats=5))
    
    preds <- predict(fit.fat, test)
    test$Fatalities <- ifelse(preds > 0, as.numeric(round(preds)), 0)
    submission$Fatalities[submission$ForecastId %in% test$ForecastId] <- ifelse(preds > 0, round(preds), 0)
    
    preds2 <- predict(fit.fat, train)
    preds2[preds2 < 0] <- 0
    cat("RMSLE :-  ", round(rmsle(train$Fatalities, preds2), 4), "\n\n")
    
    rm(preds)
    rm(preds2)
  }
}

write.csv(submission, 'Kaggle_covid_forecast_120419', row.names = F)


#plot forecasts - all countries
test_df2<-merge(test_df, submission, by='ForecastId')
all_df<-merge(test_df2, train_df, all=T)%>%subset(., select=c(-Id, -ForecastId, -Province_State))
all_plot_fc<-all_df%>%group_by(Date, set)%>%dplyr::summarise(all_cases=sum(ConfirmedCases),
                                                             all_deaths=sum(Fatalities))
all_df_melt<-melt(all_plot_fc, id.vars=c('set', 'Date'))
ggplot(all_df_melt, aes(x=Date, y=value, col=variable, group=variable, fill=set))+
  geom_smooth(se=F)+geom_vline(xintercept=as.Date('2020-04-02'), linetype='dotted')+theme_minimal()

#plot forecasts - top 10 countries
plot_fc<-all_df%>%group_by(Date, set, Country_Region)%>%dplyr::summarise(all_cases=sum(ConfirmedCases),
                                                             all_deaths=sum(Fatalities))
top10_fc<-plot_fc%>%group_by(Country_Region)%>%dplyr::summarise(all_cases=sum(all_cases))%>%
  arrange(desc(all_cases))%>%slice(1:10)

fc_cases<-ggplot(subset(plot_fc, Country_Region %in% top10_fc$Country_Region), 
       aes(x=Date, y=all_cases, col=Country_Region))+geom_smooth(se=F)+ theme_minimal()+
  theme(legend.position = "none") +
  ylab('Individuals')+ ggtitle("Total cases by country")+
  geom_dl(aes(label = Country_Region), method = list(dl.trans(x = x + .2), "last.points"))+
  geom_vline(xintercept=as.Date('2020-04-02'), linetype='dotted')

fc_deaths<-ggplot(subset(plot_fc, Country_Region %in% top10_fc$Country_Region), 
       aes(x=Date, y=all_deaths, col=Country_Region))+geom_smooth(se=F)+ theme_minimal()+
  theme(legend.position = "none") +
  ylab('Individuals')+ ggtitle("Total deaths by country")+
  geom_dl(aes(label = Country_Region), method = list(dl.trans(x = x + .2), "last.points"))+
  geom_vline(xintercept=as.Date('2020-04-02'), linetype='dotted')

grid.arrange(fc_cases, fc_deaths)


