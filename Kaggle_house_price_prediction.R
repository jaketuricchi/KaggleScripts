###############################################
# Kaggle - House price prediction competition #
###############################################

#load libraries
library(dplyr)
library(summarytools)
library(stringr)
library(car)
library('caret')
library(imputeTS)
library('DescTools')
library(forcats)
library(purrr)
library(reshape2)
library(gridExtra)
library(data.table)
library(xgboost)
library('mlr')
library('Ckmeans.1d.dp') #required for ggplot clustering

options(scipen=999)


#read data
#setwd('C:/Users/jaket/Dropbox/Kaggle/House_prices')
setwd("~/Dropbox/Kaggle/House_prices")


train_df<-read.csv('train.csv', stringsAsFactors = F)%>%mutate(set='train')
test_df<-read.csv('test.csv', stringsAsFactors = F)%>%mutate(set='test')
sample_sub<-read.csv('sample_submission.csv')

df<-bind_rows(train_df, test_df)



########################
#consider missingness ##
#######################
str(df)
initial_missing<-data.frame(colSums(is.na(df)))
colnames(initial_missing)[1]<-'missing'
top_missing<-initial_missing%>%mutate(variable=rownames(.))%>%arrange(desc(missing))%>%print(rownames(.)) #rough idea of amount of missingness

#replace missingness
df$PoolQC[is.na(df$PoolQC)] <- 'None' # PoolQC: pool quality therefore NA = no pool
df$MiscFeature[is.na(df$MiscFeature)] <- 'None' #MiscFeature: Miscellaneous feature not covered in other categories
df$Alley[is.na(df$Alley)] <- 'None' #Alley: Type of alley access to property
df$Fence[is.na(df$Fence)] <- 'None' #Fence: Fence quality
df$FireplaceQu[is.na(df$FireplaceQu)] <- 'None' #FireplaceQu: Fireplace quality
df$LotFrontage[is.na(df$LotFrontage)] <- '0' # Linear feet of street connected to property
df$GarageYrBlt[is.na(df$GarageYrBlt)] <- '0' #GarageYrBlt: Year garage was built
df$GarageFinish[is.na(df$GarageFinish)] <- 'None' #GarageFinish: Interior finish of the garage
df$GarageQual[is.na(df$GarageQual)] <- 'None' #GarageQual: Garage quality
df$GarageCond[is.na(df$GarageCond)] <- 'None' #GarageCond: Garage condition
df$GarageType[is.na(df$GarageType)] <- 'None' #GarageType: Garage location
df$BsmtCond[is.na(df$BsmtCond)] <- 'None' #BsmtCond: Evaluates the general condition of the basement
df$BsmtExposure[is.na(df$BsmtExposure)] <- 'None' #BsmtExposure: Refers to walkout or garden level walls
df$BsmtQual[is.na(df$BsmtQual)] <- 'None' #BsmtQual: Evaluates the height of the basement
df$BsmtFinType2[is.na(df$BsmtFinType2)] <- 'None' #BsmtFinType2: Rating of basement finished area (if multiple types)
df$BsmtFinType1[is.na(df$BsmtFinType1)] <- 'None' #BsmtFinType1: Rating of basement finished area
df$MasVnrType[is.na(df$MasVnrType)] <- 'None' #MasVnrType: Masonry veneer type
df$MasVnrArea[is.na(df$MasVnrArea)] <- '0' 

#because the others are small (<5 missing), we will mode impute
contains_NA=names(df[, colSums(is.na(df)) > 0 ])[-17] #get names of columns with NA, dont include sale price

#calculate mode
Mode <- function(x) {
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}
anyNA(x)
#impute missing with mode
mode_impute<-function(x){
  x<-as.vector(x)
  if (anyNA(x)==T) { #if there are NAs in the column
    mode<-Mode(x)
    x[is.na(x)]<-mode ## fill these NAs with the mode
    } else {
  return(x) # else do nothing
    }
}

df_filled<-as.data.frame(sapply(df, mode_impute))
colSums(is.na(df_filled)) #now no NAs apart from the outcome.

#ensure formatting is correct in accordance with variable description (sometimes numbers are factors etc)
#need to do this by eye.
categorical_vars<-as.character(data.frame(names(df_filled))[c(-1,-4,-5, -20, -21, -27, -35, -37, -38, -39, -44, -45, -46, -47,
                                                  -60,-62, -63, -67, -68, -69, -70, -71, -72, -76, -77, -78, -81),])

numeric_vars<-as.character(data.frame(names(df_filled))[c(1,4,5, 20, 21, 27, 35, 37, 38, 39, 44, 45, 46, 47,
                                             60,62, 63, 67, 68, 69, 70, 71, 72, 76, 77, 78, 81),])
names(df_filled)

#change the structures
df_filled[numeric_vars]<-lapply(df_filled[numeric_vars], as.numeric)
df_filled[categorical_vars]<-lapply(df_filled[categorical_vars], factor)
df_filled$SalePrice<-df$SalePrice

#categorical variables now need to be ordered to impose a meaningful relationship with price
#for this we need to split into train/test for now

train_df_new<-filter(df_filled, set=='train')

categorical_df_train<-train_df_new[categorical_vars] #create a categorical only df

categorical_sort<-list()
i=1
for (i in 1:ncol(categorical_df_train)){ #this loop takes each row and reorders the factors by the highest SalePrice from traindfnew
  cat_var<-categorical_df_train[i]
  cat_SP<-cbind(cat_var, train_df_new$SalePrice)
  colnames(cat_SP)<-c('var', 'SalePrice')
  cat_SP<-cat_SP%>%mutate(var=fct_reorder(var, SalePrice))
  categorical_sort[[i]]<-cat_SP[1]
}

categorical_df2<-bind_cols(categorical_sort)%>%bind_cols(subset(train_df_new, select=c(Id)), .) #add Id var back in
colnames(categorical_df2)[2:56]<-colnames(categorical_df_train)[1:55]


#remove factors with 1 level
single_level_factors<-names(categorical_df2[, sapply(categorical_df2, nlevels) < 2])
categorical_df3<-categorical_df2%>%dplyr::select(-single_level_factors, Id)#select, keep Id to merge

#merge back into full df
train_df2<-merge(train_df_new[numeric_vars], categorical_df3, by='Id')%>%
  droplevels()

#quick boxplot to check factors ordered correctly - any var shows positive relationship with
boxplot(train_df2$SalePrice~train_df2$RoofStyle)


#####################
# outlier detection #
#####################
#http://r-statistics.co/Outlier-Treatment-With-R.html 
#lets start by examining Cook's distance
outlier_lm<-lm(SalePrice~., data=subset(train_df2, select=c(-Id, -set)))

cooksd <- cooks.distance(outlier_lm)
#In general use, those observations that have a cook's distance greater than 4 times the mean may be classified as 
#influential. This is not a hard boundary.

#plot outliers               
plot(cooksd, pch="*", cex=2, main="Influential Obs by Cooks distance")  # plot cook's distance
abline(h = 4*mean(cooksd, na.rm=T), col="red")  # add cutoff line
text(x=1:length(cooksd)+1, y=cooksd, labels=ifelse(cooksd>4*mean(cooksd, na.rm=T),names(cooksd),""), col="red")  # add labels

#extract outlier row numbers
outliers<-data.frame(ifelse(cooksd>4*mean(cooksd, na.rm=T), 1, 0))%>%
  mutate(row=1:nrow(.))
colnames(outliers)[1]<-'out'
outliers<-filter(outliers, out==1)
outliers<-as.numeric(outliers$row)

#subset data to remove outliers
train_df3<-train_df2[-outliers,]
colSums(is.na(train_df2))
setdiff(single_level_factors, names(test_df_new))

#add testing data back in for feature generation: also we need to ensure fct levels are the same.
test_df_new<-filter(df_filled, set=='test')

df2<-merge(train_df2, test_df_new, all=T)
df2<-df2%>%dplyr::select(-single_level_factors, Id)#select, keep Id to merge

#now we have ensured that all fct levels remain sorted after merging. Check with boxplot:
boxplot(df2$SalePrice~df2$FireplaceQu) #all factors still ordered after binding

########################
# data visualisation #
########################

#first, lets look at the correlation of all numeric variables
library(corrplot)
cor_mat<-as.matrix(subset(df2, select=c(numeric_vars))%>%dplyr::select(-Id, -GarageCars))
cor_mat2 = cor_mat[,colSums(cor_mat) > 0] #remove columns with all 0
cor<-cor(cor_mat2)
cor<-cor[-20,-20]
corrplot(cor,method='square')

#living area on sale price
df2%>%subset(., select=c(X1stFlrSF, X2ndFlrSF, GrLivArea, SalePrice))%>%
  melt(id.var='SalePrice')%>%ggplot(., aes(x=value, y=SalePrice, col=variable))+
  geom_point()+geom_smooth(method='lm', se=F)+theme_light()+
  labs(title='Associations between square feet of living area and sale price',
       x='Square Feet')

#year built and remodelled on sale price
df2%>%subset(., select=c(YearBuilt, YearRemodAdd, SalePrice))%>%
  melt(id.var='SalePrice')%>%ggplot(., aes(x=value, y=SalePrice, col=variable))+
  geom_point()+geom_smooth(method='lm', se=F)+theme_light()+
  labs(title='Associations between square feet of living area and sale price',
       x='Square Feet')

#lot variables and remodelled on sale price
df2%>%subset(., select=c(LotArea, LotFrontage, SalePrice))%>%
  mutate(LotFrontage_1000=LotFrontage*1000)%>%dplyr::select(-LotFrontage)%>%
  melt(id.var='SalePrice')%>%ggplot(., aes(x=value, y=SalePrice, col=variable))+
  geom_point()+geom_smooth(method='lm', se=F)+theme_light()+
  labs(title='Associations between lot areas and sale price',
       x='Square Feet')

#garage variables and sale price
gartype_plot<-ggplot(df2, aes(x=GarageType, y=SalePrice))+geom_boxplot()+theme_light()+
  labs(title='Association between Garage Type and sale price')
garfinish_plot<-ggplot(df2, aes(x=GarageFinish, y=SalePrice))+geom_boxplot()+theme_light()+
  labs(title='Association between Garage Finish and sale price')
garquality_plot<-ggplot(df2, aes(x=GarageQual, y=SalePrice))+geom_boxplot()+theme_light()+
  labs(title='Association between Garage Quality and sale price')
garcond_plot<-ggplot(df2, aes(x=GarageCond, y=SalePrice))+geom_boxplot()+theme_light()+
  labs(title='Association between Garage Condition and sale price')

grid.arrange(gartype_plot, garfinish_plot, garquality_plot, garcond_plot)

#Rooms and sale price
bedrooms_plot<-ggplot(df2, aes(x=BedroomAbvGr, y=SalePrice))+geom_boxplot()+theme_light()+
  labs(title='Association between bedrooms and sale price')
baths_plot<-ggplot(filter(df2, !is.na(SalePrice)), aes(x=FullBath, y=SalePrice))+geom_boxplot()+
  theme_light()+labs(title='Association between bedrooms and sale price')
kitchen_plot<-ggplot(filter(df2, !is.na(SalePrice)), 
                     aes(x=KitchenAbvGr, y=SalePrice))+geom_boxplot()+
  theme_light()+labs(title='Association between kitchens and sale price')
total_rm_plot<-ggplot(filter(df2, !is.na(SalePrice)), 
                      aes(x=as.numeric(as.character(TotRmsAbvGrd)), y=SalePrice))+
  geom_smooth(method='lm')+geom_point()+
  theme_light()+labs(title='Association between total rooms and sale price')

grid.arrange(bedrooms_plot, baths_plot, kitchen_plot, total_rm_plot)

#######################
# feature engineering #
#######################
#here we rely on variable knowledge and information

#we start by interacting clusters of variables
#these include: lot, garage, Type (bldgType, HouseStyle, MSSubClass),
#Quality(overall, cond), year (built, remod), roof, Exterior/masonry,
#basement, heating/AC, size (1stflrsq etc), baths, fireplace, 
#deck/porch, pool, sale (type/condition)
#

#make sure we dont include tha factors we remove due to single level
single_level_factors

vars_list<-list()
vars_list[[1]]<-c('LotFrontage', 'LotArea', 'LotConfig')
vars_list[[2]]<-c('BldgType', 'HouseStyle', 'MSSubClass')
vars_list[[3]]<-c('OverallQual', 'OverallCond', 'LowQualFinSF',
                'GrLivArea')
vars_list[[4]]<-c('YearBuilt', 'YearRemodAdd', 'GarageYrBlt')
vars_list[[5]]<-c('RoofStyle', 'RoofMatl')
vars_list[[6]]<-c('BsmtQual', 'BsmtCond', 'BsmtExposure',
                 'BsmtFinType1', 'BsmtFinSF1',
                 'BsmtFinType2', 'TotalBsmtSF')
vars_list[[7]]<-c('Heating', 'HeatingQC')
vars_list[[8]]<-c('X1stFlrSF', 'X2ndFlrSF', 'LowQualFinSF', 
             'GrLivArea')
vars_list[[9]]<-c('Fireplaces', 'FireplaceQu')
vars_list[[10]]<-c('GarageType', 'GarageYrBlt',
               'GarageCars', 'GarageArea', 'GarageQual')
vars_list[[11]]<-c('WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch',
              'X3SsnPorch', 'ScreenPorch')

#interact all related variables using model.matrix
function_interactions<-function(x){
  print(x)
  y<-df2[x]
  y2<-lapply(y, as.numeric)%>%bind_cols()
  interactions<-data.frame(model.matrix(~.^2, y2))
  y3<-cbind(y2, interactions)%>%subset(., select=c(-X.Intercept.))
}

#bring all the interacted variables together
interaction_df<-lapply(vars_list, function_interactions)%>%bind_cols()

#concat them to the original df
df3 <- bind_cols(df2, interaction_df)

#remove any numeric column with a sum =0 (these are useless)
zero_sum_cols<-as.matrix(dplyr::select_if(df3, is.numeric))
zero_sum_cols<-colnames(zero_sum_cols[,colSums(zero_sum_cols)==0])
zero_sum_cols<-na.omit(zero_sum_cols)
df3<-df3%>%dplyr::select(-zero_sum_cols, Id)#select, keep Id to merge

#now for one-hot encoding on cat_df to turn into binary.
#https://towardsdatascience.com/one-hot-encoding-multicollinearity-and-the-dummy-variable-trap-b5840be3c41a 
library('fastDummies')
categorical_df_all<-df3[, names(df3) %in% categorical_vars]
cat_dummy_df<-fastDummies::dummy_cols(categorical_df_all, remove_first_dummy = TRUE)

#remove_first_dummy which by default is FALSE. If TRUE, it removes the first dummy variable created from each column. 
#This is done to avoid multicollinearity in a multiple regression model caused by included all dummy variables. 

#now we have dummies, lets remove original categoricals 

cat_dummy_df2<-cat_dummy_df[, !names(cat_dummy_df) %in% categorical_vars]
df4<-cbind(df3, cat_dummy_df2)

#now we can remove the original categorical vars from all
df5<-df4[, !names(df4) %in% categorical_vars]

##########################
# modelling & prediction #
##########################
colSums(is.na(df5)) # no NAs apart from SalePrice now

#check distribution of sale price
qqPlot(df5$SalePrice) #non-normal

#take log, remove original
df5<-df5%>%mutate(LogPrice=log(SalePrice))%>%dplyr::select(-SalePrice)
qqPlot(df5$LogPrice) #reasonably normal now

#ensure all cols are numeric
df5<-as.data.frame(apply(df5,2, as.numeric))
str(df5) 

## split train/test
train_data<-filter(df5, !is.na(LogPrice))%>%subset(., Id %in% train_df3$Id)%>%
  dplyr::select(-Id) #make sure outliers are removed.
test_data<-filter(df5, is.na(LogPrice))%>%
  dplyr::select(-Id)

#we will begin with using XGBoost for our first model. The reasons are:
# Regulairization (aids avoiding overfitting), X-validation, application to regression, flexible and highly tunable

#################
# XGB resources #
#################
#for considerable background/underhood knowledge on XGB:
# https://www.slideshare.net/ShangxuanZhang/kaggle-winning-solution-xgboost-algorithm-let-us-learn-from-its-author 
# credit to https://www.hackerearth.com/practice/machine-learning/machine-learning-algorithms/beginners-tutorial-on-xgboost-parameter-tuning-r/tutorial/ 
# for help in tuning xgb parameter syntax

#format to outcome/predictor
train_labels<-train_data%>%dplyr::select(LogPrice)
test_labels<-test_data%>%dplyr::select(LogPrice)

train_data<-train_data%>%dplyr::select(-LogPrice)
test_data<-test_data%>%dplyr::select(-LogPrice)

#prepare xgb matrix
dtrain = xgb.DMatrix(as.matrix(sapply(train_data, as.numeric)), label=as.matrix(sapply(train_labels, as.numeric)))
dtest = xgb.DMatrix(as.matrix(sapply(test_data, as.numeric)), label=as.matrix(sapply(test_labels, as.numeric)))

##################
# untuned XGB #
##################

#default parameters
params <- list(booster = "gbtree", objective = "reg:linear", 
               eta=0.3, gamma=0, 
               max_depth=6, min_child_weight=1, 
               subsample=1, 
               colsample_bytree=1)
set.seed(123)
xgbcv <- xgb.cv( params = params, data = dtrain, 
                 nrounds = 100, nfold = 5, 
                 showsd = T, stratified = T, 
                 print_every_n = 10, 
                 early_stopping_rounds = 20, maximize = F)

print(xgbcv$evaluation_log[which.min(xgbcv$evaluation_log$test_mae_mean)])
nrounds <- xgbcv$best_iteration #best iteration at n=83
min(xgbcv$test.error.mean)

xgb1 <- xgb.train (params = params, data = dtrain, 
                   nrounds = 83, 
                   watchlist = list(val=dtest,train=dtrain), 
                   print_every_n = 10, 
                   early_stop_round = 10, 
                   maximize = F, 
                   eval_metric = c("rmse"))

#error of the basic xgb model?
min(xgb1$evaluation_log$train_rmse) #min rmse of 0.01686

#predict test, take exponent as we had taken log previously
xgbpred <- predict (xgb1,dtest)%>%exp()
untuned_submission<-cbind.data.frame(sample_sub$Id, xgbpred)
colnames(untuned_submission)<-c('ID', 'SalePrice')
write.csv(untuned_submission, 'Kaggle_housePrice_Xgb_untuned.csv', row.names = F)

#var importance plot...
untuned_importance<- xgb.importance (feature_names = colnames(dtrain), model = xgb1)
xgb.ggplot.importance(importance_matrix = untuned_importance[1:25], rel_to_first = TRUE)


########################
# tuning XGB using caret #
########################

## split train/test
train_data<-filter(df5, !is.na(LogPrice))%>%subset(., Id %in% train_df3$Id)%>%
test_data<-filter(df5, is.na(LogPrice))%>%

colnames(train_data)<-make.names(colnames(train_data), unique=T)
colnames(test_data)<-make.names(colnames(test_data), unique=T)

#computationally expensive/timely so for the current purposes we limit options
xgb_grid = expand.grid(
  nrounds = 1000,
  eta = c(0.1, 0.05),
  max_depth = c(2, 4, 6),
  gamma = 0,
  colsample_bytree=1,
  min_child_weight=c(1, 3 ,5),
  subsample=c(0.5, 1)
)

my_control <-trainControl(method="cv", number=3)
xgb_caret <- caret::train(x=train_data, y=train_data$LogPrice, method='xgbTree', trControl= my_control, tuneGrid=xgb_grid) 
xgb_caret$bestTune

##  set new params according to grid search results
parameters_tuned<-list(
  objective = "reg:linear",
  booster = "gbtree",
  eta=0.05, 
  gamma=0,
  max_depth=3, #default=6
  min_child_weight=1, #default=1
  subsample=1,
  colsample_bytree=1
)

xgbcv_tuned <- xgb.cv( params = parameters_tuned, 
                 data = dtrain, 
                 nrounds = 500, nfold = 3, showsd = T, stratified = T, print_every_n = 10, 
                 early_stopping_rounds = 10, maximize = F)
#best iteration at n=391

xgb_mod_tuned <- xgb.train(data = dtrain, params=parameters_tuned, nrounds = 391,
                           showsd = T, stratified = T, print_every_n = 10) 
#achieves a slightly better test_rmse than untuned though not by much

#predict test data
xgbpred_tuned <- predict (xgb_mod_tuned,dtest)%>%exp()%>%round(.,1)

#submit to Kaggle
submission_tuned_xgv<-cbind.data.frame(test_df$Id, xgbpred_tuned)
colnames(submission_tuned_xgv)<-c('ID', 'SalePrice')
write.csv(submission_tuned_xgv, 'Kaggle_housePrice_Xgb.csv', row.names = F)






