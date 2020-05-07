##################################################
# Kaggle competition - Disaster Tweets using NLP #
##################################################

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
library('imputeMissings')
library('tm')
library('wordcloud')
library('textreg')
library('syuzhet')
library('tidytext')
library(ggplot2)
library('sqldf')
library(plyr)
library(tidyr)


#aim: predict whether the tweets are about a real disaster
# resources: NLP/text mining: https://www.tidytextmining.com/tfidf.html 
# https://www.datacamp.com/community/tutorials/R-nlp-machine-learning 

#set wd and load
#setwd("~/Dropbox/Kaggle/Disaster_Tweets")
setwd("C:/Users/jaket/Dropbox/Kaggle/Disaster_Tweets")

train_df<-read.csv('train.csv', stringsAsFactors = F, na.strings = c(""))%>%mutate(set='train')
test_df<-read.csv('test.csv', stringsAsFactors = F, na.strings = c(""))%>%mutate(set='test')
sample_sub<-read.csv('sample_submission.csv')

#merge train/test
all<-merge(train_df, test_df, all=T)

#############
#   EDA     #
#############

colSums(is.na(all)) #keywords and locations missing

unique_keywords<-unique(all$keyword) #222 unqiue keywords
unique_locations<-unique(all$location) #from 4522 locations
unique_tweets<-unique(all$text) #10679 of 10876 tweets are unique (98%)

#######################
##### EDA-keywords ####
#######################

#first of all, from viewing it is clear that %20 seems to appear in
# key words to represent a space, lets replace this to make it 1 string

all$keyword<-gsub('%20','_', all$keyword)

#top keywords
top_keywords<-all%>%
  filter(!is.na(keyword))%>%group_by(keyword)%>%
  dplyr::summarise(n=n())%>%arrange(desc(n))%>%
  as.data.frame()

#plot top 20 keywords
ggplot(data=(top_keywords%>%dplyr::slice(1:20)), aes(x=keyword, fill=keyword))+
  geom_bar()+coord_flip() #not much use as many all seem to = 50.

#plot relation of keywords to target
#absolute relationship
keyword_target_abs<-all%>%group_by(keyword)%>%filter(set=='train')%>%
  dplyr::summarise(times_used=n(), 
                   total_disasters=sum(target, na.rm=T))%>%
  arrange(desc(total_disasters))%>%dplyr::slice(1:20)%>%
  na.omit()%>%melt(id.var='keyword')

abs_key_plot<-ggplot(keyword_target_abs,
       aes(x=reorder(keyword,value), y=value, fill=variable))+
  geom_bar(stat="identity", position=position_dodge())+coord_flip()+
  theme_light()

#fractional relationship
keyword_target_perc<-all%>%group_by(keyword)%>%filter(set=='train')%>%
  dplyr::summarise(times_used=n(), 
                   total_disasters=sum(target, na.rm=T),
                   fractional_disasters=(total_disasters/times_used)*100)%>%
  arrange(desc(fractional_disasters))%>%dplyr::slice(1:20)%>%
  na.omit()%>%melt(id.var='keyword')

#percentage of words associated with true disaster
perc_key_plot<-ggplot(subset(keyword_target_perc,variable=='fractional_disasters'),
       aes(x=reorder(keyword,value), y=value, fill=keyword))+
  geom_bar(stat="identity", position=position_dodge())+coord_flip()+
  theme_light()

grid.arrange(abs_key_plot, perc_key_plot)

#### keyword sentiments ######
keyword_sentiments<-get_nrc_sentiment(all$keyword)
keyword_cor<-cbind(data.frame(keyword_sentiments), all$target)%>%na.omit()%>%cor()

#we'll make these keyword sentiments predictors in our main df- rename and bind
colnames(keyword_sentiments)<-paste0(colnames(keyword_sentiments), '_keyword')
all<-bind_cols(all, keyword_sentiments)

#associations between sentiments and
library(corrplot)
corrplot.mixed(keyword_cor)

##################
#  EDA- location #
##################
#here we see that USA and United states are being used, U.S.A, U.S
# this is happening a lot, we cannot clean all manually but we will try to sort the populated/obvious ones.
all <- all%>%transform(location=revalue(location,c("United States"="USA",
                                                   'U.S.A' = 'USA',
                                                   'US' = 'USA',
                                                   'U.S' = 'USA',
                                                   'usa'='USA',
                                                   'United States of America'= 'USA',
                                                   'UK.'= 'UK',
                                                   'U.K.'= 'UK',
                                                   'uk'= 'UK',
                                                   'United Kingdom' = 'UK')))

location_target_abs<-all%>%group_by(location)%>%filter(set=='train')%>%
  dplyr::summarise(use_in_location=n(), 
                   total_disasters=sum(target, na.rm=T),
                   fractional_disasters=(total_disasters/use_in_location)*100)%>%
  arrange(desc(total_disasters))%>%dplyr::slice(1:20)%>%
  na.omit()%>%reshape2::melt(id.var='location')

#where are tweets coming from?
ggplot(location_target_abs,
                     aes(x=reorder(location,value), y=value, fill=variable))+
  geom_bar(stat="identity", position=position_dodge())+coord_flip()+
  theme_light()

#its clear that a lot of tweets are coming from US states but not categorised as USA,
# we will generate a Country column to try an capture this. Most is US and some UK.
US_strings<-c('USA', 'usa', 'U.S.A', 'United Stated', 'The United States of America', 'USA.',
              'US', 'US.', 'U.S.')
States_strings<-c('Alabama', 'Alaska', 'Arizona', 'Arkansas', 'Baltimore', 'California', 'Colorado', 'Connecticut', 'Delaware', 'Florida', 
'Georgia', 'Hawaii', 'Idaho', 'Illinois', 'Indiana', 'Iowa', 'Kansas', 'Kentucky', 'Louisiana', 'Maine', 'Maryland',
'Massachusetts', 'Michigan', 'Minnesota', 'Mississippi', 'Missouri', 'Montana', 'Nebraska', 'Nevada', 
'New Hampshire', 'New Jersey', 'New Mexico', 'New York', 'North Carolina', 'North Dakota', 'Ohio', 'Oklahoma',
'Oregon', 'Pennsylvania', 'Rhode Island', 'South Carolina', 'South Dakota', 'Tennessee', 'Texas', 'Utah', 'Vermont',
 'Virginia', 'Washington', 'West Virginia', 'Wisconsin', 'Wyoming')
UK_strings<-c('UK', 'U.K', 'UK.', 'U.K.', 'United Kingdom', 'UnitedKingdom', 'England', 'Scotland', 'Ireland', 'Wales',
              'Northern Ireland')
UK_cities<-c('Glasgow', 'Leeds', 'Liverpool', 'Newcastle', 'Sheffield', 'Belfast', 'Bristol', 'Nottingham', 
   'Southampton/Portsmouth', 'Leicester', 'London', 'Manchester', 'Edinburugh', 'Aberdeen', 'Middleborough', 'Sussex')

all$country<-ifelse(grepl(paste(US_strings, collapse ='|'), all$location, ignore.case = T), 'USA',
                    ifelse(grepl(paste(States_strings, collapse ='|'), all$location, ignore.case = T), 'USA',
                           ifelse(grepl(paste(UK_strings, collapse ='|'), all$location, ignore.case = T), 'UK',
                                  ifelse(grepl(paste(UK_cities, collapse ='|'), all$location, ignore.case = T), 'UK',all$location))))
#we could do this for many more than UK and US but it would have to be manual - these seem like the main ones to catch



######################
### EDA - tweets ###
######################
#lets start with some basic pre-processing 
#https://hackernoon.com/text-processing-and-sentiment-analysis-of-twitter-data-22ff5e51e14c 
trainingcorpus <- VCorpus(VectorSource(all$text))

removeURL <- function(x) gsub("http[[:alnum:]]*", "", x)
removeBlank <- function(x) gsub("rt", "", x)
removeTabs<-function(x) gsub("[ |\t]{2,}", "", x)
removeBlankSpacesEnd<-function(x) gsub(" $", "", x)
removeBlankSpacesStart<- function(x) gsub("^ ", "", x)
removePunct<- function(x) gsub('[[:punct:] ]+',' ',x)
#some of these tm() text functions dont work perfectly so we make our own

preprocess <- function(document){
  document <- tm_map(document, removePunctuation)
  document <- tm_map(document, removeNumbers)
  document <- tm_map(document, stripWhitespace)
  document <- tm_map(document, content_transformer(tolower))
  document <- tm_map(document, PlainTextDocument)
  document <- tm_map(document, content_transformer(removeURL)) 
  document <- tm_map(document, content_transformer(removeBlank))
  document <- tm_map(document, content_transformer(removeTabs))
  document <- tm_map(document, content_transformer(removeBlankSpacesEnd))
  document <- tm_map(document, content_transformer(removeBlankSpacesStart))
  document <- tm_map(document, content_transformer(removePunct))
  return(document)
}
trainingcorpus <- preprocess(trainingcorpus)


#remove stop words - 
text_char<-as.data.frame(convert.tm.to.character(trainingcorpus))
colnames(text_char)<-'text_simple'
all<-cbind(all, text_char)

#generate a wordcloud to visualise what words are used most frequently
#on raw text
wordcloud(all$text,min.freq = 10,colors=brewer.pal(8, "Dark2"),random.color = TRUE,max.words = 500)

#on simplified text
wordcloud(all$text_simple,min.freq = 10,colors=brewer.pal(8, "Dark2"),random.color = TRUE,max.words = 500)

## is the word/character count associated with the target?
all$charcount_raw<-nchar(all$text)
all$charcount_tidy<-nchar(all$text_simple)

all$wordcount_raw<-sapply(strsplit(all$text, " "), length)
all$wordcount_tidy<-sapply(strsplit(all$text_simple, " "), length)

#plot association with target
ggplot(all, aes(x=target, y=charcount_tidy))+
  geom_point(position = position_jitter(width= 0.2, height = 0), size = 2)+
  geom_smooth( method='lm', se=F)

ggplot(all, aes(x=target, y=charcount_tidy))+
  geom_point(position = position_jitter(width= 0.2, height = 0), size = 2)+
  geom_smooth( method='lm', se=F)
#target= 1 tend to be longer in characters

#words vs target
ggplot(all, aes(x=target, y=wordcount_raw))+
  geom_point(position = position_jitter(width= 0.2, height = 0), size = 2)+
  geom_smooth( method='lm', se=F)

ggplot(all, aes(x=target, y=wordcount_tidy))+
  geom_point(position = position_jitter(width= 0.2, height = 0), size = 2)+
  geom_smooth( method='lm', se=F)
#no clear trend between word count and target.



# getting sentiments from tweets using Syuzhet
tweet_sentiments<-get_nrc_sentiment((all$text_simple))
tweet_sentiment_scores<-data.frame(colSums(tweet_sentiments[,]))
print(tweet_sentiment_scores)

#plot frequency of the occurence of different sentiments
sentiment_plot<-as.data.frame(reshape2::melt(tweet_sentiments))
ggplot(sentiment_plot, aes(x=reorder(variable, value), y=value, fill=as.factor(variable)))+geom_bar(stat="identity")+
  theme_light() 
#here we see that negative and fear are the two most frequent sentiments - what are 
# the implications for disaster prediction?

#add sentiments to the main df as predictors
colnames(tweet_sentiments)<-paste0(colnames(tweet_sentiments), '_tweets')
all<-bind_cols(all, tweet_sentiments)

##############################################
##### Term frequency in tweets - tokenizing #
##############################################


# tokenizing syntax helped by: https://rpubs.com/abhijitjantre/NgramModelWithNaturalLanguageProcessingNLP 


#creating tokenizers
Unigramtokenizer <- function(x){
  unlist(lapply(ngrams(words(x), 1), paste, collapse = " "), use.names = FALSE)}
Bigramtokenizer <- function(x){
  unlist(lapply(ngrams(words(x), 2), paste, collapse = " "), use.names = FALSE)}
Trigramtokenizer <-function(x){
  unlist(lapply(ngrams(words(x), 3), paste, collapse = " "), use.names = FALSE)}

#creating document matrix
unigramdocumentmatrix <- TermDocumentMatrix(trainingcorpus,control = list(tokenize = Unigramtokenizer))
bigramdocumentmatrix <- TermDocumentMatrix(trainingcorpus,control = list(tokenize = Bigramtokenizer))
trigramdocumentmatrix <- TermDocumentMatrix(trainingcorpus,control = list(tokenize = Trigramtokenizer))

#computing frequencies
unigramf <- findFreqTerms(unigramdocumentmatrix,lowfreq =10) #these patterns must occur at least 10 times
bigramf <- findFreqTerms(bigramdocumentmatrix,lowfreq = 10)
trigramf <- findFreqTerms(trigramdocumentmatrix,lowfreq = 10)

Unigramfreq <- rowSums(as.matrix(unigramdocumentmatrix[unigramf,]))
Unigramfreq <- data.frame(word=names(Unigramfreq),frequency=Unigramfreq)
Bigramfreq <- rowSums(as.matrix(bigramdocumentmatrix[bigramf,]))
Bigramfreq <- data.frame(word=names(Bigramfreq),frequency=Bigramfreq)
Trigramfreq <- rowSums(as.matrix(trigramdocumentmatrix[trigramf,]))
Trigramfreq <- data.frame(word=names(Trigramfreq),frequency=Trigramfreq)

plot_ngram <- function(data,title,num){
  df <- data[order(-data$frequency),][1:num,]
  barplot(df[1:num,]$freq, las = 2, names.arg = df[1:num,]$word,
          col ="red", main = title,
          ylab = "Word frequencies",cex.axis =0.8)
}
par(mar=c(10,4,4,2))

#plot the top uni, bi and tri grams
top_unigrams<-plot_ngram(Unigramfreq,"Top Unigrams",20)
top_bigrams<-plot_ngram(Bigramfreq,"Top Bigrams",20)
top_trigrams<-plot_ngram(Trigramfreq,"Top Trigrams",20)

#now we'll manually 1-hot encode n-grams into
unigram_list<-list()
for (i in unique(Unigramfreq$word)){
  var<-as.data.frame(as.numeric(grepl(i, all$text_simple)))
  position<-as.numeric(match(i, Unigramfreq$word))
  colnames(var)<-paste0('unifreq_', position)
  unigram_list[[i]]<-var
}
unigram_bin<-cbind.data.frame(unigram_list)

## bi

bigram_list<-list()
for (i in unique(Bigramfreq$word)){
  var<-as.data.frame(as.numeric(grepl(i, all$text_simple)))
  position<-as.numeric(match(i, Bigramfreq$word))
  colnames(var)<-paste0('bifreq_', position)
  bigram_list[[i]]<-var
}
bigram_bin<-cbind.data.frame(bigram_list)

## tri
trigram_list<-list()
for (i in unique(Trigramfreq$word)){
  var<-as.data.frame(as.numeric(grepl(i, all$text_simple)))
  position<-as.numeric(match(i, Trigramfreq$word))
  colnames(var)<-paste0('trifreq_', position)
  trigram_list[[i]]<-var
}
Trigram_bin<-cbind.data.frame(trigram_list)

all2<-cbind.data.frame(all, unigram_bin, bigram_bin, Trigram_bin)
all<-NULL



###################
###### hashtags ###
###################

#we know that hashtags are used to group/identify tops - how many # were used?
hashtags_per_tweet<- as.data.frame(str_count(all$text, "#"))
colnames(hashtags_per_tweet)<-'hashtags'

#how many hashtags were used?
total_hashtags<-sum(hashtags_per_tweet$hashtags) #4946 hashtags were used

#how many of the tweets used hashtags?
hashtags_per_tweet$binary<-as.numeric(ifelse(hashtags_per_tweet$hashtags>0, 1, 0)) 
n_tweets_with_hastags<-sum(hashtags_per_tweet$binary) #2569 tweets used hashtags
perc_tweets_using_hashtags<-(n_tweets_with_hastags/nrow(all))*100 #23% of tweets contains hashtags

#extract hashtags
hashtags<-regmatches(all$text, gregexpr("#\\S+", all$text)) #get all characters beginning with hashtag
hashtag_df<-unlist(hashtags)
unique_hashtags<-data.frame(unique(hashtag_df)) #get unique hashtags in df
colnames(unique_hashtags)<-'hashtags'
unique_hashtags$hashtags<-str_replace(unique_hashtags$hashtags, '#', '')

#manual one-hot encoding of hashtags to main df
hashtag_list<-list()
for (i in unique_hashtags$hashtags){
  var<-as.data.frame(as.numeric(grepl(i, all$text)))
  position<-as.numeric(match(i, unique_hashtags$hashtags))
  colnames(var)<-paste0('hashtag_', position)
  hashtag_list[[i]]<-var
}
hashtag_bin<-cbind.data.frame(hashtag_list)

#bind encoded hashtags to all
all3<-cbind(all2, hashtag_bin)

###############
# modelling ###
###############
#set up df for modelling

#remove strings
character_vars<-train_all %>%select_if(is.character)

#take training data from all
train_all<-filter(all3, set=='train')%>%dplyr::select(-text, -text_simple, -set, -id)
  

#check for NAs, we have missing data from locations and keywords which we will replace with blanks 
colSums(is.na(train_all))
train_all<-train_all%>%mutate(keyword=as.factor(ifelse(is.na(keyword), 'unknown', keyword)),
                              location=as.factor(ifelse(is.na(location), 'unknown', location)),
                              country=as.factor(ifelse(is.na(country), 'unknown', country)))%>%droplevels()
sum(is.na(train_all)) # no NAs left


#check for factors with <2 levels
factor_levels<-sapply(TweetsTrain[,sapply(TweetsTrain, is.factor)], nlevels)


trainIndex <- createDataPartition(train_all$target, p = .8, 
                                  list = FALSE, 
                                  times = 1)

TweetsTrain <- train_all[ trainIndex,]
TweetsTest  <- train_all[-trainIndex,]

#only doing a basic glm with no parameter searching because we have >5k predictors and not a lot of computation power

model <- train(
  target ~., data = TweetsTrain,
  method = "glm",
  trControl = trainControl("cv", number = 10)
)

# Make predictions on the test data
predict_train.test <- model %>% predict(test.data)

#final predictions on test df
test_final_df<-filter(test, set=='test')%>%dplyr::select(-text, -text_simple, -set, -id)%>%
  mutate(keyword=as.factor(ifelse(is.na(keyword), 'unknown', keyword)),
         location=as.factor(ifelse(is.na(location), 'unknown', location)),
         country=as.factor(ifelse(is.na(country), 'unknown', country)))%>%droplevels()
  
predict_train.test <- model %>% predict(test_final_df)

#write
write.csv(predict_train.test, 'kaggle_disaster_tweet_submission.csv')




