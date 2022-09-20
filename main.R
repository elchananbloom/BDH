#install.packages("RSQLite")
#install.packages("tidyverse")
#install.packages("cgwtools")

start_time <- Sys.time()

library("RSQLite")
library("tidyverse")
library("recommenderlab")
library("stringr")
library("dplyr")
library("caret")
library("cgwtools")



my.predict <- function (model, newdata, n = 10, data = NULL,
                        type = c("topNList", "ratings", "ratingMatrix"), ...)
{
  type <- match.arg(type)
  newdata_id <- NULL
  if (is.numeric(newdata)) {
    if (model$sample)
      stop("(EE) User id in newdata does not work when sampling is used!")
    newdata_id <- newdata
    newdata <- model$data[newdata, ]
  }
  else {
    if (ncol(newdata) != ncol(model$data))
      stop("(EE) number of items in newdata does not match model.")
    if (!is.null(model$normalize))
      newdata <- normalize(newdata, method = model$normalize)
  }
  
  cat('(II) running similarity() calculation\n')
  sim <- similarity(newdata, model$data, method = model$method,
                    min_matching = model$min_matching_items,
                    min_predictive = model$min_predictive_items)
  cat('(II) similarity() done\n')
  
  if (!is.null(newdata_id))
    sim[cbind(seq(length(newdata_id)), newdata_id)] <- NA
  
  cat(paste('(II) creating knn with', model$nn ,'neighbors\n'))
  neighbors <- .knn(sim, model$nn)
  cat('(II) knn done\n')
  
  if(model$weighted) {
    ## similarity of the neighbors
    s_uk <- sapply(1:nrow(sim), FUN=function(i)
      sim[i, neighbors[[i]]])
    if(!is.matrix(s_uk)) s_uk <- as.matrix(t(s_uk))
    
    ## calculate the weighted sum
    ratings <- t(sapply(1:nrow(newdata), FUN=function(i) {
      ## neighbors ratings of active user i
      r_neighbors <- as(model$data[neighbors[[i]]], "dgCMatrix")
      ## normalize by the sum by the number of neighbors
      drop(as(crossprod(r_neighbors, as.numeric(as.data.frame(s_uk[, i])[,])), "matrix")) /
        colSums(r_neighbors, na.rm = TRUE)
    }))
    
  }
  else {
    ratings <- t(sapply(1:nrow(newdata), FUN=function(i) {
      colCounts(model$data[neighbors[[i]]])
    }))
  }
  rownames(ratings) <- rownames(newdata)
  ratings <- new("realRatingMatrix", data = dropNA(ratings),
                 normalize = getNormalize(newdata))
  
  cat ('(II) de-normalize the ratings (back to rating scale)\n')
  ratings <- denormalize(ratings)
  cat ('(II) de-normalize done\n')
  
  returnRatings(ratings, newdata, type, n)
}
.knn <- function(sim, k)
  lapply(1:nrow(sim), FUN = function(i)
    head(order(sim[i,], decreasing = TRUE, na.last = NA), k))

#########################################################################


books.db <- dbConnect(RSQLite::SQLite(), "D:\\AVIEL\\Downloads\\BigDataHackathon-master\\BX-Books_hkv1.db")
books <- dbReadTable(books.db, "bx-books")
books <- distinct(books)
rm(books.db)  # clean RAM


ratings.db <- dbConnect(RSQLite::SQLite(), "D:\\AVIEL\\Downloads\\BigDataHackathon-master\\BX-Ratings_hkv1.db")
ratings <- dbReadTable(ratings.db, "bx-book-ratings")
#ratings <- filter(ratings, ratings$Book.Rating > 1)
#ratings <- arrange(ratings, ratings$User.ID)
rm(ratings.db)  # clean RAM

users.db <- dbConnect(RSQLite::SQLite(), "D:\\AVIEL\\Downloads\\BigDataHackathon-master\\BX-Users_hkv1.db")
users <- dbReadTable(users.db, "bx-users")
rm(users.db)  # clean RAM


#3a
nb_users <- nrow(users[!duplicated(users$User.ID),])
#3b
nb_books <- nrow(books[!duplicated(books$ISBN),])
#3c
nb_ratings <- nrow(ratings[ratings$Book.Rating,])

#3d
N <- 50
user_ratings <- data.frame(table(ratings$User.ID))
freq.user <- filter(user_ratings, Freq >= N )
freq.user <- arrange(freq.user, desc(Freq))
colnames(freq.user)<- c("ID", "Frequence")

filter(ratings, User.ID %in% freq.user$ID)


#3e
N2 <- 7
book.freq.ratings <- data.frame(table(ratings$ISBN))
filter(book.freq.ratings, Freq >= N2 )
#3f
arrange(book.freq.ratings, desc(Freq))
#3g
arrange(user_ratings, desc(Freq))



ratings <- ratings[(ratings$ISBN %in% books$ISBN),] #remove unknowns ISBNs from ratings.
ratings <- ratings[(ratings$User.ID %in% users$User.ID),] #remove unknowns ISBNs from ratings.


min.rating.user <- 4
min.rating.book <- 4


real.rating.matrix <- as(ratings, "realRatingMatrix")



while(min(rowCounts(real.rating.matrix)) < min.rating.user || min(colCounts(real.rating.matrix)) < min.rating.book){
  real.rating.matrix <- real.rating.matrix[rowCounts(real.rating.matrix) >= min.rating.user, colCounts(real.rating.matrix) >= min.rating.book]
  cat(".")
}

cat(paste("\nMin rating user", min(rowCounts(real.rating.matrix)), "\n" ,collapse = " "))
cat(paste("Min rating book", min(colCounts(real.rating.matrix)) , "\n" ,collapse = " "))
cat(paste("Dim of rating matrix is:", real.rating.matrix@data@Dim[1], ":", real.rating.matrix@data@Dim[2], "(", object.size(real.rating.matrix)/1000000000 , "Gb)\n" ,collapse = " "))

gc()

sets <- evaluationScheme(data = real.rating.matrix, method = "split",
                         train = 0.8, given = min.rating.user,
                         goodRating = 5, k = 5)


UB_recommender <- Recommender(data = getData(sets, "train"),
                              method = "UBCF", parameter = NULL)


UB_prediction <- my.predict(UB_recommender@model,
                            newdata = getData(sets, "known"),
                            n = 10,
                            type = "ratings")


R.UB <- round(UB_prediction@data, digits = 1 )
colnames(R.UB) <- sapply( colnames(R.UB), function(c) books[books$ISBN == c,]$Book.Title )
R.UB <- as.matrix(R.UB)

accuracy.R.UB <- calcPredictionAccuracy(x = UB_prediction, given = min.rating.user,
                                        data = getData(sets, "unknown"), byUser=TRUE)


all.accuracy.R.UB <- calcPredictionAccuracy(x = UB_prediction, given = min.rating.user,
                                            data = getData(sets, "unknown"), byUser=FALSE)

V.RMSE <- list()
V.RMSE[['UBCF']] <- accuracy.R.UB[,"RMSE"]

save(real.rating.matrix, sets, R.UB, file = "model.rdata")
save(UB_recommender,UB_prediction, accuracy.R.UB,  file = "temp.rdata")
rm(R.UB, UB_prediction, UB_recommender, accuracy.R.UB)
gc()



Sys.time() - start_time



# To only save users that gived grades to more than N books and return in rating_test2
min.rating.user <- 8
min.rating.book <- 8


real.rating.matrix <- as(ratings, "realRatingMatrix")



while(min(rowCounts(real.rating.matrix)) < min.rating.user || min(colCounts(real.rating.matrix)) < min.rating.book){
  real.rating.matrix <- real.rating.matrix[rowCounts(real.rating.matrix) >= min.rating.user, colCounts(real.rating.matrix) >= min.rating.book]
  cat(".")
}

cat(paste("\nMin rating user", min(rowCounts(real.rating.matrix)), "\n" ,collapse = " "))
cat(paste("Min rating book", min(colCounts(real.rating.matrix)) , "\n" ,collapse = " "))
cat(paste("Dim of rating matrix is:", real.rating.matrix@data@Dim[1], ":", real.rating.matrix@data@Dim[2], "(", object.size(real.rating.matrix)/1000000000 , "Gb)\n" ,collapse = " "))

gc()

sets <- evaluationScheme(data = real.rating.matrix, method = "split",
                         train = 0.8, given = min.rating.user,
                         goodRating = 5, k = 5)


rm(real.rating.matrix)
gc()


IB_recommender <- Recommender(data = getData(sets, "train"),
                              method = "IBCF", parameter = NULL)

IB_prediction <- predict(IB_recommender,
                         newdata = getData(sets, "known"),
                         n = 10,
                         type = "ratings")


R.IB <- round(IB_prediction@data, digits = 1 )
colnames(R.IB) <- sapply( colnames(R.IB), function(c) books[books$ISBN == c,]$Book.Title )
R.IB <- as.matrix(R.IB)



accuracy.R.IB <- calcPredictionAccuracy(x = IB_prediction, given = min.rating.user,
                                        data = getData(sets, "unknown"), byUser=TRUE)


all.accuracy.R.IB <- calcPredictionAccuracy(x = IB_prediction, given = min.rating.user,
                                            data = getData(sets, "unknown"), byUser=FALSE)

V.RMSE[['IBCF']] <- accuracy.R.IB[,"RMSE"]

resave(R.IB, V.RMSE, file = "model.rdata")
resave(IB_recommender,IB_prediction, accuracy.R.IB, all.accuracy.R.IB, all.accuracy.R.UB,  file = "temp.rdata")


end_time <- Sys.time()
dif_time <- end_time - start_time

end_time - start_time

load("temp.rdata")
load("model.rdata")


h.UBCF <- hist(V.RMSE[["UBCF"]], breaks = seq(min(V.RMSE[["UBCF"]], na.rm = TRUE),ceiling(max(V.RMSE[["UBCF"]], na.rm = TRUE)),0.5))
data.frame(h.UBCF$breaks[1:length(h.UBCF$counts)] , h.UBCF$counts)

h.IBCF <- hist(V.RMSE[["IBCF"]], breaks = seq(min(V.RMSE[["IBCF"]], na.rm = TRUE),ceiling(max(V.RMSE[["IBCF"]], na.rm = TRUE)),0.5))
data.frame(h.IBCF$breaks[1:length(h.IBCF$counts)] , h.IBCF$counts)

top10.UBCF <- R.UB[1:500, 1:10] 
colnames(top10.UBCF) <- substr(colnames(top10.UBCF), 1, 12)
write.table(top10.UBCF, "UB.txt", append = FALSE, sep = "\t\t\t", dec = ".",
                     row.names = TRUE, col.names = TRUE)


top10.IBCF <- R.IB[1:500, 1:10] 
colnames(top10.IBCF) <- substr(colnames(top10.IBCF), 1, 12)
write.table(top10.IBCF, "IB.txt", append = FALSE, sep = "\t\t\t", dec = ".",
            row.names = TRUE, col.names = TRUE)