library(dplyr)
library(text2vec)
library(tm)
library(readr)
library(readxl)
library(data.table)
library(quanteda)

# load data 
data_model0 <- readRDS("BiH_mega_corpus_sent_lem.RDS")
Sys.setlocale(locale = "Croatian_Croatia.1250")
meta <- read_xlsx("BiH_mps_parties_speeches.xlsx", 2)
wb <- c("serb", "croat", "bosniak")

# words to check the stabilization on
kw <- read_xlsx("War_dict_BiH.xlsx", 2)
  kw <- unique(kw$feature)
textst

# corpus stats
#corpus_dummy <- corpus(as.character(data_model$text))
#stats_dummy <- textstat_frequency(dfm(corpus_dummy))
#stats_war <- stats_dummy[which(statst_dummy$feature %in% kw)]
#write_excel_csv(as.data.frame(stats_war), "stats_war_bos.csv")
  
# set hyper parameters for modelinh
n_bootstraps <- 100 # number of bootstraps
n_clust <- 8L # set number of cores to be used
l_r <- 0.15 # learning rate; default 0.15
n_i <- 100L # number of iterations (epochs)
n_dim <- 100L # number of dimension in vector
win <- 8L # context window
count_min <- 10L # min count of a word
c_t <- 0.001 # convergence tolerance for the model to be trained

### Training GloVe model
for (x in 1:length(wb)) {
  i_wb <- subset(meta, meta$ethnic_aff == wb[x])$order 
  data_model <- subset(data_model0, doc_id %in% i_wb)

data <- Corpus(VectorSource(tolower(data_model$text))) %>%
  tm_map(removeNumbers) %>% 
  tm_map(removePunctuation) %>% 
  tm_map(stripWhitespace) 
corpus0 <- data$content

# bootstraping the learning
sim_stats <- data.table()
#all_indexes <- list()
for (n in 1:n_bootstraps) {
    print(paste0("model: ", wb[x], " | ", n, " out of ", n_bootstraps, " bootstraps"))
    sample_index <- sample(length(corpus0))  # generate indexes for shuffle
      # all_indexes <- append(all_indexes, list(sample_index))  
    corpus <- corpus0[sample_index] # shuffle the strings in corpus
  
  # Create iterator over tokens
  tokens <- space_tokenizer(corpus)
      
  # Create vocabulary. Terms will be unigrams (simple words).
  it <- itoken(tokens)
  vocab0 <- create_vocabulary(it)
    
  # pruning
  vocab <- prune_vocabulary(vocab0, term_count_min = count_min)
  length(vocab$term)
  
  # Use our filtered vocabulary
  vectorizer <- vocab_vectorizer(vocab)
      
  # create tcm
  tcm <- create_tcm(it, vectorizer, skip_grams_window = win)
  
  #train a model
  glove <- GlobalVectors$new(rank = n_dim, x_max = 10)
  wv_main <- glove$fit_transform(tcm, n_iter = n_i, learning_rate = l_r, convergence_tol = c_t, n_threads = n_clust)

  # glove paper suggests to average main and context vector
  wv_context <- glove$components # create context vector
  word_vectors <- wv_main + t(wv_context)

  # find cosine similarity of kw
  for (i in 1:length(kw)) { # for each reference word
    try({w1 <- word_vectors[kw[i], ]
    cos_sim <- textstat_simil(x = as.dfm(word_vectors), y = as.dfm(matrix(w1, nrow = 1, ncol = length(w1))),
                              method = "cosine")
     close_w <- sort(cos_sim[, 1], decreasing = TRUE)[1:101] # must be high as models are not stable
     ref_w <- names(close_w)[1]
     w <- names(close_w)[2:101]
     cos_value <- unname(close_w)[2:101]
     sim_stats <- rbind(sim_stats, cbind(bootstrap = n, ref_w, w, cos_value))
    }, silent = T)
      }
}
 write_excel_csv(sim_stats, paste0("sim_stats_all_", wb[x], "_p.csv"))
}
#saveRDS(all_indexes, "all_indexes.RDS")

# test means and confidence intervals (CIs)
df <- read_csv("sim_stats_all_serb_p.csv")
kw_i <- unique(df$ref_w)
n_b <- 50 # threshold for occurrences for extracting pairs of words

test_final <- data.table(stringsAsFactors = F)
for (s in kw_i) {
  df0 <- subset(df, ref_w == s)
  consistent_w <- names(table(df0$w))[which(unname(table(df0$w)) >= n_b)]
  for (o in consistent_w) {
    df1 <- subset(df0, w == o)
    test_mean <- mean(df1$cos_value)
    test_error <- 1.95 * (sd(df1$cos_value)/sqrt(n_b))
    test_ci <- cbind(lower = test_mean - test_error, upper = test_mean + test_error)
    test_stats <- cbind(kw = s, w = o, mean = test_mean, test_ci)
    test_final <- rbind(test_final, test_stats) 
  }
}

write_excel_csv(test_final, "stabilized_cos_pairs/filtered/test_stats_all_srb.csv")
