library(data.table)
library(ggplot2)

f <- fread("../data/robustness_field_cnt.csv")
# final list of countries for analysis
countr <- fread("./orcid_countrylist.csv")
f <- f[country %in% countr$iso2]
pred <- read_parquet("../data/stem_and_med_classifications.parquet")
mg <- data.table(merge(pred, f, by.x="cmd",by.y="clean_affiliation"))
mg$index <- NULL
mg[, prop :=count_val/sum(.SD$count_val),by=country]
mg <- mg[order(-prop)]
mg[, prop_cum := cumsum(prop),by=country]

top_250_per_country <- mg[, .SD[1:250],by=country][!is.na(cmd)]
write.csv(top_250_per_country, 
          "top_250_per_country.csv",
          row.names=F)

## Now run 01_flag_nonacademic.py, and then manual cleaning to remove things that actually are fields (mostly med)
## Manually cleaned output is in flagged_manual_removal.csv

flagged <- fread("flagged_manual_removal.csv")
flagged[reason =="not_english"]$reason <- "non_english"
flagged[reason =="not_englis"]$reason <- "non_english"
flagged <- flagged[cmd %in% top_250_per_country$cmd]
flagged[,.N/length(unique(top_250_per_country$cmd)), by=reason]
write.csv(flagged[reason == "not_academic"],"../../data/classified_nonacademic.csv",row.names=F)