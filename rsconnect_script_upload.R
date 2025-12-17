setwd("D:/OneDrive/前桌面")

library(rsconnect)

rsconnect::setAccountInfo(
  name = 'pcos-cardiometabolic-cluster',
  token = 'token',
  secret = 'secret'
)

rsconnect::deployApp(
  appDir = ".",
  appFiles = c("app.R", "best_classification_model.rds"),
  appName = "pcos-final-working",
  forceUpdate = TRUE
)
