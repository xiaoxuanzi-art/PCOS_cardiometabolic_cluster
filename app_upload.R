library(shiny)
library(ggplot2)
custom_css <- "
body { font-family: Arial, sans-serif; background: #f8f9fa; }
.card { border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 20px; border: none; }
.card-header { background: #007bff; color: white; border-radius: 12px 12px 0 0; font-weight: bold; padding: 15px 20px; }
.btn-primary { background: #007bff; border-color: #007bff; border-radius: 8px; padding: 12px 30px; font-size: 18px; font-weight: bold; }
.result-box { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 12px; padding: 25px; margin: 20px 0; }
.calc-result { background: #e9f7fe; border-left: 4px solid #17a2b8; padding: 10px 15px; margin: 10px 0; border-radius: 4px; font-weight: bold; }
.calc-result .value { color: #17a2b8; font-size: 18px; }
"
cat("初始化...\n")
model_loaded <- FALSE
model <- NULL

tryCatch({
  if (file.exists("best_classification_model.rds")) {
    model <- readRDS("best_classification_model.rds")
    model_loaded <- TRUE
    cat("模型加载成功\n")
  }
}, error = function(e) {
  cat("模型加载失败，使用模拟模式:", e$message, "\n")
})

calculate_BMI <- function(w, h) if (h > 0) round(w/(h*h), 1) else NA
calculate_LH_FSH <- function(lh, fsh) if (fsh > 0) round(lh/fsh, 2) else NA
calculate_HOMA <- function(ins, glu) round((ins*glu)/22.5, 2)
clinical_predictor <- function(inputs) {
  if (any(is.na(unlist(inputs)))) return(NULL)
  if (model_loaded && !is.null(model)) {
    tryCatch({
      LH_FSH_ratio <- calculate_LH_FSH(inputs$LH, inputs$FSH)
      HOMA_IR <- calculate_HOMA(inputs$Fasting_Insulin, inputs$Fasting_Glucose)
      
      model_data <- data.frame(
        T = as.numeric(inputs$T),
        HDL = as.numeric(inputs$HDL),
        BMI0 = as.numeric(inputs$BMI0),
        LH_FSH = LH_FSH_ratio,
        HOMA = HOMA_IR,
        TG = as.numeric(inputs$TG),
        stringsAsFactors = FALSE
      )
      pred_labels <- predict(model, newdata = model_data)
      pred_probs <- predict(model, newdata = model_data, type = "prob")
      class_map <- c("X1" = 1, "X2" = 2, "X3" = 3)
      pred_class <- class_map[as.character(pred_labels)]
      
      class_desc <- if(pred_class == 1) "SHP（高雄激素状态亚型/Sever hyperandrogenism phenotype）" else
        if(pred_class == 2) "HCRP（高心血管代谢风险亚型/High cardiometabolic disease risk phenotype）" else "MP（轻型/Mild PCOS phenotype）"
      
      probs <- if (all(c("X1", "X2", "X3") %in% colnames(pred_probs))) {
        list(SHP = pred_probs[, "X1"], HCRP = pred_probs[, "X2"], MP = pred_probs[, "X3"])
      } else {
        list(SHP = 0.33, HCRP = 0.34, MP = 0.33)
      }
      
      return(list(
        predicted_class = pred_class,
        class_description = class_desc,
        probabilities = probs,
        confidence = max(unlist(probs))
      ))
      
    }, error = function(e) {
      cat("模型预测错误/Error:", e$message, "\n")
    })
  }
  if (inputs$T > 0.8 && inputs$BMI0 > 25) {
    class_desc <- "SHP（高雄激素状态亚型/Sever hyperandrogenism phenotype）"
    probs <- c(0.6, 0.2, 0.2)
  } else if (inputs$HDL < 1.2 || inputs$TG > 1.8) {
    class_desc <- "HCRP（高心血管代谢风险亚型/High cardiometabolic disease risk phenotype）"
    probs <- c(0.2, 0.6, 0.2)
  } else {
    class_desc <- "MP（轻型/Mild PCOS phenotype）"
    probs <- c(0.2, 0.2, 0.6)
  }
  
  probs <- probs / sum(probs)
  
  list(
    predicted_class = if(grepl("SHP", class_desc)) 1 else if(grepl("HCRP", class_desc)) 2 else 3,
    class_description = class_desc,
    probabilities = list(SHP = probs[1], HCRP = probs[2], MP = probs[3]),
    confidence = max(probs)
  )
}
ui <- fluidPage(
  tags$head(tags$style(HTML(custom_css))),
  
  titlePanel(
    div(
      h1("PCOS心血管代谢亚型预测系统/PCOS cardiometabolic phynotype clustering system"),
      hr(style = "border-top: 3px solid #007bff;")
    )
  ),
  
  fluidRow(
    column(6,
           div(class = "card",
               div(class = "card-header", h4("输入临床数据/Clinical data input")),
               div(class = "card-body",
                   numericInput("input_T", "总睾酮/Serum total testosteron level (T,ng/mL)", 0.5, 0, 10, 0.01),
                   numericInput("input_HDL", "高密度脂蛋白/Serum HDL-C level (HDL-C,mmol/L)", 1.5, 0, 10, 0.01),
                   
                   div(class = "card",
                       div(class = "card-header", h5("BMI计算/BMI calculation")),
                       div(class = "card-body",
                           fluidRow(
                             column(6, numericInput("input_Weight", "体重/Weight (m,kg)", 60, 30, 200, 0.1)),
                             column(6, numericInput("input_Height", "身高/Height (h,m)", 1.65, 1.4, 2.0, 0.01))
                           ),
                           uiOutput("bmi_display")
                       )
                   ),
                   
                   numericInput("input_TG", "甘油三酯Serum TG level (TG,mmol/L)", 1.2, 0, 20, 0.1),
                   
                   div(class = "card",
                       div(class = "card-header", h5("LH/FSH比值/LH to FSH ratio")),
                       div(class = "card-body",
                           fluidRow(
                             column(6, numericInput("input_LH", "LH (mIU/mL)", 6, 0, 50, 0.1)),
                             column(6, numericInput("input_FSH", "FSH (mIU/mL)", 8, 0.1, 50, 0.1))
                           ),
                           uiOutput("lh_fsh_display")
                       )
                   ),
                   
                   div(class = "card",
                       div(class = "card-header", h5("HOMA-IR指数/HOMA-IR index")),
                       div(class = "card-body",
                           fluidRow(
                             column(6, numericInput("input_Insulin", "空腹胰岛素/Serum fasting insulin level (FINS,μIU/mL)", 10, 0, 100, 0.1)),
                             column(6, numericInput("input_Glucose", "空腹血糖/Serum fasting glucose level (FG,mmol/L)", 5.0, 0, 20, 0.1))
                           ),
                           uiOutput("homa_display")
                       )
                   ),
                   
                   actionButton("predict", "开始预测/Predicting", class = "btn-primary btn-block btn-lg"),
                   actionButton("reset", "重置数据/Reset", class = "btn-secondary btn-block")
               )
           )
    ),
    
    column(6,
           uiOutput("result_box"),
           div(class = "card",
               div(class = "card-header", h4("HCRP概率/HCRP probability")),
               div(class = "card-body", uiOutput("hcrp_display"))
           ),
           div(class = "card",
               div(class = "card-header", h4("亚型概率分布/Phenotype probability")),
               div(class = "card-body", plotOutput("prob_plot", height = 300))
           )
    )
  )
)

server <- function(input, output, session) {
  bmi <- reactive(calculate_BMI(input$input_Weight, input$input_Height))
  lh_fsh <- reactive(calculate_LH_FSH(input$input_LH, input$input_FSH))
  homa <- reactive(calculate_HOMA(input$input_Insulin, input$input_Glucose))
  output$bmi_display <- renderUI({
    if(!is.na(bmi())) div(class = "calc-result", paste("BMI:", bmi(), "kg/m²"))
  })
  
  output$lh_fsh_display <- renderUI({
    if(!is.na(lh_fsh())) div(class = "calc-result", paste("LH/FSH:", lh_fsh()))
  })
  
  output$homa_display <- renderUI({
    if(!is.na(homa())) div(class = "calc-result", paste("HOMA-IR:", homa()))
  })
  prediction <- reactive({
    inputs <- list(
      T = input$input_T,
      HDL = input$input_HDL,
      BMI0 = bmi(),
      TG = input$input_TG,
      LH = input$input_LH,
      FSH = input$input_FSH,
      Fasting_Insulin = input$input_Insulin,
      Fasting_Glucose = input$input_Glucose
    )
    
    if (any(is.na(unlist(inputs)))) return(NULL)
    clinical_predictor(inputs)
  })
  output$result_box <- renderUI({
    res <- prediction()
    if (is.null(res)) return(div(class = "result-box", h4("请输入数据/Please input the indicator values")))
    
    color <- if(grepl("SHP", res$class_description)) "#dda5c8" else
      if(grepl("HCRP", res$class_description)) "#ff9900" else "#688cc8"
    
    div(class = "result-box",
        h3("预测结果/Result"),
        h2(res$class_description, style = paste0("color: ", color, "; text-align: center;"))
    )
  })
  output$hcrp_display <- renderUI({
    res <- prediction()
    if (is.null(res)) return(div(h4("等待输入/Please input")))
    
    prob <- res$probabilities$HCRP * 100
    color <- if(prob > 50) "red" else if(prob > 30) "orange" else "green"
    
    div(style = "text-align: center;",
        h1(style = paste0("color: ", color), paste0(round(prob, 1), "%")),
        h4(if(prob > 50) "高风险/High Risk" else if(prob > 30) "中等风险/Moderate Risk" else "低风险/Low Risk")
    )
  })
  output$prob_plot <- renderPlot({
    res <- prediction()
    if (is.null(res)) {
      data <- data.frame(类型 = c("SHP", "HCRP", "MP"), 概率 = c(0, 0, 0))
    } else {
      data <- data.frame(
        类型 = c("SHP", "HCRP", "MP"),
        概率 = c(res$probabilities$SHP, res$probabilities$HCRP, res$probabilities$MP) * 100
      )
    }
    
    ggplot(data, aes(x = 类型, y = 概率, fill = 类型)) +
      geom_bar(stat = "identity") +
      geom_text(aes(label = ifelse(概率 > 0, paste0(round(概率, 1), "%"), "")), 
                vjust = -0.5, size = 5) +
      scale_fill_manual(values = c("#dda5c8", "#ff9900", "#688cc8")) +
      labs(y = "概率/Probabilyt (%)") +
      theme_minimal() +
      theme(legend.position = "none")
  })
  observeEvent(input$reset, {
    updateNumericInput(session, "input_T", value = 0.5)
    updateNumericInput(session, "input_HDL", value = 1.5)
    updateNumericInput(session, "input_Weight", value = 60)
    updateNumericInput(session, "input_Height", value = 1.65)
    updateNumericInput(session, "input_TG", value = 1.2)
    updateNumericInput(session, "input_LH", value = 6)
    updateNumericInput(session, "input_FSH", value = 8)
    updateNumericInput(session, "input_Insulin", value = 10)
    updateNumericInput(session, "input_Glucose", value = 5.0)
  })
}
shinyApp(ui, server)