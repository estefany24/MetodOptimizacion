# Establecer el repositorio CRAN
options(repos = c(CRAN = "https://cran.rstudio.com"))
library(shiny)

install.packages("shiny")

# Función Gauss-Jordan
gj <- function(m) {
  n <- nrow(m)
  for (j in 1:n) {
    if (m[j, j] == 0) {
      fila_no_cero <- which(m[(j+1):n, j] != 0)
      if (length(fila_no_cero) == 0) stop("No hay solución única")
      fila_intercambiar <- fila_no_cero[1] + j
      temp <- m[j, ]
      m[j, ] <- m[fila_intercambiar, ]
      m[fila_intercambiar, ] <- temp
    }
    m[j, ] <- m[j, ] / m[j, j]
    for (i in 1:n) {
      if (i != j) {
        m[i, ] <- m[i, ] - m[i, j] * m[j, ]
      }
    }
  }
  round(m[, n+1], 6)
}

# Interfaz de Usuario
ui <- fluidPage(
  titlePanel("Método de Gauss-Jordan"),
  
  sidebarLayout(
    sidebarPanel(
      numericInput("n", "Número de incógnitas:", value = 2, min = 2, max = 10),
      actionButton("crear_tabla", "Crear tabla"),
      actionButton("resolver", "Resolver sistema")
    ),
    
    mainPanel(
      h4("Ingrese los coeficientes y términos independientes:"),
      uiOutput("matriz_inputs"),
      h4("Resultado:"),
      verbatimTextOutput("resultado")
    )
  )
)

# Servidor
server <- function(input, output, session) {
  
  observeEvent(input$crear_tabla, {
    output$matriz_inputs <- renderUI({
      n <- input$n
      inputs <- list()
      for (i in 1:n) {
        fila_inputs <- lapply(1:(n+1), function(j) {
          numericInput(
            inputId = paste0("cell_", i, "_", j),
            label = NULL,
            value = 0,
            width = "120px"   
          )
        })
        
        fila_completa <- fluidRow(
          column(2, strong(paste0("Ecuación ", i, ":"))),
          lapply(fila_inputs, function(x) column(2, x)) # También más espacio por columna
        )
        
        inputs[[i]] <- fila_completa
      }
      do.call(tagList, inputs)
    })
  })
  
  observeEvent(input$resolver, {
    n <- input$n
    m <- matrix(0, nrow = n, ncol = n+1)
    
    for (i in 1:n) {
      for (j in 1:(n+1)) {
        valor <- input[[paste0("cell_", i, "_", j)]]
        if (is.null(valor)) {
          output$resultado <- renderText("Complete todos los valores antes de resolver.")
          return()
        }
        m[i, j] <- valor
      }
    }
    
    tryCatch({
      soluciones <- gj(m)
      res_text <- paste(paste0("x", 1:n, " = ", soluciones), collapse = "\n")
      output$resultado <- renderText(res_text)
    }, error = function(e) {
      output$resultado <- renderText(paste("Error:", e$message))
    })
  })
}

# Ejecutar app
shinyApp(ui, server)

