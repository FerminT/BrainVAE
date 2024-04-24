library(MVN)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("Usage: Rscript run_analysis.R <input_file_path> <mvn_test_type>")
}
input_file <- args[1]
mvn_test_type <- args[2]
data <- read.csv(input_file)

if ("embedding" %in% names(data)) {
  embeddings <- gsub("\\[|\\]", "", data$embedding)  # Remove square brackets
  embeddings <- gsub("^\\s+|\\s+$", "", embeddings)   # Trim leading and trailing spaces

  # Convert the character array to a numeric matrix
  embeddings_matrix <- do.call(rbind, lapply(embeddings, function(x) {
    # Split each embedding string by spaces and convert to numeric
    numeric_vector <- as.numeric(unlist(strsplit(x, split="\\s+")))
    if (length(numeric_vector) != 354) {
      stop("An embedding does not contain 354 numeric elements. Found ", length(numeric_vector), " elements.")
    }
    return(numeric_vector)
  }))

  mvn_result <- mvn(embeddings_matrix, mvnTest = mvn_test_type)
  multivariateNormality <- mvn_result$multivariateNormality
  output_file <- paste0(dirname(input_file), "/multivariate_normality_results.csv")
  write.csv(multivariateNormality, output_file)

  cat("Multivariate normality test results:\n")
  print(multivariateNormality)
} else {
  cat("The 'embedding' column does not exist in the dataset.\n")
}