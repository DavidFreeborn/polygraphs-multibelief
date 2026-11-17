suppressPackageStartupMessages(library(bnlearn))
suppressPackageStartupMessages(library(gRain))
suppressPackageStartupMessages(library(jsonlite))

respond <- function(value) {
  cat(toJSON(value, auto_unbox = TRUE, digits = NA, null = "null"), "\n", sep = "")
  flush(stdout())
}

infer <- function(request) {
  nodes <- request$nodes
  names <- vapply(nodes, function(node) node$name, character(1))
  states <- setNames(lapply(nodes, function(node) unlist(node$states)), names)
  edges <- matrix(character(), ncol = 2)
  distributions <- list()
  for (node in nodes) {
    parents <- unlist(node$parents)
    scope <- c(node$name, parents)
    dimensions <- states[scope]
    distributions[[node$name]] <- as.table(array(
      unlist(node$cpt), dim = lengths(dimensions), dimnames = dimensions
    ))
    if (length(parents)) edges <- rbind(edges, cbind(parents, node$name))
  }
  observed <- character()
  for (i in seq_along(request$evidence)) {
    factor <- request$evidence[[i]]
    name <- paste0("E", i)
    parents <- unlist(factor$scope)
    weights <- unlist(factor$weights)
    dimensions <- c(setNames(list(c("0", "1")), name), states[parents])
    distributions[[name]] <- as.table(array(
      as.vector(rbind(1 - weights, weights)),
      dim = lengths(dimensions), dimnames = dimensions
    ))
    edges <- rbind(edges, cbind(parents, name))
    observed <- c(observed, name)
  }
  graph <- empty.graph(c(names, observed))
  if (nrow(edges)) arcs(graph) <- edges
  fitted <- custom.fit(graph, distributions)
  engine <- as.grain(fitted)
  if (length(observed)) {
    engine <- setEvidence(engine, nodes = observed, states = rep("1", length(observed)))
  }
  answers <- lapply(request$queries, function(query) {
    query <- unlist(query)
    result <- querygrain(engine, nodes = query, type = "joint")
    if (length(query) > 1) result <- aperm(result, match(query, names(dimnames(result))))
    unname(as.vector(result))
  })
  list(probabilities = answers)
}

respond(list(ready = TRUE, versions = list(
  R = as.character(getRversion()), bnlearn = as.character(packageVersion("bnlearn")),
  gRain = as.character(packageVersion("gRain"))
)))
input <- file("stdin", open = "r")
repeat {
  line <- readLines(input, n = 1, warn = FALSE)
  if (!length(line)) break
  result <- tryCatch(infer(fromJSON(line, simplifyVector = FALSE)),
                     error = function(error) list(error = conditionMessage(error)))
  respond(result)
}
close(input)
