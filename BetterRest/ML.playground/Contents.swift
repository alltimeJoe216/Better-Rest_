import CreateML
import Foundation

// Snag our data from fileURL
let data = try MLDataTable(contentsOf: URL(fileURLWithPath: "/Users/joeveverka/Desktop/Not School/Personal Dev/better/better-rest.json"))

// Train 80%, Test 20%
let (trainingData, testingData) = data.randomSplit(by: 0.8)

let regressor = try MLRegressor(trainingData: trainingData, targetColumn: "actualSleep") // Pick my regressor :]

let evaluationMetrics = regressor.evaluation(on: testingData) // We want to evaluate our metrics below


print(evaluationMetrics.rootMeanSquaredError)
print(evaluationMetrics.maximumError)

let metadata = MLModelMetadata(author: "Daddy", shortDescription: "A model to prdict sleep time for coffee drinks", license: nil, version: "1.0")

try regressor.write(to: URL(fileURLWithPath: "/Users/joeveverka/Desktop/Not School/Personal Dev/Better Rest_/BetterRest/sleepcalculator.mlmodel"), metadata: metadata)
