// Databricks notebook source
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{HashingTF, Tokenizer,StopWordsRemover,StringIndexer}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.Row
import org.apache.spark.sql.functions._
import org.apache.spark.sql.functions.regexp_replace
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils

// COMMAND ----------

///FileStore/tables/Tweets.csv
val airlines = spark.read.option("header","true").option("delimiter",","). option("inferSchema","true").csv("/FileStore/tables/Tweets.csv")
val sentiment0 = airlines.select("tweet_id","airline_sentiment","text").toDF("id","sentiment","text")
//sentiment.count() // 14837 tweets
// drop the rows where text is null
val sentiment = sentiment0.na.drop(Seq("text")).withColumn("text", regexp_replace(sentiment0.col("text"), "\\W+", " "))
sentiment.count()  // keep the 14632 not null rows.

// COMMAND ----------

// Configure an ML pipeline, which consists of three stages: tokenizer, hashingTF, and lr.
val tokenizer = new Tokenizer()
  .setInputCol("text")
  .setOutputCol("token")
val remover = new StopWordsRemover()
  .setInputCol("token")
  .setOutputCol("words")
val hashingTF = new HashingTF()
  .setInputCol(remover.getOutputCol)
  .setOutputCol("features")
val indexer = new StringIndexer()
  .setInputCol("sentiment")
  .setOutputCol("label")

val pipeline = new Pipeline().setStages(
  Array(
    tokenizer, 
    remover,
    hashingTF,
    indexer
  )
)

val result = pipeline.fit(sentiment).transform(sentiment)

val twedata = result.select("id","label","features")

// COMMAND ----------

display(twedata)

// COMMAND ----------

val Array(train, test) = twedata.randomSplit(Array(0.9, 0.1), seed=123)
val lr = new LogisticRegression()
  .setMaxIter(10)
  

val paramGrid = new ParamGridBuilder()
  .addGrid(hashingTF.numFeatures, Array(10, 100, 1000))
  .addGrid(lr.regParam, Array(0.1, 0.01))
  .build()

val cv = new CrossValidator()
  .setEstimator(lr)
  .setEvaluator(new MulticlassClassificationEvaluator)
  .setEstimatorParamMaps(paramGrid)
  .setNumFolds(5)  // Use 3+ in practice
  .setParallelism(2)  // Evaluate up to 2 parameter settings in parallel

// Run cross-validation, and choose the best set of parameters.
val cvModel = cv.fit(train)

// COMMAND ----------

val predictions = cvModel.transform(test)
val selectPrediction = predictions.select("id","label", "prediction")
    selectPrediction.show(10)

// COMMAND ----------



// COMMAND ----------

   // Overall Statistics
val predictionAndLabels = predictions
      .select("prediction", "label")
      .rdd.map(x => (x(0).asInstanceOf[Double], x(1)
        .asInstanceOf[Double]))
val metrics = new MulticlassMetrics(predictionAndLabels)

// Confusion matrix
println("Confusion matrix:")
println(metrics.confusionMatrix)
val accuracy = metrics.accuracy
println("Summary Statistics")
println(s"Accuracy = $accuracy")

// Precision by label
val labels = metrics.labels
labels.foreach { l =>
  println(s"Precision($l) = " + metrics.precision(l))
}

// Recall by label
labels.foreach { l =>
  println(s"Recall($l) = " + metrics.recall(l))
}

// False positive rate by label
labels.foreach { l =>
  println(s"FPR($l) = " + metrics.falsePositiveRate(l))
}

// F-measure by label
labels.foreach { l =>
  println(s"F1-Score($l) = " + metrics.fMeasure(l))
}

// Weighted stats
println(s"Weighted precision: ${metrics.weightedPrecision}")
println(s"Weighted recall: ${metrics.weightedRecall}")
println(s"Weighted F1 score: ${metrics.weightedFMeasure}")
println(s"Weighted false positive rate: ${metrics.weightedFalsePositiveRate}")

// COMMAND ----------

test.select("label").groupBy("label").count().show()

// COMMAND ----------


