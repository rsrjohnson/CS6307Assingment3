// Databricks notebook source

import org.apache.spark.sql.functions.desc
import org.graphframes.GraphFrame
import org.apache.spark.sql.functions._
import scala.math._


val flights = spark.read.option("header","true").csv("/FileStore/tables/1062433686_T_T100D_SEGMENT_US_CARRIER_ONLY.csv")
val airports = flights.select("ORIGIN").toDF("id").distinct() // we have 831 vertices.
val airportEdges = flights.select("ORIGIN", "DEST").toDF("src","dst")

val airportGraph = GraphFrame(airports, airportEdges)

// COMMAND ----------

// the iteration procedure
val outd = airportGraph.outDegrees
val Edges=airportGraph.edges
val out_ep = outd.withColumn("pr",lit(2.0))  // change the pr value to get compare with the pagerank library
                 .withColumn("epr", col("pr")/col("outDegree"))
                 .withColumnRenamed("id","src")
                 .join(Edges,Seq("src"))
var N= 831
var alpha =0.15
var c1 = alpha/N
var c2 = 1- alpha
var iter_out = out_ep
for (i <- 1 until 5){   // adjust the iteration cycles
  
  val sump_c = iter_out.groupBy("dst").sum("epr").withColumnRenamed("dst","src").withColumnRenamed("sum(epr)","sump_c")
       iter_out = iter_out.join(sump_c, Seq("src"))
                  .withColumn("pr",col("sump_c")*c2 + c1)
                  .withColumn("epr", col("pr")/col("outDegree"))
                  .drop("sump_c")
  }

iter_out.select("src","pr").distinct().orderBy(desc("pr")).show(10)

// COMMAND ----------

// verify the output, by compareing with pageRank function

val ranks = airportGraph.pageRank.resetProbability(0.15).maxIter(3).run()

ranks.vertices.orderBy(desc("pagerank")).select("id", "pagerank").show(10)
