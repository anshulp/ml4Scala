package org.machine.learning.classifiers

import Utils._
import org.machine.learning.util.Utils

object LogisticRegression {
  def apply(
    dataRows: List[List[BigDecimal]],
    seedWeights: List[BigDecimal],
    learningRate: BigDecimal,
    maxIterations: Int): LogisticRegression = {

    assert(dataRows != null, "data should not be null")
    assert(seedWeights != null, "weights should not be null")

    val ZERO_LOGIT: BigDecimal = math.pow(math.E, -10)

    new LogisticRegression(
      labels = dataRows.map(row => row(0)),
      features = dataRows.map(row => List(BigDecimal(1.0)) ++ row.slice(1, row.length)), // pre-pend each feature with 1
      seedWeights = seedWeights,
      learningRate = learningRate,
      maxIterations = maxIterations)
  }
}

// Refer section 2 - http://cs229.stanford.edu/notes/cs229-notes1.pdf
class LogisticRegression(
  labels: List[BigDecimal],
  features: List[List[BigDecimal]],
  seedWeights: List[BigDecimal],
  learningRate: BigDecimal,
  maxIterations: Int) {

  //TODO: Add tolerance (predictedLabel - actualLabel) > tolerance, break the training loop

  private val learnedWeights: List[BigDecimal] = {
    // learn till maxIteration
    // for each feature/label combination
    (0 to (features.size - 1) * (maxIterations - 1)).foldLeft(seedWeights) { (weights, featureIndex) => {

      val tempW = {
        val idx = featureIndex%(features.size-1)
        val predictedLabel: BigDecimal = sigmoidFor(x = features(idx), weights = weights)

        val actualLabel: BigDecimal = labels(idx)

        val diff = (actualLabel - predictedLabel)

        weights.map(w => multiplyVectors(weights, features(idx)).sum * diff)
      }
      sumVectors(weights, tempW)
    }
    }
  }

  private def sigmoidFor(x: List[BigDecimal], weights: List[BigDecimal]): BigDecimal = {

    val actualLogit: BigDecimal = {
      val multipliedValue = multiplyVectors(weights, x).sum
      if (multipliedValue != 0) -1 * multipliedValue else 0
    }

    val logit: BigDecimal =
      if (actualLogit == 0) 0.00000000000000000000001
      else if (actualLogit == 1) 0.99999999999999999999999
      else actualLogit

    1 / (1 + math.exp(logit.doubleValue()))
  }

  def predict(x: List[BigDecimal]): Int = {

    // prepend Ist feature (1) to the list
    val features: List[BigDecimal] = List(BigDecimal(1.0)) ++ x

    val probability = sigmoidFor(features, learnedWeights)
    if (probability >= 0.5)
      1
    else
      0
  }

}
