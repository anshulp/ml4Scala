package org.machine.learning.classifiers

import org.scalatest.{FlatSpec, Matchers, OptionValues}

class LogisticRegressionSpec extends FlatSpec with Matchers with OptionValues {

  val dataRows: List[List[BigDecimal]] = List(List(1, 2), List(0, -1), List(1, 3), List(0, -2), List(1, 4))
  val seedWeights: List[BigDecimal] = List(0.001, 0)
  val learningRate: BigDecimal = 0.1
  val maxIterations: Int = 20

  behavior of "LogisticRegression.scala"

  it should "correctly predict value for +ve 'y' for given training set" in {
    val lr = LogisticRegression(dataRows, seedWeights, learningRate, maxIterations)

    val p = lr.predict(List(5))
    p should be (1)
  }

  it should "correctly predict value for -ve 'y' for given training set" in {
    val lr = LogisticRegression(dataRows, seedWeights, learningRate, maxIterations)

    val p = lr.predict(List(-5))
    p should be (0)
  }

}
