package org.machine.learning.regressions

import org.scalatest.{FlatSpec, Matchers, OptionValues}

class LinearRegressionSpec extends FlatSpec with Matchers with OptionValues {

  val x: Array[Int] = Array(1, 2)
  val y: Array[Int] = Array(1, 2)

  behavior of "LinearRegression.scala"

  it should "correctly predict value for 'y' for given training set" in {
    val lr = LinearRegression(x, y)
    lr.predict(3, 0).prediction should be (3)
  }

}
