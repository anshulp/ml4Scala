package org.machine.learning.regressions

/***
  * Implement http://mathworld.wolfram.com/LeastSquaresFitting.html
  * Author - Anshul Pandey
  */

object LinearRegression {

  def apply(
    xValues: Array[Int],
    yValues: Array[Int]): LinearRegression = {

    assert(xValues != null, "values of x should not be null.")
    assert(yValues != null, "values of y should not be null.")
    assert(xValues.length == yValues.length, "Dimensions of X and Y should be similar.")

    new LinearRegression(xValues = xValues, yValues = yValues)
  }
}

// creates the linear model and exposes a method to get the output for a var
// http://mathworld.wolfram.com/LeastSquaresFitting.html
class LinearRegression(
    xValues: Array[Int],
    yValues: Array[Int]) {

  private val avgX: BigDecimal = xValues.sum/xValues.length
  private val avgY: BigDecimal = yValues.sum/yValues.length

  private val sumOfSquaresOfX: BigDecimal = xValues.map(x => {
    val xDiff = x-avgX
    xDiff * xDiff
  }).sum

  private val sumOfSquaresOfY: BigDecimal = yValues.map(y => {
    val yDiff = y-avgY
    yDiff * yDiff
  }).sum

  private val sumOfSquaresOfXAndY = (xValues zip yValues).map(xy => (xy._1 - avgX) * (xy._2 - avgY)).sum

  // linear function is f(a, b) = a + b.x
  private val b = sumOfSquaresOfXAndY / sumOfSquaresOfX
  private val a = avgY - (b * avgX)

  def predict(x: Int, y: Int = 0): Prediction = {
    val predictedY = a + b * x
    Prediction(predictedY, if (y == 0) y else y - predictedY)
  }
}


