package org.machine.learning.util

object Utils {

  def multiplyVectors(v1: List[BigDecimal], v2: List[BigDecimal]): List[BigDecimal] ={

    assert(v1.size == v2.size, "Vectors should be of the same size for valid multiplication.")

    v1.zip(v2).map(wx => wx._1 * wx._2)
  }

  def sumVectors(v1: List[BigDecimal], v2: List[BigDecimal]): List[BigDecimal] ={

    assert(v1.size == v2.size, "Vectors should be of the same size for valid multiplication.")

    v1.zip(v2).map(wx => wx._1 + wx._2)
  }

}
