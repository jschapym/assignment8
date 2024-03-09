package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"

	"gonum.org/v1/gonum/mat"
)

func main() {

	rand.Seed(time.Now().UnixNano())
	nObs := 100
	nPredictors := 2

	X := mat.NewDense(nObs, nPredictors, nil)
	y := make([]float64, nObs)
	for i := 0; i < nObs; i++ {
		X.Set(i, 0, 1)
		X.Set(i, 1, rand.NormFloat64())
		y[i] = 2*X.At(i, 0) + 3*X.At(i, 1) + rand.NormFloat64()*0.5
	}

	nIterations := 1000
	coefficients := make([]float64, nPredictors)
	for i := range coefficients {
		coefficients[i] = rand.NormFloat64()
	}
	variance := rand.ExpFloat64()

	for iter := 0; iter < nIterations; iter++ {

		for j := 0; j < nPredictors; j++ {

			mean, variance := conditionalPosterior(X, y, coefficients, variance, j)

			coefficients[j] = rand.NormFloat64()*math.Sqrt(variance) + mean
		}

		sumSquaredResiduals := 0.0
		for i := 0; i < nObs; i++ {
			predicted := mat.Dot(X.RowView(i), mat.NewVecDense(nPredictors, coefficients))
			residual := y[i] - predicted
			sumSquaredResiduals += residual * residual
		}
		variance = 1.0 / rand.ExpFloat64() * sumSquaredResiduals / float64(nObs)
	}

	fmt.Println("Estimated coefficients:", coefficients)
	fmt.Println("Estimated variance:", variance)
}

func conditionalPosterior(X *mat.Dense, y []float64, coefficients []float64, variance float64, j int) (float64, float64) {
	nObs, nPredictors := X.Dims()

	Xj := mat.Col(nil, j, X)
	var XjXj float64
	for i := 0; i < nObs; i++ {
		XjXj += Xj[i] * Xj[i]
	}

	XXWithoutJ := mat.NewDense(nPredictors-1, nPredictors-1, nil)
	for i := 0; i < nPredictors-1; i++ {
		for k := 0; k < nPredictors-1; k++ {
			rowIndex := i
			colIndex := k
			if i >= j {
				rowIndex++
			}
			if k >= j {
				colIndex++
			}
			if rowIndex < nPredictors && colIndex < nPredictors {
				XXWithoutJ.Set(i, k, X.At(rowIndex, colIndex))
			}
		}
	}

	for i := j; i < nPredictors-1; i++ {
		for k := j; k < nPredictors-1; k++ {
			XXWithoutJ.Set(i, k, X.At(i+1, k+1))
		}
	}

	var XjY float64
	for i := 0; i < nObs; i++ {
		XjY += Xj[i] * y[i]
	}

	var XY float64
	for i := 0; i < nObs; i++ {
		row := X.RowView(i)
		for k := 0; k < nPredictors; k++ {
			XY += row.AtVec(k) * y[i]
		}
	}

	mean := (XjY - coefficients[j]*XjXj + coefficients[j]*XjXj) / XXWithoutJ.At(j, j)

	variance = variance / XXWithoutJ.At(j, j)

	return mean, variance
}
