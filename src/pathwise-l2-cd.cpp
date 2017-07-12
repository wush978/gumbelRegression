#include <Rcpp.h>
#include <RcppParallel.h>
#include <pathwise-l2-cd.h>
using namespace Rcpp;

class FoldTask : public RcppParallel::Worker {

  const S4& X;

  const NumericVector& y;

  const IntegerVector& foldId;

  const NumericVector& labmdaSeq;

  const double tolerance;

  std::vector<NumericMatrix>& gumbelCoefList;

  std::vector<NumericVector>& cvMseList;

public:

  FoldTask(
    const S4& _X,
    const NumericVector& _y,
    const IntegerVector& _foldId,
    const NumericVector& _labmdaSeq,
    const double _tolerance,
    std::vector<NumericMatrix>& _gumbelCoefList,
    std::vector<NumericVector>& _cvMseList) :
  X(_X), y(_y), foldId(_foldId), labmdaSeq(_labmdaSeq), tolerance(_tolerance),
  gumbelCoefList(_gumbelCoefList), cvMseList(_cvMseList)
  { }

  ~FoldTask() { }

  void operator()(std::size_t begin, std::size_t end) {
    for(std::size_t foldTarget = begin;foldTarget < end;foldTarget++) {

    }
  }

};

static int ncol(const S4& X) {
  const IntegerVector Dim(X.slot("Dim"));
  return Dim[1];
}

//[[Rcpp::export(.gumbelRegression.cpp.internal)]]
List gumbelRegressionCpp(S4 X, NumericVector y, IntegerVector foldId, NumericVector lambdaSeq, double tolerance) {
  int nfold = Rcpp::max(foldId);
  std::vector<NumericMatrix> gumbelCoefList;
  std::vector<NumericVector> cvMseList;
  for(int i = 0;i < nfold;i++) {
    gumbelCoefList.push_back(NumericMatrix(1 + ncol(X), lambdaSeq.size()));
    cvMseList.push_back(NumericVector(lambdaSeq.size()));
  }
  gumbelCoefList.push_back(NumericMatrix(1 + ncol(X), lambdaSeq.size()));
  FoldTask foldTask(X, y, foldId, lambdaSeq, tolerance, gumbelCoefList, cvMseList);
  RcppParallel::parallelFor(1, nfold + 2, foldTask);
  List result(nfold + 1);
  for(int i = 0;i < nfold;i++) {
    result[i] = List::create(
      Named("coef") = gumbelCoefList[i],
      Named("cv.mse") = cvMseList[i]
    );
  }
  result[nfold] = List::create(Named("coef") = gumbelCoefList[nfold]);
  return result;
}
