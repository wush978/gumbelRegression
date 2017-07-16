#include <Rcpp.h>
#include <gumbel.h>
using namespace Rcpp;

//[[Rcpp::export(".get.loss.cpp.internal")]]
double get_loss(const S4& X, const NumericVector& y, NumericVector& w) {
  GumbelRegression::GumbelRegressionData data(X, y, nullptr);
  GumbelRegression::ComputationArguments args(data, 0);
  GumbelRegression::GumbelRegressionLoss loss(data, args);
  args.set_loss_type(GumbelRegression::LossType::All);
  return loss(&w[0]);
}

//[[Rcpp::export(".test.loss.cpp")]]
double test_loss(const S4& X, const NumericVector& y, NumericVector& w0, NumericVector& w, int lossType, double l2 = 0, IntegerVector foldId = IntegerVector(0), int foldTarget = 0) {
  boost::variant<std::nullptr_t, IntegerVector> vfoldId;
  if (foldId.size() == 0) vfoldId = nullptr; else vfoldId = foldId;
  GumbelRegression::GumbelRegressionData data(X, y, vfoldId);
  GumbelRegression::ComputationArguments args(data, foldTarget);
  GumbelRegression::GumbelRegressionLoss loss(data, args);
  double * pw0 = &w0[0];
  args.set_l2(l2);
  args.set_pw(&pw0);
  args.set_loss_type(static_cast<GumbelRegression::LossType>(lossType));
  return loss(&w[0]);
}

//[[Rcpp::export(".get.gradient.cpp.internal")]]
NumericVector get_gradient(const S4& X, const NumericVector& y, NumericVector& w) {
  GumbelRegression::GumbelRegressionData data(X, y, nullptr);
  GumbelRegression::ComputationArguments args(data, 0);
  GumbelRegression::GumbelRegressionGradient grad(data, args);
  args.set_loss_type(GumbelRegression::LossType::All);
  NumericVector result(data.ncolX + 1);
  grad(&w[0], &result[0]);
  return result;
}

//[[Rcpp::export(".test.gradient.cpp")]]
NumericVector test_gradient(const S4& X, const NumericVector& y, NumericVector& w0, NumericVector& w, int lossType, double l2 = 0, IntegerVector foldId = IntegerVector(0), int foldTarget = 0) {
  boost::variant<std::nullptr_t, IntegerVector> vfoldId;
  if (foldId.size() == 0) vfoldId = nullptr; else vfoldId = foldId;
  GumbelRegression::GumbelRegressionData data(X, y, vfoldId);
  GumbelRegression::ComputationArguments args(data, foldTarget);
  GumbelRegression::GumbelRegressionGradient grad(data, args);
  double * pw0 = &w0[0];
  args.set_l2(l2);
  args.set_pw(&pw0);
  args.set_loss_type(static_cast<GumbelRegression::LossType>(lossType));
  std::vector<double> result(data.ncolX + 1, 0.0);
  grad(&w[0], &result[0]);
  switch(lossType) {
  case 0 :
  {
    result.resize(1);
    break;
  }
  case 1 :
  {
    result.resize(result.size() - 1);
    break;
  }
  }
  return wrap(result);
}

//[[Rcpp::export(".get.Hv.cpp.internal")]]
NumericVector get_Hv(const S4& X, const NumericVector& y, NumericVector& w, NumericVector& v) {
  GumbelRegression::GumbelRegressionData data(X, y, nullptr);
  GumbelRegression::ComputationArguments args(data, 0);
  GumbelRegression::GumbelRegressionGradient grad(data, args);
  GumbelRegression::GumbelRegressionHessianV Hv(data, args);
  args.set_loss_type(GumbelRegression::LossType::All);
  NumericVector result(data.ncolX + 1);
  grad(&w[0], &result[0]);
  Hv(&v[0], &result[0]);
  return result;
}

//[[Rcpp::export(".test.Hv.cpp")]]
NumericVector test_Hv(const S4& X, const NumericVector& y, NumericVector& w0, NumericVector& w, NumericVector& v, int lossType, IntegerVector foldId = IntegerVector(0), double l2 = 0, int foldTarget = 0) {
  boost::variant<std::nullptr_t, IntegerVector> vfoldId;
  if (foldId.size() == 0) vfoldId = nullptr; else vfoldId = foldId;
  GumbelRegression::GumbelRegressionData data(X, y, vfoldId);
  GumbelRegression::ComputationArguments args(data, foldTarget);
  GumbelRegression::GumbelRegressionGradient grad(data, args);
  GumbelRegression::GumbelRegressionHessianV Hv(data, args);
  double * pw0 = &w0[0];
  args.set_l2(l2);
  args.set_pw(&pw0);
  args.set_loss_type(static_cast<GumbelRegression::LossType>(lossType));
  std::vector<double> result(data.ncolX + 1, 0.0);
  grad(&w[0], &result[0]);
  Hv(&v[0], &result[0]);
  Hv(&v[0], &result[0]);
  Hv(&v[0], &result[0]);
  switch(lossType) {
  case 0 :
  {
    result.resize(1);
    break;
  }
  case 1 :
  {
    result.resize(result.size() - 1);
    break;
  }
  }
  return wrap(result);
}

