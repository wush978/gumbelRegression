// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include "../inst/include/gumbelRegression.h"
#include <Rcpp.h>

using namespace Rcpp;

// get_loss
double get_loss(const S4& X, const NumericVector& y, NumericVector& w);
RcppExport SEXP _gumbelRegression_get_loss(SEXP XSEXP, SEXP ySEXP, SEXP wSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const S4& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const NumericVector& >::type y(ySEXP);
    Rcpp::traits::input_parameter< NumericVector& >::type w(wSEXP);
    rcpp_result_gen = Rcpp::wrap(get_loss(X, y, w));
    return rcpp_result_gen;
END_RCPP
}
// test_loss
double test_loss(const S4& X, const NumericVector& y, NumericVector& w0, NumericVector& w, int lossType, double l2, IntegerVector foldId, int foldTarget);
RcppExport SEXP _gumbelRegression_test_loss(SEXP XSEXP, SEXP ySEXP, SEXP w0SEXP, SEXP wSEXP, SEXP lossTypeSEXP, SEXP l2SEXP, SEXP foldIdSEXP, SEXP foldTargetSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const S4& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const NumericVector& >::type y(ySEXP);
    Rcpp::traits::input_parameter< NumericVector& >::type w0(w0SEXP);
    Rcpp::traits::input_parameter< NumericVector& >::type w(wSEXP);
    Rcpp::traits::input_parameter< int >::type lossType(lossTypeSEXP);
    Rcpp::traits::input_parameter< double >::type l2(l2SEXP);
    Rcpp::traits::input_parameter< IntegerVector >::type foldId(foldIdSEXP);
    Rcpp::traits::input_parameter< int >::type foldTarget(foldTargetSEXP);
    rcpp_result_gen = Rcpp::wrap(test_loss(X, y, w0, w, lossType, l2, foldId, foldTarget));
    return rcpp_result_gen;
END_RCPP
}
// get_gradient
NumericVector get_gradient(const S4& X, const NumericVector& y, NumericVector& w);
RcppExport SEXP _gumbelRegression_get_gradient(SEXP XSEXP, SEXP ySEXP, SEXP wSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const S4& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const NumericVector& >::type y(ySEXP);
    Rcpp::traits::input_parameter< NumericVector& >::type w(wSEXP);
    rcpp_result_gen = Rcpp::wrap(get_gradient(X, y, w));
    return rcpp_result_gen;
END_RCPP
}
// test_gradient
NumericVector test_gradient(const S4& X, const NumericVector& y, NumericVector& w0, NumericVector& w, int lossType, double l2, IntegerVector foldId, int foldTarget);
RcppExport SEXP _gumbelRegression_test_gradient(SEXP XSEXP, SEXP ySEXP, SEXP w0SEXP, SEXP wSEXP, SEXP lossTypeSEXP, SEXP l2SEXP, SEXP foldIdSEXP, SEXP foldTargetSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const S4& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const NumericVector& >::type y(ySEXP);
    Rcpp::traits::input_parameter< NumericVector& >::type w0(w0SEXP);
    Rcpp::traits::input_parameter< NumericVector& >::type w(wSEXP);
    Rcpp::traits::input_parameter< int >::type lossType(lossTypeSEXP);
    Rcpp::traits::input_parameter< double >::type l2(l2SEXP);
    Rcpp::traits::input_parameter< IntegerVector >::type foldId(foldIdSEXP);
    Rcpp::traits::input_parameter< int >::type foldTarget(foldTargetSEXP);
    rcpp_result_gen = Rcpp::wrap(test_gradient(X, y, w0, w, lossType, l2, foldId, foldTarget));
    return rcpp_result_gen;
END_RCPP
}
// get_Hv
NumericVector get_Hv(const S4& X, const NumericVector& y, NumericVector& w, NumericVector& v);
RcppExport SEXP _gumbelRegression_get_Hv(SEXP XSEXP, SEXP ySEXP, SEXP wSEXP, SEXP vSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const S4& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const NumericVector& >::type y(ySEXP);
    Rcpp::traits::input_parameter< NumericVector& >::type w(wSEXP);
    Rcpp::traits::input_parameter< NumericVector& >::type v(vSEXP);
    rcpp_result_gen = Rcpp::wrap(get_Hv(X, y, w, v));
    return rcpp_result_gen;
END_RCPP
}
// test_Hv
NumericVector test_Hv(const S4& X, const NumericVector& y, NumericVector& w0, NumericVector& w, NumericVector& v, int lossType, IntegerVector foldId, double l2, int foldTarget);
RcppExport SEXP _gumbelRegression_test_Hv(SEXP XSEXP, SEXP ySEXP, SEXP w0SEXP, SEXP wSEXP, SEXP vSEXP, SEXP lossTypeSEXP, SEXP foldIdSEXP, SEXP l2SEXP, SEXP foldTargetSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const S4& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const NumericVector& >::type y(ySEXP);
    Rcpp::traits::input_parameter< NumericVector& >::type w0(w0SEXP);
    Rcpp::traits::input_parameter< NumericVector& >::type w(wSEXP);
    Rcpp::traits::input_parameter< NumericVector& >::type v(vSEXP);
    Rcpp::traits::input_parameter< int >::type lossType(lossTypeSEXP);
    Rcpp::traits::input_parameter< IntegerVector >::type foldId(foldIdSEXP);
    Rcpp::traits::input_parameter< double >::type l2(l2SEXP);
    Rcpp::traits::input_parameter< int >::type foldTarget(foldTargetSEXP);
    rcpp_result_gen = Rcpp::wrap(test_Hv(X, y, w0, w, v, lossType, foldId, l2, foldTarget));
    return rcpp_result_gen;
END_RCPP
}
// gumbelRegressionCpp
List gumbelRegressionCpp(S4 X, NumericVector y, IntegerVector foldId, NumericVector lambdaSeq, double tolerance);
RcppExport SEXP _gumbelRegression_gumbelRegressionCpp(SEXP XSEXP, SEXP ySEXP, SEXP foldIdSEXP, SEXP lambdaSeqSEXP, SEXP toleranceSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< S4 >::type X(XSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type y(ySEXP);
    Rcpp::traits::input_parameter< IntegerVector >::type foldId(foldIdSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type lambdaSeq(lambdaSeqSEXP);
    Rcpp::traits::input_parameter< double >::type tolerance(toleranceSEXP);
    rcpp_result_gen = Rcpp::wrap(gumbelRegressionCpp(X, y, foldId, lambdaSeq, tolerance));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_gumbelRegression_get_loss", (DL_FUNC) &_gumbelRegression_get_loss, 3},
    {"_gumbelRegression_test_loss", (DL_FUNC) &_gumbelRegression_test_loss, 8},
    {"_gumbelRegression_get_gradient", (DL_FUNC) &_gumbelRegression_get_gradient, 3},
    {"_gumbelRegression_test_gradient", (DL_FUNC) &_gumbelRegression_test_gradient, 8},
    {"_gumbelRegression_get_Hv", (DL_FUNC) &_gumbelRegression_get_Hv, 4},
    {"_gumbelRegression_test_Hv", (DL_FUNC) &_gumbelRegression_test_Hv, 9},
    {"_gumbelRegression_gumbelRegressionCpp", (DL_FUNC) &_gumbelRegression_gumbelRegressionCpp, 5},
    {NULL, NULL, 0}
};

RcppExport void R_init_gumbelRegression(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
