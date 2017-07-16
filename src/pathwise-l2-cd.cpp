#include <Rcpp.h>
#include <RcppParallel.h>
#include <pathwise-l2-cd.h>
using namespace Rcpp;

class FoldTask : public RcppParallel::Worker {

  const GumbelRegression::GumbelRegressionData& data;

  const NumericVector& lambdaSeq;

  const double tolerance;

  std::vector<NumericMatrix>& gumbelCoefList;

  std::vector<NumericVector>& cvMseList;

  const int nfold;

  const double init_log_sigma;

  const double init_intercept;

  const bool verbose;

public:

  FoldTask(
    const GumbelRegression::GumbelRegressionData& _data,
    const NumericVector& _lambdaSeq,
    const double _tolerance,
    std::vector<NumericMatrix>& _gumbelCoefList,
    std::vector<NumericVector>& _cvMseList,
    int _nfold,
    double _init_log_sigma,
    double _init_intercept,
    bool _verbose) :
  data(_data), lambdaSeq(_lambdaSeq), tolerance(_tolerance),
  gumbelCoefList(_gumbelCoefList), cvMseList(_cvMseList),
  nfold(_nfold), init_log_sigma(_init_log_sigma), init_intercept(_init_intercept), verbose(_verbose)
  { }

  ~FoldTask() { }

  // This is running in a thread
  void cvPredict(std::size_t foldTarget) {
    if (foldTarget == nfold + 1) return;
    auto& cvMse(cvMseList[foldTarget - 1]);
    auto& gumbelCoef(gumbelCoefList[foldTarget - 1]);
    std::vector<double> mu(data.nrowX, 0.0);
    for(std::size_t lambda_i = 0;lambda_i < lambdaSeq.size();lambda_i++) {
      const double *w = &gumbelCoef(0, lambda_i);
      const double *w1 = w + 1;
      // Xv::Xv_dgCMatrix_numeric_folded(data.X, pw1, buf, data.foldId, foldTarget, true);
      mu.resize(data.nrowX);
      std::fill(mu.begin(), mu.end(), 0.0);
      auto muSize = Xv::Xv_dgCMatrix_numeric_folded(data.X, w1, mu, data.foldId, foldTarget, false);
      mu.resize(muSize);
      double& result(cvMse[lambda_i]);
      result = 0.0;
      std::size_t index = 0;
      const IntegerVector& foldId(boost::get<IntegerVector>(data.foldId));
      const double adjustment = GumbelRegression::EulerMascheroniConstant() * std::exp(w[0]);
      for(std::size_t yi = 0;yi < data.y.size();yi++) {
        if (foldId[yi] != foldTarget) continue;
        double error = (mu[index] + adjustment - data.y[yi]);
        result += error * error;
        index++;
      }
    }
  }

  // This is running in a thread
  void train(std::size_t foldTarget) {
    auto verbose_printer = [&foldTarget](const char* s) {
      std::cout << "(" << foldTarget << ") " << s << std::endl;
    };
    auto silent_printer = [](const char* s) {
    };
    auto& gumbelCoef(gumbelCoefList[foldTarget - 1]);
    std::vector<double> w1(data.ncolX + 1, 0.0), w2(data.ncolX + 1, 0.0);
    w1[0] = init_log_sigma;
    w1[1] = init_intercept;
    double *pw1 = &w1[0];
    double **pw = &pw1;
    bool is_converge;
    GumbelRegression::ComputationArguments args(data, (foldTarget == nfold + 1 ? 0 : foldTarget));
    GumbelRegression::GumbelRegressionLoss loss(data, args);
    GumbelRegression::GumbelRegressionGradient grad(data, args);
    GumbelRegression::GumbelRegressionHessianV hv(data, args);
    args.set_pw(pw);
    // using dlib bfgs to solve scale
    typedef dlib::matrix<double, 0, 1> DLibVector;
    DLibVector log_sigma(1);

    auto location_kernel = init_tronC(loss, grad, hv, data.ncolX);
    for(std::size_t lambda_i = 0;lambda_i < lambdaSeq.size();lambda_i++) {
      double lambda = lambdaSeq[lambda_i];
      is_converge = false;
      args.set_l2(lambda);
      while(true) {
        w2 = w1;
        args.set_loss_type(GumbelRegression::LossType::Scale);
        log_sigma(0) = w1[0];
        dlib::find_min_box_constrained(
          dlib::bfgs_search_strategy(),
          dlib::objective_delta_stop_strategy(tolerance),
          [&loss, &foldTarget, this](const DLibVector& log_sigma) {
            const double *p = log_sigma.begin();
            double result = loss(p);
            if (verbose) {
              std::cout << "(" << foldTarget << ")log(sigma): " << *p << std::endl;
              std::cout << "(" << foldTarget << ")f: " << result << std::endl;
            }
            return result;
          },
          [&grad, &foldTarget, this](const DLibVector& log_sigma) {
            const double *p = log_sigma.begin();
            DLibVector result(1);
            double *g = result.begin();
            grad(p, g);
            if (verbose) {
              std::cout << "(" << foldTarget << ")g: " << result(0) << std::endl;
            }
            return result;
          },
          log_sigma,
          w1[0] - 1,
          w1[0] + 1
        );
        w1[0] = log_sigma(0);
        args.set_loss_type(GumbelRegression::LossType::Location);
        if (verbose) {
          tronC(location_kernel, pw1 + 1, tolerance, verbose_printer);
        } else {
          tronC(location_kernel, pw1 + 1, tolerance, silent_printer);
        }
        if (verbose) {
          std::cout << "(" << foldTarget << ")dist: " << dist(w1, w2) << std::endl;
        }
        is_converge = dist(w1, w2) < tolerance;
        if (is_converge) break;
      }
      double *presult = &gumbelCoef(0, lambda_i);
      std::copy(w1.begin(), w1.end(), presult);
    }
  }

  void operator()(std::size_t begin, std::size_t end) {
    for(std::size_t foldTarget = begin;foldTarget < end;foldTarget++) {
      train(foldTarget);
      cvPredict(foldTarget);
    }
  }
private:

  static const double dist(const std::vector<double>& w1, const std::vector<double>& w2) {
    double result = 0, tmp;
    for(std::size_t i = 0;i < w1.size();i++) {
      tmp = std::abs(w1[i] - w2[i]);
      if (tmp > result) result = tmp;
    }
    return result;
  }

};

static int ncol(const S4& X) {
  const IntegerVector Dim(X.slot("Dim"));
  return Dim[1];
}

//[[Rcpp::export(.gumbelRegression.cpp.internal)]]
List gumbelRegressionCpp(S4 X, NumericVector y, IntegerVector foldId, NumericVector lambdaSeq, double tolerance, double init_log_sigma, double init_intercept, bool verbose = false, bool parallel = true) {
  int nfold = Rcpp::max(foldId);
  std::vector<NumericMatrix> gumbelCoefList;
  std::vector<NumericVector> cvMseList;
  for(int i = 0;i < nfold;i++) {
    gumbelCoefList.push_back(NumericMatrix(1 + ncol(X), lambdaSeq.size()));
    cvMseList.push_back(NumericVector(lambdaSeq.size()));
  }
  gumbelCoefList.push_back(NumericMatrix(1 + ncol(X), lambdaSeq.size()));
  GumbelRegression::GumbelRegressionData data(X, y, foldId);
  FoldTask foldTask(data, lambdaSeq, tolerance, gumbelCoefList, cvMseList, nfold, init_log_sigma, init_intercept, verbose);
  if (parallel) {
    Rcout << "Fit gumbel regression in parallel mode" << std::endl;
    RcppParallel::parallelFor(1, nfold + 2, foldTask);
  } else {
    Rcout << "Fit gumbel regression in single mode" << std::endl;
    for(std::size_t i = 1;i < nfold + 2;i++) {
      foldTask(i, i + 1);
    }
  }
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
