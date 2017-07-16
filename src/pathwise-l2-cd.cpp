#include <Rcpp.h>
#include <RcppParallel.h>
#include <pathwise-l2-cd.h>
using namespace Rcpp;

typedef Rcpp::NumericMatrix gumbelCoefType;
typedef Rcpp::NumericVector cvMseType;

class FoldTask : public RcppParallel::Worker {

  const GumbelRegression::GumbelRegressionData& data;

  const NumericVector& lambdaSeq;

  const double tolerance;

  std::vector< gumbelCoefType >& gumbelCoefList;

  std::vector< cvMseType >& cvMseList;

  const int nfold;

  const double init_log_sigma;

  const double init_intercept;

  const int verbose;

public:

  FoldTask(
    const GumbelRegression::GumbelRegressionData& _data,
    const NumericVector& _lambdaSeq,
    const double _tolerance,
    std::vector< gumbelCoefType >& _gumbelCoefList,
    std::vector< cvMseType >& _cvMseList,
    int _nfold,
    double _init_log_sigma,
    double _init_intercept,
    int _verbose) :
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
    bool is_converge;
    GumbelRegression::ComputationArguments args(data, (foldTarget == nfold + 1 ? 0 : foldTarget));
    GumbelRegression::GumbelRegressionLoss loss(data, args);
    GumbelRegression::GumbelRegressionGradient grad(data, args);
    GumbelRegression::GumbelRegressionHessianV hv(data, args);
    typedef dlib::matrix<double, 0, 1> DLibVector;
    DLibVector log_sigma(1);
    log_sigma(0) = init_log_sigma;
    DLibVector w1(data.ncolX);
    w1(0) = init_intercept;
    std::vector<double> w2(data.ncolX + 1, 0.0);
    double last_log_sigma;
    double *pw2 = &w2[0];
    args.set_pw(&pw2);
    DLibVector mg1(1), mg2(data.ncolX);
    double *pmg1 = mg1.begin(), *pmg2 = mg2.begin();
    // auto location_kernel = init_tronC(loss, grad, hv, data.ncolX);

    for(std::size_t lambda_i = 0;lambda_i < lambdaSeq.size();lambda_i++) {
      double lambda = lambdaSeq[lambda_i];
      if (verbose > 0) std::cout << "(" << foldTarget << ") lambda: " << lambda << std::endl;
      is_converge = false;
      args.set_l2(lambda);
      while(true) {
        w2[0] = log_sigma(0);
        std::copy(w1.begin(), w1.end(), w2.begin() + 1);


        args.set_loss_type(GumbelRegression::LossType::Scale);
        double current_log_sigma, loss_result;
        do {
          current_log_sigma = log_sigma(0);
          dlib::find_min_box_constrained(
            dlib::bfgs_search_strategy(),
            dlib::objective_delta_stop_strategy(tolerance),
            [&](const DLibVector& log_sigma) {
              loss_result = loss(log_sigma.begin());
              return loss_result;
            },
            [&](const DLibVector& log_sigma) {
              grad(log_sigma.begin(), pmg1);
              return mg1;
            },
            log_sigma,
            log_sigma(0) - 1,
            log_sigma(0) + 1
          );
        } while (std::abs(log_sigma(0) - current_log_sigma) > tolerance);
        if (verbose > 1) {
          std::cout << "(" << foldTarget << " scale) log(sigma): " << log_sigma(0) <<
            ", f: " << loss_result <<
            ", |g|: " << std::abs(mg1(0)) << std::endl;
        }
        last_log_sigma = w2[0];
        w2[0] = log_sigma(0);
        args.set_loss_type(GumbelRegression::LossType::Location);
        dlib::find_min(
          dlib::lbfgs_search_strategy(1000),
          dlib::objective_delta_stop_strategy(tolerance),
          [&](const DLibVector& mw1) {
            loss_result = loss(mw1.begin());
            return loss_result;
          },
          [&](const DLibVector& mw1) {
            grad(mw1.begin(), pmg2);
            return mg2;
          },
          w1,
          -1
        );
        if (verbose > 1) {
          std::cout << "(" << foldTarget << " location) f: " << loss_result <<
            ", |g|: " << std::sqrt(std::accumulate(mg2.begin(), mg2.end(), 0.0, [](const double left, const double right) {
              return left + right * right;
            })) << std::endl;
        }
        double distance = std::abs(log_sigma(0) - last_log_sigma);
        for(auto i = 0;i < w1.size();i++) {
          double e = std::abs(w1(i) - w2[i + 1]);
          if (e > distance) distance = e;
        }
        if (verbose > 1) {
          std::cout << "(" << foldTarget << ")dist: " << distance << std::endl;
        }
        is_converge = distance < tolerance;
        if (is_converge) break;
      }
      double *presult = &gumbelCoef(0, lambda_i);
      presult[0] = log_sigma(0);
      std::copy(w1.begin(), w1.end(), presult + 1);
    }
  }

  void operator()(std::size_t begin, std::size_t end) {
    for(std::size_t foldTarget = begin;foldTarget < end;foldTarget++) {
      train(foldTarget);
      cvPredict(foldTarget);
    }
  }
private:

};

static int ncol(const S4& X) {
  const IntegerVector Dim(X.slot("Dim"));
  return Dim[1];
}

//[[Rcpp::export(.gumbelRegression.cpp.internal)]]
List gumbelRegressionCpp(S4 X, NumericVector y, IntegerVector foldId, NumericVector lambdaSeq, double tolerance, double init_log_sigma, double init_intercept, int verbose = 0, bool parallel = true) {
  int nfold = Rcpp::max(foldId);
  std::vector< gumbelCoefType > gumbelCoefList;
  std::vector< cvMseType > cvMseList;
  for(int i = 0;i < nfold;i++) {
    gumbelCoefList.push_back(gumbelCoefType(NumericMatrix(1 + ncol(X), lambdaSeq.size())));
    cvMseList.push_back(cvMseType(NumericVector(lambdaSeq.size())));
  }
  gumbelCoefList.push_back(gumbelCoefType(NumericMatrix(1 + ncol(X), lambdaSeq.size())));
  GumbelRegression::GumbelRegressionData data(X, y, foldId);
  FoldTask foldTask(data, lambdaSeq, tolerance, gumbelCoefList, cvMseList, nfold, init_log_sigma, init_intercept, verbose);
  if (parallel) {
    std::cout << "Fit gumbel regression in parallel mode" << std::endl;
    RcppParallel::parallelFor(1, nfold + 2, foldTask);
  } else {
    std::cout << "Fit gumbel regression in single mode" << std::endl;
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
