#include <ctgmath>
#include <fstream>
#include <boost/process/environment.hpp>
#include <Rcpp.h>
#include <RcppParallel.h>
#include <pathwise-l2-cd.h>
#include <HsTrust.h>
#include "gumbel_optimization_stop_strategy.h"
#include "prefixstream.h"

// #define USE_OBJECT_DELTA_STOP_STRATEGY
#define USE_GRADIENT_NORM_STOP_STRATEGY

struct GumbelRegressionFunction : public ::function {

  const GumbelRegression::GumbelRegressionData& data;

  GumbelRegression::ComputationArguments args;

  GumbelRegression::GumbelRegressionLoss loss;

  GumbelRegression::GumbelRegressionGradient gradient;

  GumbelRegression::GumbelRegressionHessianV hv;

public:

  GumbelRegressionFunction(const GumbelRegression::GumbelRegressionData& _data, int foldTarget)
    : data(_data), args(data, foldTarget), loss(data, args), gradient(data, args), hv(data, args)
  { }

  virtual ~GumbelRegressionFunction() { }

  virtual double fun(double *w) {
    double result = loss(w);
    if (std::isinf(result)) throw std::runtime_error("infinite output");
    return result;
  }

  virtual void grad(const double *w, double *g) {
    return gradient(w, g);
  }

  virtual void Hv(const double *s, double *Hs) {
    return hv(s, Hs);
  }

  virtual int get_nr_variable(void) {
    switch(args.get_loss_type()) {
    case GumbelRegression::LossType::Scale :
      return 1;
    case GumbelRegression::LossType::Location :
      return data.ncolX;
    case GumbelRegression::LossType::All :
      return data.ncolX + 1;
    }
  }



};

using namespace Rcpp;

typedef Rcpp::NumericMatrix gumbelCoefType;

class FoldTask : public RcppParallel::Worker {

  const GumbelRegression::GumbelRegressionData& data;

  const NumericVector& lambdaSeq;

  const double tolerance;

  std::vector< gumbelCoefType >& gumbelCoefList;

  const int nfold;

  const double init_log_sigma;

  const double init_intercept;

  const int verbose;

  const IntegerVector& foldId;

  double *pOutOfFoldPrediction;

public:

  FoldTask(
    const GumbelRegression::GumbelRegressionData& _data,
    const NumericVector& _lambdaSeq,
    const double _tolerance,
    std::vector< gumbelCoefType >& _gumbelCoefList,
    int _nfold,
    double _init_log_sigma,
    double _init_intercept,
    int _verbose,
    const IntegerVector& _foldId,
    NumericMatrix& _outOfFoldPrediction) :
  data(_data), lambdaSeq(_lambdaSeq), tolerance(_tolerance),
  gumbelCoefList(_gumbelCoefList),
  nfold(_nfold), init_log_sigma(_init_log_sigma), init_intercept(_init_intercept), verbose(_verbose),
  foldId(_foldId), pOutOfFoldPrediction(&_outOfFoldPrediction[0])
  { }

  ~FoldTask() { }

  // This is running in a thread
  void cvPredict(std::size_t foldTarget) {
    if (nfold + 1 == foldTarget) return;
    auto& gumbelCoef(gumbelCoefList[foldTarget - 1]);
    std::vector<double> mu(data.nrowX, 0.0);
    for(std::size_t lambda_i = 0;lambda_i < lambdaSeq.size();lambda_i++) {
      const double *w = &gumbelCoef(0, lambda_i);
      const double *w1 = w + 1;
      // Xv::Xv_dgCMatrix_numeric_folded(data.X, pw1, buf, data.foldId, foldTarget, true);
      std::fill(mu.begin(), mu.end(), 0.0);
      auto muSize = Xv::Xv_dgCMatrix_numeric_folded(data.X, w1, mu, data.foldId, foldTarget, false);
      int muIndex = 0;
      double *pstart = pOutOfFoldPrediction + lambda_i * data.nrowX;
      for(int row = 0;row < foldId.size();row++) {
        if (foldId[row] == foldTarget) {
          pstart[row] = mu[muIndex++];
        }
      }
      if (muIndex != muSize) throw std::logic_error("muIndex != muSize");
    }
  }

  // This is running in a thread
  void train(std::size_t foldTarget) {
    std::unique_ptr<std::iostream> progress_logger;
    if (verbose > 2) {
      std::stringstream progress_logger_path;
      progress_logger_path << ".gumbelRegression-pathwise-l2-cd." <<
        boost::this_process::get_id() << "." << foldTarget << ".log";
      progress_logger.reset(new std::fstream(progress_logger_path.str(), std::iostream::out));
    }
    // auto verbose_printer = [&foldTarget](const char* s) {
    //   std::cout << "(" << foldTarget << ") " << s << std::endl;
    // };
    // auto silent_printer = [](const char* s) {
    // };
    auto& gumbelCoef(gumbelCoefList[foldTarget - 1]);
    bool is_converge;
    GumbelRegressionFunction grfunction(data, (foldTarget == nfold + 1 ? 0 : foldTarget));
    typedef dlib::matrix<double, 0, 1> DLibVector;
    DLibVector log_sigma(1);
    log_sigma(0) = init_log_sigma;
    DLibVector w1(data.ncolX);
    std::fill(w1.begin(), w1.end(), 0.0);
    w1(0) = init_intercept;
    std::vector<double> w2(data.ncolX + 1, 0.0);
    // compute reference loss
    w2[0] = init_log_sigma;
    w2[1] = init_intercept;
    grfunction.args.set_loss_type(GumbelRegression::LossType::All);
    double reference_loss = grfunction.loss(&w2[0]);
#ifdef USE_OBJECT_DELTA_STOP_STRATEGY
    double cd_threshold = reference_loss * tolerance;
#endif
#ifdef USE_GRADIENT_NORM_STOP_STRATEGY
    double cd_threshold = tolerance;
#endif
    double last_log_sigma;
    double *pw2 = &w2[0];
    grfunction.args.set_pw(&pw2);
    DLibVector mg1(1), mg2(data.ncolX);
#ifdef USE_GRADIENT_NORM_STOP_STRATEGY
    std::vector<double> stop_strategy_gradient_buffer(1 + data.ncolX, 0.0);
#endif
    double *pmg1 = mg1.begin(), *pmg2 = mg2.begin();
    std::shared_ptr<TRON> location_tron(HsTrust::init_tron(&grfunction, tolerance, 100), [](TRON* tron) {
      HsTrust::finalize_tron(tron);
    });
    if (verbose > 1) {
      HsTrust::set_print_string(location_tron.get(), [&foldTarget](const char* s) {
        std::cout << "(" << foldTarget << ") " << s << std::endl;
      });
    } else {
      HsTrust::set_print_string(location_tron.get(), [&foldTarget](const char* s) { });
    }
    double last_loss = reference_loss, current_loss = reference_loss;
    for(std::size_t lambda_i = 0;lambda_i < lambdaSeq.size();lambda_i++) {
      std::stringstream prefixss;
      prefixss << "(" << foldTarget << "-" << lambda_i << ") ";
      GumbelRegression::oprefixstream ost(prefixss.str(), std::cout);
      double lambda = lambdaSeq[lambda_i];
      if (verbose > 0) std::cout << "(" << foldTarget << "-" << lambda_i << ") lambda: " << lambda << std::endl;
      if (verbose > 2) *progress_logger << "lambda: " << lambda << std::endl;
      is_converge = false;
      grfunction.args.set_l2(lambda);
      w2[0] = log_sigma(0);
      std::copy(w1.begin(), w1.end(), w2.begin() + 1);
      while(true) {
        double current_log_sigma, loss_result;
        grfunction.args.set_loss_type(GumbelRegression::LossType::Scale);
        // optimizing scale parameter
        if (verbose > 2) {
          *progress_logger << "optimizing sigma... ";
          progress_logger->flush();
        }
        bool successfully_search_once = false;
        do {
          current_log_sigma = log_sigma(0);
          try {
            dlib::find_min_box_constrained(
              dlib::bfgs_search_strategy(),
              dlib::objective_delta_stop_strategy(tolerance),
              [&](const DLibVector& log_sigma) {
                loss_result = grfunction.loss(log_sigma.begin());
                return loss_result;
              },
              [&](const DLibVector& log_sigma) {
                grfunction.grad(log_sigma.begin(), pmg1);
                return mg1;
              },
              log_sigma,
              log_sigma(0) - 1,
              log_sigma(0) + 1
            );
            successfully_search_once = true;
          } catch (std::runtime_error& e) {
            std::cout << "(" << foldTarget << "-" << lambda_i << ") scale: got infinite result";
            std::cout << " ( f(" << log_sigma(0) << ") = " << grfunction.loss(log_sigma.begin()) << ")" << std::endl;
            if (successfully_search_once) {
              log_sigma(0) = current_log_sigma;
              break;
            } else {
              log_sigma(0) = current_log_sigma + 2;
            }
          } catch (dlib::error& e) {
            double current_loss = grfunction.loss(log_sigma.begin());
            std::cout << "(" << foldTarget << "-" << lambda_i << ") scale: got dlib error: " << e.what();
            std::cout << " ( f(" << log_sigma(0) << ", " << w1(0) << ") = " << current_loss << ")" << std::endl;
            if (successfully_search_once) {
              log_sigma(0) = current_log_sigma;
              break;
            } else {
              log_sigma(0) = current_log_sigma + 2;
            }
          }
        } while (std::abs(log_sigma(0) - current_log_sigma) > tolerance);
        if (verbose > 1) {
          std::cout << "(" << foldTarget << "-" << lambda_i << ") finish scale part. log(sigma): " << log_sigma(0) <<
            ", f: " << loss_result <<
            ", |g|: " << std::abs(mg1(0)) << std::endl;
        }
        if (verbose > 2) {
          *progress_logger << "log(sigma): " << log_sigma(0) <<
            ", f: " << loss_result <<
            ", |g|: " << std::abs(mg1(0)) << std::endl;
        }
        last_log_sigma = w2[0];
        w2[0] = log_sigma(0);
        // optimizing location
        grfunction.args.set_loss_type(GumbelRegression::LossType::Location);
        if (verbose > 2) {
          *progress_logger << "optimizing location parameters... ";
          progress_logger->flush();
        }
        // HsTrust::tron(location_tron.get(), w1.begin());
#ifdef USE_OBJECT_DELTA_STOP_STRATEGY
        typedef GumbelRegression::objective_delta_stop_strategy StopStrategy;
#endif
#ifdef USE_GRADIENT_NORM_STOP_STRATEGY
        typedef GumbelRegression::gradient_norm_stop_strategy StopStrategy;
#endif
        std::shared_ptr<StopStrategy> pstop_strategy(NULL);
        if (std::abs(last_log_sigma - w2[0]) > 0.5) {
          pstop_strategy.reset(new StopStrategy(tolerance, 10));
        } else {
          pstop_strategy.reset(new StopStrategy(tolerance));
        }
        if (verbose > 1) {
          pstop_strategy->be_verbose(ost);
        }
        dlib::find_min(
          dlib::lbfgs_search_strategy(30),
          *pstop_strategy,
          [&](const DLibVector& mw1) {
            loss_result = grfunction.loss(mw1.begin());
            return loss_result;
          },
          [&](const DLibVector& mw1) {
            grfunction.grad(mw1.begin(), pmg2);
            return mg2;
          },
          w1,
          -1
        );
        loss_result = grfunction.fun(w1.begin());
        grfunction.grad(w1.begin(), pmg2);
        if (verbose > 1) {
          std::cout << "(" << foldTarget << "-" << lambda_i << ") finish location part. f: " << loss_result <<
            ", |g|: " << std::sqrt(std::accumulate(mg2.begin(), mg2.end(), 0.0, [](const double left, const double right) {
              return left + right * right;
            })) << std::endl;
        }
        if (verbose > 2) {
          *progress_logger << "f: " << loss_result <<
            ", |g|: " << std::sqrt(std::accumulate(mg2.begin(), mg2.end(), 0.0, [](const double left, const double right) {
              return left + right * right;
            })) << std::endl;
        }
        // copy updated result to w2
        w2[0] = log_sigma(0);
        std::copy(w1.begin(), w1.end(), w2.begin() + 1);
        // compute latest loss
        grfunction.args.set_loss_type(GumbelRegression::LossType::All);
        current_loss = grfunction.loss(&w2[0]);
#ifdef USE_OBJECT_DELTA_STOP_STRATEGY
        double distance = std::abs(last_loss - current_loss);
        if (verbose > 0) {
          std::cout << "(" << foldTarget << "-" << lambda_i <<
            ") The coordinate descent improves the loss from " <<
            last_loss << " to " <<
            current_loss << "(improves: " <<
            distance << ", reference: " <<
            reference_loss << ", threshold: " <<
            cd_threshold << ")" << std::endl;
        }
        is_converge = distance < cd_threshold;
#endif
        last_loss = current_loss;
#ifdef USE_GRADIENT_NORM_STOP_STRATEGY
        std::fill(stop_strategy_gradient_buffer.begin(), stop_strategy_gradient_buffer.end(), 0.0);
        grfunction.grad(&w2[0], &stop_strategy_gradient_buffer[0]);
        double gnorm = std::accumulate(stop_strategy_gradient_buffer.begin(), stop_strategy_gradient_buffer.end(), 0.0, [](const double left, const double right) {
          return left + right * right;
        });
        if (verbose > 0) {
          std::cout << "(" << foldTarget << "-" << lambda_i <<
            ") The coordinate descent improves the norm of the gradient to " <<
            gnorm << " ( threshold: " <<
            cd_threshold << ")" << std::endl;
        }
        is_converge = gnorm < cd_threshold;
#endif
        if (is_converge) break;
      }
      if (verbose > 2) {
        *progress_logger << "done!" << std::endl;
      }
      double *presult = &gumbelCoef(0, lambda_i);
      presult[0] = log_sigma(0);
      std::copy(w1.begin(), w1.end(), presult + 1);
    }
  }

  void operator()(std::size_t begin, std::size_t end) {
    for(std::size_t foldTarget = begin;foldTarget < end;foldTarget++) {
      try {
        train(foldTarget);
        cvPredict(foldTarget);
      } catch (std::exception& e) {
        std::cerr << "(" << foldTarget << ") An error occurred. The message is: " << e.what() << std::endl;
        throw;
      }
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
  for(int i = 0;i < nfold;i++) {
    gumbelCoefList.push_back(gumbelCoefType(NumericMatrix(1 + ncol(X), lambdaSeq.size())));
  }
  gumbelCoefList.push_back(gumbelCoefType(NumericMatrix(1 + ncol(X), lambdaSeq.size())));
  GumbelRegression::GumbelRegressionData data(X, y, foldId);
  NumericMatrix outOfFoldPrediction(data.nrowX, lambdaSeq.size());
  FoldTask foldTask(data, lambdaSeq, tolerance, gumbelCoefList, nfold, init_log_sigma, init_intercept, verbose, foldId, outOfFoldPrediction);
  HsTrust::init();
  if (parallel) {
    std::cout << "Fit gumbel regression in parallel mode" << std::endl;
    RcppParallel::parallelFor(1, nfold + 2, foldTask);
  } else {
    std::cout << "Fit gumbel regression in single mode" << std::endl;
    for(std::size_t i = 1;i < nfold + 2;i++) {
      foldTask(i, i + 1);
    }
  }
  List resultCoef(nfold + 1), result;
  for(int i = 0;i < nfold + 1;i++) {
    resultCoef[i] = gumbelCoefList[i];
  }
  result["coef"] = resultCoef;
  result["fit.preval"] = outOfFoldPrediction;
  return result;
}
