#ifndef __GUMBEL_H__
#define __GUMBEL_H__

#include <Rcpp.h>
#include <boost/variant.hpp>
#include <Xv.h>

namespace GumbelRegression {

typedef enum {
    Scale = 0,
    Location = 1,
    All = 2
  } LossType;

inline int nrow(const Rcpp::S4& X) {
  return INTEGER(X.attr("Dim"))[0];
}

inline int ncol(const Rcpp::S4& X) {
  return INTEGER(X.attr("Dim"))[1];
}

struct GumbelRegressionData {

  const Rcpp::S4& X;

  const int ncolX;

  const int nrowX;

  const Rcpp::NumericVector& y;

  const boost::variant<std::nullptr_t,Rcpp::IntegerVector>& foldId;

  GumbelRegressionData(
    const Rcpp::S4& _X,
    const Rcpp::NumericVector& _y,
    const boost::variant<std::nullptr_t,Rcpp::IntegerVector>& _foldId
  ) :
    X(_X), nrowX(nrow(_X)), ncolX(ncol(X)), y(_y), foldId(_foldId)
  { }

};

class ComputationArguments {

  const GumbelRegressionData& data;

  int foldTarget;

  double l2;

  LossType lossType;

  std::vector<double> buf, buf2;

  double** pw;

  double* pw0;

  double* pw1;

  double* last_w;

public:

  ComputationArguments(const GumbelRegressionData& _data, const int _foldTarget)
    :
    data(_data), foldTarget(_foldTarget), l2(0.0), lossType(LossType::All), 
    buf(_data.nrowX, 0.0), buf2(_data.nrowX, 0.0), 
    pw(nullptr), pw0(nullptr), pw1(nullptr)
  { }

  const int get_foldTarget() const {
    return foldTarget;
  }

  void set_l2(double _l2) {
    l2 = _l2;
  }

  const double get_l2() const {
    return l2;
  }

  void set_pw(double** _pw) {
    pw = _pw;
    pw0 = *pw;
    pw1 = pw0 + 1;
  }

  const double * get_pw1(const double* w) const {
    switch(lossType) {
    case LossType::Scale :
      return pw1;
    case LossType::Location :
      return w;
    case LossType::All :
      return w + 1;
    }
  }

  void set_loss_type(LossType _lossType) {
    lossType = _lossType;
  }

  LossType get_loss_type() const {
    return lossType;
  }

  const double get_log_sigma(const double* w) {
    switch(lossType) {
    case LossType::Scale :
      return w[0];
    case LossType::Location :
      return pw0[0];
    case LossType::All :
      return w[0];
    }
  }

  const double get_sigma(const double* w) {
    return std::exp(get_log_sigma(w));
  }

  const std::vector<double>& get_z(const double* w, boost::variant<std::nullptr_t, double> sigma = nullptr, bool compute_z = true) {
    if (compute_z) {
      if (sigma.which() == 0) sigma = get_sigma(w);
      const double * pw1 = get_pw1(w);
      std::fill(buf.begin(), buf.end(), 0);
      Xv::Xv_dgCMatrix_numeric_folded(data.X, pw1, buf, data.foldId, foldTarget, true);
      mu2z(boost::get<double>(sigma));
    }
    return buf;
  }

  const std::vector<double>& get_enzm1(const double* w, boost::variant<std::nullptr_t, double> sigma = nullptr, bool compute_z = true, bool compute_enzm1 = true) {
    if (compute_z) get_z(w, sigma);
    if (compute_enzm1) {
      for(std::size_t i = 0;i < buf.size();i++) {
        buf2[i] = std::exp(-buf[i]) - 1;
      }
    }
    return buf2;
  }

  double * get_last_w() const {
    return last_w;
  }

  void set_last_w(double* w) {
    last_w = w;
  }

private:

  void mu2z(double sigma) {
    for(std::size_t i = 0;i < buf.size();i++) {
      buf[i] = (data.y[i] - buf[i]) / sigma;
    }
  }

};

class GumbelRegressionLoss {

  const GumbelRegressionData& data;

  ComputationArguments& args;

public:

  GumbelRegressionLoss(
    const GumbelRegressionData& _data,
    ComputationArguments& _args
  )
    : data(_data), args(_args)
  { }

  ~GumbelRegressionLoss() { }

  double operator()(double* w) {
    double l2 = args.get_l2();
    if (l2 == 0) return negloglik(w);
    double regularization = 0.0;
    const double * pw1 = args.get_pw1(w);
    for(int i = 1;i < data.ncolX;i++) {
      regularization += pw1[i] * pw1[i];
    }
    return negloglik(w) + l2 * regularization;
  }

private:

  double negloglik(const double* w) {
    double log_sigma = args.get_log_sigma(w);
    const std::vector<double>& z(args.get_z(w, std::exp(log_sigma)));
    double result = 0.0;
    for(std::size_t i = 0;i < z.size();i++) {
      result += z[i] + std::exp(-z[i]);
    }
    result += z.size() * log_sigma;
    return result;
  }

};

class GumbelRegressionGradient {

  const GumbelRegressionData& data;

  ComputationArguments& args;

public:

  GumbelRegressionGradient(
    const GumbelRegressionData& _data,
    ComputationArguments& _args
  )
    : data(_data), args(_args)
  { }

  ~GumbelRegressionGradient() { }

  void operator()(double* w, double* g) {
    args.set_last_w(w);
    double l2 = args.get_l2();
    negloglikGrad(w, g);
    if (l2 == 0) return;
    double * g1;
    switch(args.get_loss_type()) {
    case LossType::Scale :
      return;
    case LossType::Location :
      g1 = g;
      break;
    case LossType::All :
      g1 = g + 1;
      break;
    }
    const double * pw1 = args.get_pw1(w);
    for(std::size_t i = 1;i < data.ncolX;i++) {
      g1[i] += 2 * l2 * pw1[i];
    }
  }

private:

  void negloglikGrad(double* w, double* g) {
    double log_sigma = args.get_log_sigma(w);
    double sigma= std::exp(log_sigma);
    const std::vector<double>& z(args.get_z(w, sigma));
    switch(args.get_loss_type()) {
    case LossType::Scale :
    {
      g[0] = data.nrowX;
      for(std::size_t i = 0;i < z.size();i++) {
        g[0] += z[i] * (std::exp(-z[i]) - 1);
      }
      return;
    }
    case LossType::Location :
    {
      const std::vector<double>& enzm1(args.get_enzm1(w, sigma, false));
      std::fill(g, g + data.ncolX, 0);
      Xv::vX_dgCMatrix_numeric_folded(data.X, enzm1, g, data.foldId, args.get_foldTarget(), true);
      for(size_t i = 0;i < data.ncolX;i++) {
        g[i] = g[i] / sigma;
      }
      return;
    }
    case LossType::All :
    {
      const std::vector<double>& enzm1(args.get_enzm1(w, sigma, false));
      g[0] = data.nrowX;
      double * g1 = g + 1;
      for(std::size_t i = 0;i < z.size();i++) {
        g[0] += z[i] * enzm1[i];
      }
      std::fill(g1, g1 + data.ncolX, 0);
      Xv::vX_dgCMatrix_numeric_folded(data.X, enzm1, g1, data.foldId, args.get_foldTarget(), true);
      for(size_t i = 0;i < data.ncolX;i++) {
        g1[i] = g1[i] / sigma;
      }
      return;
    }
    }
  }

};

class GumbelRegressionHessianV {

  const GumbelRegressionData& data;

  ComputationArguments& args;

public:

  GumbelRegressionHessianV(
    const GumbelRegressionData& _data,
    ComputationArguments& _args
  )
    : data(_data), args(_args)
  { }

  ~GumbelRegressionHessianV() { }

  void operator()(const double* s, double* Hs) {
    negloglikHessianV(s, Hs);
    double l2 = args.get_l2();
    if (l2 == 0) return;
    double *Hs1;
    const double *s1;
    switch(args.get_loss_type()) {
    case LossType::Scale :
      return;
    case LossType::Location :
    {
      Hs1 = Hs;
      s1 = s;
      break;
    }
    case LossType::All :
    {
      Hs1 = Hs + 1;
      s1 = s + 1;
      break;
    }
    }
    for(std::size_t i = 1;i < data.ncolX;i++) {
      Hs1[i] += 2 * l2 * s1[i];
    }
  }

private :

  std::vector<double> buf, buf2;

  void negloglikHessianV(const double* s, double *Hs) {
    double * w = args.get_last_w();
    double sigma = args.get_sigma(w);
    double H11 = 0.0;
    const std::vector<double>& z(args.get_z(w, sigma, false));
    buf.resize(z.size());
    buf2.resize(data.ncolX);
    switch(args.get_loss_type()) {
    case LossType::Scale :
    {
      const std::vector<double>& enzm1(args.get_enzm1(w, sigma, false, true));
      for(std::size_t i = 0;i < z.size();i++) {
        double ez = enzm1[i] + 1;
        H11 += (z[i] * z[i] - z[i]) * ez + z[i];
        buf[i] = ((z[i] - 1) * ez + 1) / sigma;
      }
      std::fill(buf2.begin(), buf2.end(), 0);
      Xv::vX_dgCMatrix_numeric_folded(data.X, buf, buf2, data.foldId, args.get_foldTarget(), true);
      Hs[0] = H11 * s[0];
      break;
    }
    case LossType::Location :
    {
      const std::vector<double>& enzm1(args.get_enzm1(w, sigma, false, false));
      std::fill(buf.begin(), buf.end(), 0);
      Xv::Xv_dgCMatrix_numeric_folded(data.X, s, buf, data.foldId, args.get_foldTarget(), true);
      double sigma2 = sigma * sigma;
      for(std::size_t i = 0;i < z.size();i++) {
        buf[i] = buf[i] * (enzm1[i] + 1) / sigma2;
      }
      std::fill(Hs, Hs + data.ncolX, 0.0);
      Xv::vX_dgCMatrix_numeric_folded(data.X, buf, Hs, data.foldId, args.get_foldTarget(), true);
      break;
    }
    case LossType::All :
    {
      const std::vector<double>& enzm1(args.get_enzm1(w, sigma, false, false));
      for(std::size_t i = 0;i < z.size();i++) {
        double enz = enzm1[i] + 1;
        H11 += (z[i] * z[i] - z[i]) * enz + z[i];
        buf[i] = ((z[i] - 1) * enz + 1) / sigma;
      }
      // buf2 is H1n
      std::fill(buf2.begin(), buf2.end(), 0.0);
      Xv::vX_dgCMatrix_numeric_folded(data.X, buf, buf2, data.foldId, args.get_foldTarget(), true);
      Hs[0] = H11 * s[0];
      const double * s1 = s + 1;
      double * Hs1 = Hs + 1;
      for(std::size_t i = 0;i < data.ncolX;i++) {
        Hs[0] += s1[i] * buf2[i];
        Hs1[i] = buf2[i] * s[0];
      }
      std::fill(buf.begin(), buf.end(), 0);
      Xv::Xv_dgCMatrix_numeric_folded(data.X, s1, buf, data.foldId, args.get_foldTarget(), true);
      double sigma2 = sigma * sigma;
      for(std::size_t i = 0;i < z.size();i++) {
        // Hnn = enz / sigma^2
        buf[i] = buf[i] * (enzm1[i] + 1) / sigma2;
      }
      Xv::vX_dgCMatrix_numeric_folded(data.X, buf, Hs1, data.foldId, args.get_foldTarget(), true);
      break;
    }
    }
  }

};

}

#endif // __GUMBEL_H__
