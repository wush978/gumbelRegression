/**
 * Modifified from http://dlib.net/dlib/optimization/optimization_stop_strategies.h
 */

#ifndef __GUMBEL_REGRESSION_OPTIMIZATION_STOP_STRATEGIES_H__
#define __GUMBEL_REGRESSION_OPTIMIZATION_STOP_STRATEGIES_H__

#include <cmath>
#include <limits>
#include <ostream>
#include <dlib/matrix.h>
#include <dlib/algs.h>
#include <dlib/optimization/optimization_stop_strategies.h>

namespace GumbelRegression {

class objective_delta_stop_strategy
{
public:
  explicit objective_delta_stop_strategy (
      double min_delta = 1e-7
  ) : _verbose(false), _been_used(false), _min_delta(min_delta), _max_iter(0), _cur_iter(0), _prev_funct_value(0), _post(&std::cout)
  {
    DLIB_ASSERT (
        min_delta >= 0,
        "\t objective_delta_stop_strategy(min_delta)"
        << "\n\t min_delta can't be negative"
        << "\n\t min_delta: " << min_delta
    );
  }

  objective_delta_stop_strategy (
      double min_delta,
      unsigned long max_iter
  ) : _verbose(false), _been_used(false), _min_delta(min_delta), _max_iter(max_iter), _cur_iter(0), _prev_funct_value(0), _post(&std::cout)
  {
    DLIB_ASSERT (
        min_delta >= 0 && max_iter > 0,
        "\t objective_delta_stop_strategy(min_delta, max_iter)"
        << "\n\t min_delta can't be negative and max_iter can't be 0"
        << "\n\t min_delta: " << min_delta
        << "\n\t max_iter:  " << max_iter
    );
  }

  objective_delta_stop_strategy& be_verbose(std::ostream& ost) {
    _post = &ost;
    _verbose = true;
    return *this;
  }

  template <typename T>
  bool should_continue_search (
      const T& ,
      const double funct_value,
      const T&
  )
  {
    if (_verbose)
    {
      (*_post) << "iteration: " << _cur_iter << "   objective: " << funct_value << std::endl;
    }

    ++_cur_iter;
    if (_been_used)
    {
      // Check if we have hit the max allowable number of iterations.  (but only
      // check if _max_iter is enabled (i.e. not 0)).
      if (_max_iter != 0 && _cur_iter > _max_iter)
        return false;

      // check if the function change was too small
      if (std::abs(funct_value - _prev_funct_value) < _min_delta)
        return false;
    }

    _been_used = true;
    _prev_funct_value = funct_value;
    return true;
  }

private:
  bool _verbose;

  bool _been_used;
  double _min_delta;
  unsigned long _max_iter;
  unsigned long _cur_iter;
  double _prev_funct_value;
  std::ostream* _post;
};

class gradient_norm_stop_strategy
{
public:
  explicit gradient_norm_stop_strategy (
      double min_norm = 1e-7
  ) : _verbose(false), _min_norm(min_norm), _max_iter(0), _cur_iter(0)
  {
    DLIB_ASSERT (
        min_norm >= 0,
        "\t gradient_norm_stop_strategy(min_norm)"
        << "\n\t min_norm can't be negative"
        << "\n\t min_norm: " << min_norm
    );
  }

  gradient_norm_stop_strategy (
      double min_norm,
      unsigned long max_iter
  ) : _verbose(false), _min_norm(min_norm), _max_iter(max_iter), _cur_iter(0), _post(&std::cout)
  {
    DLIB_ASSERT (
        min_norm >= 0 && max_iter > 0,
        "\t gradient_norm_stop_strategy(min_norm, max_iter)"
        << "\n\t min_norm can't be negative and max_iter can't be 0"
        << "\n\t min_norm: " << min_norm
        << "\n\t max_iter:  " << max_iter
    );
  }

  gradient_norm_stop_strategy& be_verbose(std::ostream& ost) {
    _verbose = true;
    _post = &ost;
    return *this;
  }

  template <typename T>
  bool should_continue_search (
      const T& ,
      const double funct_value,
      const T& funct_derivative
  )
  {
    if (_verbose)
    {
      (*_post) << "iteration: " << _cur_iter << "   objective: " << funct_value << "   gradient norm: " << length(funct_derivative) << std::endl;
    }

    ++_cur_iter;

    // Check if we have hit the max allowable number of iterations.  (but only
    // check if _max_iter is enabled (i.e. not 0)).
    if (_max_iter != 0 && _cur_iter > _max_iter)
      return false;

    // check if the gradient norm is too small
    if (length(funct_derivative) < _min_norm)
      return false;

    return true;
  }

private:
  bool _verbose;

  double _min_norm;
  unsigned long _max_iter;
  unsigned long _cur_iter;
  std::ostream* _post;
};

}

#endif //__GUMBEL_REGRESSION_OPTIMIZATION_STOP_STRATEGIES_H__
