/**
 * Source: https://stackoverflow.com/a/27336473/1182304
 */

#ifndef __PREFIXSTREAM_H__
#define __PREFIXSTREAM_H__

#include <iostream>

namespace GumbelRegression {

class prefixbuf : public std::streambuf {
  std::string     prefix;
  std::streambuf* sbuf;
  bool            need_prefix;

  int sync() {
    return this->sbuf->pubsync();
  }
  int overflow(int c) {
    if (c != std::char_traits<char>::eof()) {
      if (this->need_prefix
            && !this->prefix.empty()
            && this->prefix.size() != this->sbuf->sputn(&this->prefix[0], this->prefix.size())) {
            return std::char_traits<char>::eof();
      }
      this->need_prefix = c == '\n';
    }
    return this->sbuf->sputc(c);
  }
public:
  prefixbuf(std::string const& prefix, std::streambuf* sbuf)
    : prefix(prefix)
  , sbuf(sbuf)
  , need_prefix(true) {
  }
};

class oprefixstream : private virtual prefixbuf, public std::ostream {
public:
  oprefixstream(std::string const& prefix, std::ostream& out)
    : prefixbuf(prefix, out.rdbuf())
  , std::ios(static_cast<std::streambuf*>(this))
  , std::ostream(static_cast<std::streambuf*>(this)) {
  }
};

}

#endif //__PREFIXSTREAM_H__
// int main()
// {
//     oprefixstream out("prefix: ", std::cout);
//     out << "hello\n"
//         << "world\n";
// }
