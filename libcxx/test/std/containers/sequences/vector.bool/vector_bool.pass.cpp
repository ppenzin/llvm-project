//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <functional>

// template <class T>
// struct hash
// {
//     size_t operator()(T val) const;
// };

// Not very portable

#include <vector>
#include <cassert>
#include <iterator>
#include <type_traits>

#include "test_macros.h"
#include "min_allocator.h"

TEST_CONSTEXPR_CXX20 bool tests() {
  {
    typedef std::vector<bool> T;
    typedef std::hash<T> H;
#if TEST_STD_VER <= 14
    static_assert((std::is_same<H::argument_type, T>::value), "");
    static_assert((std::is_same<H::result_type, std::size_t>::value), "");
#endif
    ASSERT_NOEXCEPT(H()(T()));

    bool ba[] = {true, false, true, true, false};
    T vb(std::begin(ba), std::end(ba));
    H h;
    assert(h(vb) != 0);
  }
#if TEST_STD_VER >= 11
  {
    typedef std::vector<bool, min_allocator<bool>> T;
    typedef std::hash<T> H;
#  if TEST_STD_VER <= 14
    static_assert((std::is_same<H::argument_type, T>::value), "");
    static_assert((std::is_same<H::result_type, std::size_t>::value), "");
#  endif
    ASSERT_NOEXCEPT(H()(T()));
    bool ba[] = {true, false, true, true, false};
    T vb(std::begin(ba), std::end(ba));
    H h;
    assert(h(vb) != 0);
  }
#endif

  return true;
}

int main(int, char**) {
  tests();
#if TEST_STD_VER > 17
  static_assert(tests());
#endif
  return 0;
}
