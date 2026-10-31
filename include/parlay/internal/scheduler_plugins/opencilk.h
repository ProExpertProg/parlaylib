#ifndef PARLAY_INTERNAL_SCHEDULER_PLUGINS_OPENCILK_H_
#define PARLAY_INTERNAL_SCHEDULER_PLUGINS_OPENCILK_H_

#include <cstddef>

#include <type_traits>

#include <cilk/cilk.h>
#include <cilk/cilk_api.h>

namespace parlay {

// IWYU pragma: private, include "../../parallel.h"

inline size_t num_workers() { return __cilkrts_get_nworkers(); }

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"

inline size_t worker_id() { return __cilkrts_get_worker_number(); }

#pragma clang diagnostic pop

template <typename Lf, typename Rf>
inline void par_do(Lf&& left, Rf&& right, bool) {
  static_assert(std::is_invocable_v<Lf&&>);
  static_assert(std::is_invocable_v<Rf&&>);
  cilk_spawn std::forward<Rf>(right)();
  std::forward<Lf>(left)();
  cilk_sync;
}

// Benchmarking fork: every parallel_for is a cilk_for, so that the runtime grainsize (CILK_GRAINSIZE) and the compile-time
// coarsening factor apply to all loops. The granularity requested by the caller is ignored; the unmodified plugin only uses
// cilk_for for granularity 0 and otherwise runs small ranges serially and splits larger ones with a cilk_spawn recursion.
template <typename F>
inline void parallel_for(size_t start, size_t end, F&& f, long, bool) {
  static_assert(std::is_invocable_v<F&, size_t>);
  cilk_for (size_t i=start; i<end; i++) f(i);
}

template <typename... Fs>
void execute_with_scheduler(Fs...) {
  struct Illegal {};
  static_assert((std::is_same_v<Illegal, Fs> && ...), "parlay::execute_with_scheduler is only available in the Parlay scheduler and is not compatible with OpenCilk");
}

}  // namespace parlay

#endif  // PARLAY_INTERNAL_SCHEDULER_PLUGINS_OPENCILK_H_

