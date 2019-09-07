import numpy as np
import pdb

def search_optimal_top(pis, cumulative_ratio=0.9):
  # a bit complicated
  # as long as the first solution is found, stop and return the result
  # the search begins from low bit to high bit

  K = len(pis)
  Bit = int(np.log2(K))
  results_failed = []
  for b in range(1, Bit+1):
    results = {}

    if b == 1:
      # 1-bit: free combination
      for i in range(K):
        for j in range(i+1, K):
          results[(i, j)] = pis[i]+pis[j]
          if results[(i, j)] > cumulative_ratio:
            return (i, j), 1

    else:
      # for bits > 1, follow the algo
      k = 2 ** b
      imax = int(np.ceil((K-1)/k)) # interval_max
      for i in range(1, imax+1):
        # i: interval
        nums = int(K - (1+i*(k-1)) + 1)
        for n in range(nums):
          index = tuple([n+i*count for count in range(k)])
          results[index] = np.sum(pis[list(index)])
          if results[index] > cumulative_ratio:
            return index, b

    results_failed.append(results)

BITS = 8
quant_points = np.linspace(0., 1., 2**BITS)
pis = np.random.randn(2**BITS)
pis = pis / np.sum(pis)

# Test cases
# pis = np.asarray([0.15, 0.05, 0.1, 0.1, 0.3, 0.05, 0.2, 0.05])
# pis = [0.1, 0.8, 0.05, 0.05]

top_ind, b = search_optimal_top(pis, 0.9)
print(pis)
print(b, top_ind)
top_quant_points = quant_points[list(top_ind)]

