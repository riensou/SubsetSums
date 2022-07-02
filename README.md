# SubsetSums

Let S be a set from 1 to n, {1, 2, ..., n}. Find the number of subsets of S whose elements sum to an integer divisible by k.

Inspired by this 3Blue1Brown video: https://www.youtube.com/watch?v=bOXCLR3Wric&t=871s, which deals with the case n = 2000, k = 5.

Short Python script that first generates each subset, then finds the number whose sum is divisible by k. Then uses matplotlib to graph it out. My current conjecture is that it can be modeled as 1/k (2^n + 2^q(max(k - 2^r, 1)) where q and r are defined by the division algorithm (n = qk + r). However, I am yet to prove this, and have only been considering prime numbers for n.

My method of proving this is using generating functions and the roots of unity, as in the 3Blue1Brown video.



I also added a small neural network that attempted to 'learn' the answers, however this proved to computationally inefficient.
