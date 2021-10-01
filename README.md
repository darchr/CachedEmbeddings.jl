# CachedEmbeddings

```julia
# Steps required for `columnpointer`
# 1. Get ahold of the pointer to the column.
# 2. If that pointer is not remote - simply return the pointer.
# 3. If that pointer IS remote.
#   3-1. Try Lock pointer
#       3-1-1. Success: Acquire destination in a cache array.
#           3-1-1-1. Success:
#               Copy over data.
#               Set column index in cache.
#               Update and unlock pointer.
#           3-1-1-2. Failure: Try to acquire lock to allocate a new table.
#               3-1-1-2-1. Success:
#                   Allocate new cache table.
#                   Append cache table to caches.
#                   Goto (3-1-1-1).
#               3-1-1-2-2. Failure:
#                   Implication - another thread is allocating the new table.
#                   Short sleep.
#                   Goto (3-1-1) - hopefully after sleep, we'll be successful this time.
#      3-1-2. Failure: Another thread is moving the data. Simply return the masked pointer.
#
# Questions: How do we know if a pointer is cached?
#   Need to look at the "N" most recent cached where "N" is the number of cached
#   allocated on this lookup.
#
#   Maybe we can tag the lower bits of the locking pointer with a round number and compare
#   it with the current round. Then we could check if a vector has been cached with a single
#   64-bit load.
#
#   How many bits do we have available?
#   Lets assume BF16 elements (2 bytes per element.)
#   With a featuresize of 1 - we bet 1 free bit, which we need for locking.
#   With a feature size of 8, each vector occupies 16 bytes (4 bits of address space.).
#   Thus, we get `4 - 1 = 3` bits for round numbers.
#   This should do a pretty good job.
```
