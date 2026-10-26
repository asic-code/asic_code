/*
 * Rule evaluation kernel: bitset AND + popcount.
 * Layer 3 only; no Python, no rule/cond management.
 */
#include "rule_eval.h"
#include <stdlib.h>
#include <string.h>

/* LUT: popcount for 0..255 */
const unsigned char rule_eval_popcount_lut[256] = {
    0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
    1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
    1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
    1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
    3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
    4, 5, 5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8
};

uint64_t rule_eval_popcount_packed(const uint8_t *packed, size_t n_bytes)
{
    uint64_t total = 0;
    size_t i;
    for (i = 0; i < n_bytes; i++) {
        total += rule_eval_popcount_lut[packed[i]];
    }
    return total;
}

void rule_eval_and_reduce(
    uint8_t *out,
    const int64_t *cond_ids,
    size_t n_cond,
    const uint8_t *cond_bitsets,
    size_t n_conds,
    size_t n_bytes
)
{
    size_t i, j;
    if (n_cond == 0)
        return;
    /* First cond: copy into out */
    {
        int64_t c0 = cond_ids[0];
        if (c0 < 0 || (size_t)c0 >= n_conds) {
            memset(out, 0, n_bytes);
            return;
        }
        memcpy(out, cond_bitsets + (size_t)c0 * n_bytes, n_bytes);
    }
    for (i = 1; i < n_cond; i++) {
        int64_t c = cond_ids[i];
        if (c < 0 || (size_t)c >= n_conds) {
            memset(out, 0, n_bytes);
            return;
        }
        const uint8_t *row = cond_bitsets + (size_t)c * n_bytes;
        for (j = 0; j < n_bytes; j++)
            out[j] &= row[j];
        /* Optional: early-out if out is all zeros (popcount could be 0) */
        {
            size_t k;
            int any = 0;
            for (k = 0; k < n_bytes; k++) {
                if (out[k] != 0) { any = 1; break; }
            }
            if (!any)
                return;
        }
    }
}

void rule_eval_kernel(
    const int64_t *rule_ptr,
    const int64_t *rule_cond_ids,
    const uint8_t *cond_bitsets,
    size_t n_rules,
    size_t n_conds,
    size_t n_bytes,
    int64_t *scores,
    uint8_t *out_masks
)
{
    size_t r;
    uint8_t *work;
    if (n_bytes == 0 || n_rules == 0) {
        for (r = 0; r < n_rules; r++)
            scores[r] = 0;
        return;
    }
    work = (uint8_t *)malloc(n_bytes);
    if (!work) {
        for (r = 0; r < n_rules; r++)
            scores[r] = -1; /* signal alloc failure if needed */
        return;
    }
    for (r = 0; r < n_rules; r++) {
        size_t start = (size_t)rule_ptr[r];
        size_t end   = (size_t)rule_ptr[r + 1];
        size_t n_cond = end - start;
        uint64_t cnt;
        if (n_cond == 0) {
            scores[r] = 0;
            if (out_masks)
                memset(out_masks + r * n_bytes, 0, n_bytes);
            continue;
        }
        rule_eval_and_reduce(work, rule_cond_ids + start, n_cond,
                             cond_bitsets, n_conds, n_bytes);
        cnt = rule_eval_popcount_packed(work, n_bytes);
        scores[r] = (int64_t)cnt;
        if (out_masks)
            memcpy(out_masks + r * n_bytes, work, n_bytes);
    }
    free(work);
}
