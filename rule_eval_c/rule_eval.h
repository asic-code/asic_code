#ifndef RULE_EVAL_H
#define RULE_EVAL_H

#include <stdint.h>
#include <stddef.h>

/* Popcount LUT for bytes (256 entries). */
extern const unsigned char rule_eval_popcount_lut[256];

/* Popcount of packed bitset (uint8 array of length n_bytes). */
uint64_t rule_eval_popcount_packed(const uint8_t *packed, size_t n_bytes);

/* AND-reduce: out |= cond_masks[cond_ids[i]] for i in [0, n). in-place into out. */
void rule_eval_and_reduce(
    uint8_t *out,
    const int64_t *cond_ids,
    size_t n_cond,
    const uint8_t *cond_bitsets,
    size_t n_conds,
    size_t n_bytes
);

/* Kernel: for each rule, AND-reduce conds then popcount. Writes counts into scores[0..n_rules-1].
 * If out_masks != NULL, it must point to n_rules * n_bytes bytes; for each rule the row-level
 * packed mask is written to out_masks + r * n_bytes. */
void rule_eval_kernel(
    const int64_t *rule_ptr,      /* length n_rules+1 */
    const int64_t *rule_cond_ids,
    const uint8_t *cond_bitsets,  /* (n_conds, n_bytes) row-major */
    size_t n_rules,
    size_t n_conds,
    size_t n_bytes,
    int64_t *scores,
    uint8_t *out_masks            /* optional: (n_rules, n_bytes) row-major; may be NULL */
);

#endif /* RULE_EVAL_H */
