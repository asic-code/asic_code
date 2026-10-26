"""
rule_eval_c: C extension for bitset rule evaluation (AND-reduce + popcount).
Use: pip install -e . from this directory, then import rule_eval_c; rule_eval_c.eval_rules_c(...)
"""
try:
    from rule_eval_c._rule_eval_c import eval_rules_c
except ImportError:
    eval_rules_c = None
