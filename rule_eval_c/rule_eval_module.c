/*
 * Python C extension: eval_rules_c(rule_ptr, rule_cond_ids, cond_bitsets, n_conds, n_bytes [, out_masks]).
 * Inputs must be C-contiguous buffers (e.g. numpy arrays).
 * If out_masks is given, it must be writable, shape (n_rules, n_bytes); row-level masks are written there.
 * Returns a list of int (length n_rules = len(rule_ptr)-1).
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "rule_eval.h"
#include <stdint.h>
#include <stdlib.h>

static PyObject *
eval_rules_c_impl(PyObject *self, PyObject *args)
{
    Py_buffer v_rule_ptr = {0}, v_rule_cond_ids = {0}, v_cond_bitsets = {0}, v_out_masks = {0};
    PyObject *n_conds_obj = NULL, *n_bytes_obj = NULL, *out_masks_obj = NULL;
    size_t n_rules, n_conds, n_bytes;
    int64_t *scores = NULL;
    PyObject *result = NULL;
    size_t r;
    int have_out_masks = 0;

    (void)self;
    if (!PyArg_ParseTuple(args, "y*y*y*OO|O",
                          &v_rule_ptr,
                          &v_rule_cond_ids,
                          &v_cond_bitsets,
                          &n_conds_obj,
                          &n_bytes_obj,
                          &out_masks_obj))
        return NULL;

    if (!PyLong_Check(n_conds_obj) || !PyLong_Check(n_bytes_obj)) {
        PyErr_SetString(PyExc_TypeError, "n_conds and n_bytes must be int");
        goto fail;
    }
    n_conds = (size_t)PyLong_AsSize_t(n_conds_obj);
    n_bytes = (size_t)PyLong_AsSize_t(n_bytes_obj);
    if (n_conds == (size_t)-1 && PyErr_Occurred()) goto fail;
    if (n_bytes == (size_t)-1 && PyErr_Occurred()) goto fail;

    if (v_rule_ptr.len % sizeof(int64_t) != 0) {
        PyErr_SetString(PyExc_ValueError, "rule_ptr buffer length must be multiple of 8");
        goto fail;
    }
    n_rules = (v_rule_ptr.len / sizeof(int64_t)) - 1;

    if (v_cond_bitsets.len != (Py_ssize_t)(n_conds * n_bytes)) {
        PyErr_SetString(PyExc_ValueError, "cond_bitsets length must be n_conds * n_bytes");
        goto fail;
    }

    if (out_masks_obj != NULL && out_masks_obj != Py_None) {
        if (PyObject_GetBuffer(out_masks_obj, &v_out_masks, PyBUF_WRITABLE | PyBUF_C_CONTIGUOUS) != 0)
            goto fail;
        if (v_out_masks.len != (Py_ssize_t)(n_rules * n_bytes)) {
            PyErr_SetString(PyExc_ValueError, "out_masks length must be n_rules * n_bytes");
            PyBuffer_Release(&v_out_masks);
            goto fail;
        }
        have_out_masks = 1;
    }

    scores = (int64_t *)malloc(n_rules * sizeof(int64_t));
    if (!scores) {
        PyErr_NoMemory();
        goto fail_masks;
    }

    rule_eval_kernel(
        (const int64_t *)v_rule_ptr.buf,
        (const int64_t *)v_rule_cond_ids.buf,
        (const uint8_t *)v_cond_bitsets.buf,
        n_rules,
        n_conds,
        n_bytes,
        scores,
        have_out_masks ? (uint8_t *)v_out_masks.buf : NULL
    );

    result = PyList_New((Py_ssize_t)n_rules);
    if (!result) {
        free(scores);
        goto fail_masks;
    }
    for (r = 0; r < n_rules; r++) {
        PyObject *v = PyLong_FromLongLong((long long)scores[r]);
        if (!v) {
            Py_DECREF(result);
            free(scores);
            goto fail_masks;
        }
        PyList_SET_ITEM(result, (Py_ssize_t)r, v);
    }
    free(scores);
    if (have_out_masks)
        PyBuffer_Release(&v_out_masks);
    PyBuffer_Release(&v_rule_ptr);
    PyBuffer_Release(&v_rule_cond_ids);
    PyBuffer_Release(&v_cond_bitsets);
    return result;

fail_masks:
    if (have_out_masks)
        PyBuffer_Release(&v_out_masks);
fail:
    PyBuffer_Release(&v_rule_ptr);
    PyBuffer_Release(&v_rule_cond_ids);
    PyBuffer_Release(&v_cond_bitsets);
    return NULL;
}

static PyMethodDef rule_eval_methods[] = {
    {"eval_rules_c", eval_rules_c_impl, METH_VARARGS,
     "eval_rules_c(rule_ptr, rule_cond_ids, cond_bitsets, n_conds, n_bytes [, out_masks]) -> list of int\n"
     "Rule evaluation kernel: AND-reduce + popcount per rule. If out_masks (n_rules, n_bytes) is given, row-level masks are written there."},
    {NULL, NULL, 0, NULL}
};

static PyModuleDef rule_eval_module_def = {
    PyModuleDef_HEAD_INIT,
    //.m_name = "rule_eval_c",
    .m_name = "_rule_eval_c",
    .m_doc = "Pure C rule evaluation kernel (bitset AND + popcount).",
    .m_size = -1,
    .m_methods = rule_eval_methods,
};

PyMODINIT_FUNC
//PyInit_rule_eval_c(void)
PyInit__rule_eval_c(void)
{
    return PyModule_Create(&rule_eval_module_def);
}
