/* Packaging-only guard: compile at baseline x86-64, before importing SIMD DSOs. */
#include <Python.h>
#include <cpuid.h>

static PyObject *require_sse41(PyObject *self, PyObject *args) {
    unsigned int a, b, c, d;
    (void) self; (void) args;
    if (!__get_cpuid(1, &a, &b, &c, &d) || !(c & bit_SSE4_1)) {
        PyErr_SetString(PyExc_ImportError,
            "This Astra runtime requires an x86-64 CPU with SSE4.1. "
            "AVX-512 is optional; SSE4.1 is the baseline for this release.");
        return NULL;
    }
    Py_RETURN_NONE;
}
static PyMethodDef methods[] = {
    {"require_sse41", require_sse41, METH_NOARGS, "Reject unsupported CPUs before SIMD imports."},
    {NULL, NULL, 0, NULL}
};
static struct PyModuleDef module = {PyModuleDef_HEAD_INIT, "_cpu_check", NULL, -1, methods};
PyMODINIT_FUNC PyInit__cpu_check(void) { return PyModule_Create(&module); }
