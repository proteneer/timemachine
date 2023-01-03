def test_fma_compiler_bug(self):

    # this test case deals with a rather annoying fma compiler bug in CUDA.
    # see https://github.com/proteneer/timemachine/issues/386
    fp = gzip.open("tests/data/repro.pkl.gz", "rb")  # This assumes that primes.data is already packed with gzip
    x_t, box, lamb, nb_bp = pickle.load(fp)

    for precision in [np.float32, np.float64]:

        impl = nb_bp.unbound_impl(precision)
        du_dx, du_dp, du_dl, u = impl.execute(x_t, nb_bp.params, box, lamb)

        uimpl2 = nb_bp.unbound_impl(precision)

        uimpl2.disable_hilbert_sort()
        du_dx2, du_dp2, du_dl2, u2 = uimpl2.execute(x_t, nb_bp.params, box, lamb)

        np.testing.assert_array_equal(u2, u)
        np.testing.assert_array_equal(du_dx2, du_dx)
        np.testing.assert_array_equal(du_dp2, du_dp)
        np.testing.assert_array_equal(du_dl2, du_dl)  # this one fails without the patch.
