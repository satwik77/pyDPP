import unittest

class TestDPP(unittest.TestCase):

    def setUp(self):
        pass

    def test_DPP_import(self):
        try:
            from pydpp.dpp import DPP
        except:
            self.fail("Cannot Import DPP")

    def test_DPP_instance(self):
        try:
            from pydpp.dpp import DPP
            import numpy as np
            x =np.random.random((10,10))
            dpp =DPP(x)

        except:
            self.fail("Cannot create DPP instance")

    def test_kernel(self):
        try:
            from pydpp.dpp import DPP
            import numpy as np
            x =np.random.random((10,10))
            dpp =DPP(x)
            dpp.compute_kernel()

        except:
            self.fail("Cannot compute kernel for DPP")

    def test_sampling(self):

        from pydpp.dpp import DPP
        import numpy as np
        x =np.random.random((10,10))
        dpp =DPP(x)
        dpp.compute_kernel()
        try:
            s = dpp.sample()
        except:
            self.fail("Cannot sample from DPP")
        try:
            sk = dpp.sample_k(5)
        except:
            self.fail("Cannot sample from k-DPP")


if __name__ == '__main__':
    unittest.main()
