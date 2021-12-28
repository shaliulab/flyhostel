import unittest

from flyhostel.quantification.sleep import read_data, load_params

IMGSTORE_FOLDER = "./tests/static_data/test_imgstore"
ANALYSIS_FOLDER = "./tests/static_data/test_idtrackerai"



class TestSmoke(unittest.TestCase):

    def test_read_data(self):
        tr, chunks, store_metadata, chunk_metadata = read_data(IMGSTORE_FOLDER, ANALYSIS_FOLDER)
        shape = tr.s.shape
        target = list(range(13))
        
        self.assertTrue(shape[0] == 58498)
        self.assertTrue(shape[1] == 1)
        self.assertTrue(shape[2] == 2)
        self.assertTrue(all([chunks[i] == target[i] for i in range(len(chunks))]))
        self.assertTrue(store_metadata["exposure-time"] == 20001.0)
        self.assertTrue(store_metadata["chunksize"] == 4500.0)

        analysis_params, plotting_params = load_params(store_metadata)
        plotting_params.ld_annotation = True

        self.assertTrue(analysis_params.offset==54537.0)

        print(analysis_params)
        print(plotting_params)
        




if __name__ == "__main__":
    unittest.main()

    # analysis_params, plotting_params = load_params(args, store_metadata, tr)