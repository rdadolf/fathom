import unittest

class TestBasics(unittest.TestCase):
  modelnames = ['DeepQ','Speech','Seq2Seq']

  def test_import(self):
    import fathom
    for modelname in self.modelnames:
      assert hasattr(fathom,modelname), 'No model named "'+str(modelname)+'" found in fathom module.'

  def test_create_models(self):
    import fathom
    for modelname in self.modelnames:
      ModelClass = getattr(fathom, modelname)
      model = ModelClass()
      assert isinstance(model,ModelClass), 'Couldnt create model "'+str(modelname)+'"'
