import unittest

class TestBasics(unittest.TestCase):
  def test_import(self):
    modelnames = ['Speech','DeepQ','Seq2Seq','Autoenc','MemNet','Residual','VGG','AlexNet']
    import fathom
    for modelname in modelnames:
      assert hasattr(fathom,modelname), 'No model named "'+str(modelname)+'" found in fathom module.'
    for modelname in modelnames:
      modelname += 'Fwd'
      assert hasattr(fathom,modelname), 'No model named "'+str(modelname)+'" found in fathom module.'

  # FIXME: ALE load failure causes testing to abort.
  @unittest.SkipTest
  def test_create_deepq(self):
    from fathom import DeepQ, DeepQFwd
    model = DeepQ()
    model = DeepQFwd()

  def test_create_speech(self):
    from fathom import Speech, SpeechFwd
    model = Speech()
    model = SpeechFwd()

  def test_create_seq2seq(self):
    from fathom import Seq2Seq, Seq2SeqFwd
    model = Seq2Seq()
    model = Seq2SeqFwd()

  def test_create_autoenc(self):
    from fathom import Autoenc, AutoencFwd
    model = Autoenc()
    model = AutoencFwd()

  def test_create_memnet(self):
    from fathom import MemNet, MemNetFwd
    model = MemNet()
    model = MemNetFwd()

  def test_create_residual(self):
    from fathom import Residual, ResidualFwd
    model = Residual()
    model = ResidualFwd()

  def test_create_vgg(self):
    from fathom import VGG, VGGFwd
    model = VGG()
    model = VGGFwd()

  def test_create_alexnet(self):
    from fathom import AlexNet, AlexNetFwd
    model = AlexNet()
    model = AlexNetFwd()

