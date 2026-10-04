"""Import the public lightweight model without any available Torch package."""
import os
import subprocess
import sys


def test_tfidf_import_does_not_load_optional_torch():
    code = '''
import importlib.abc, sys
class MissingTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise ModuleNotFoundError("optional Torch deliberately unavailable", name="torch")
sys.meta_path.insert(0, MissingTorch())
from loris.models.tfidf_classifier import TFIDFClassifier
from loris.models import BaseDocumentClassifier
assert issubclass(TFIDFClassifier, BaseDocumentClassifier)
assert "torch" not in sys.modules
try:
    from loris.models import NeuralClassifier
except ModuleNotFoundError as error:
    assert error.name == "torch"
else:
    raise AssertionError("explicit neural import must still require Torch")
'''
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                            env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
    assert result.returncode == 0, result.stderr


def test_explicit_lazy_neural_export_preserves_public_class():
    from loris.models import NeuralClassifier
    from loris.models.neural_classifier import NeuralClassifier as concrete
    assert NeuralClassifier is concrete
