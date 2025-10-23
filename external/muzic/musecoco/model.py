# Re-export the model from the subdirectory
# from muzic.musecoco.2-attribute2music_model.model import *

# Use a different approach to import from directories with names starting with numbers
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "2-attribute2music_model"))
from model import *

# Explicitly import and re-export the MuseCocoModel class
try:
    from model import MuseCocoModel
except ImportError:
    # If not found in the direct import, try to find it in the module structure
    try:
        from attribute2music_model.model import MuseCocoModel
    except ImportError:
        raise ImportError("Could not find MuseCocoModel in the expected modules")

# Note: This approach allows importing from directories with names that aren't valid Python identifiers