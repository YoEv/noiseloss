# testing MusicGen-hf model structure
from transformers import AutoProcessor, MusicgenForConditionalGeneration

print("Loading MusicGen model...")
model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")

print("Model structure:")
print(f"Model type: {type(model)}")
print(f"Has decoder: {hasattr(model, 'decoder')}")

if hasattr(model, 'decoder'):
    print(f"Decoder type: {type(model.decoder)}")
    print(f"Decoder has layers: {hasattr(model.decoder, 'layers')}")
    
    if hasattr(model.decoder, 'layers'):
        print(f"Number of decoder layers: {len(model.decoder.layers)}")
        print(f"First layer type: {type(model.decoder.layers[0])}")
    else:
        print("Decoder attributes:")
        for attr in dir(model.decoder):
            if not attr.startswith('_'):
                print(f"  - {attr}")

print(f"Has lm: {hasattr(model, 'lm')}")
print(f"Has text_encoder: {hasattr(model, 'text_encoder')}")
print(f"Has audio_encoder: {hasattr(model, 'audio_encoder')}")

print("\nTesting layer access...")
try:
    layers = model.decoder.layers
    print(f"✓ model.decoder.layers works! Found {len(layers)} layers")
except Exception as e:
    print(f"✗ model.decoder.layers failed: {e}")

try:
    layers = model.lm.transformer.layers
    print(f"✓ model.lm.transformer.layers works! Found {len(layers)} layers")
except Exception as e:
    print(f"✗ model.lm.transformer.layers failed: {e}")

# Check decoder.model structure
print(f"\nDecoder has 'model' attribute: {hasattr(model.decoder, 'model')}")
if hasattr(model.decoder, 'model'):
    print(f"Decoder.model type: {type(model.decoder.model)}")
    print(f"Decoder.model has layers: {hasattr(model.decoder.model, 'layers')}")
    
    if hasattr(model.decoder.model, 'layers'):
        try:
            layers = model.decoder.model.layers
            print(f"✓ model.decoder.model.layers works! Found {len(layers)} layers")
            print(f"First layer type: {type(layers[0])}")
        except Exception as e:
            print(f"✗ model.decoder.model.layers failed: {e}")
    else:
        print("Decoder.model attributes:")
        for attr in dir(model.decoder.model):
            if not attr.startswith('_') and not callable(getattr(model.decoder.model, attr)):
                print(f"  - {attr}")

# Also check if decoder has transformer
print(f"\nDecoder has 'transformer' attribute: {hasattr(model.decoder, 'transformer')}")
if hasattr(model.decoder, 'transformer'):
    print(f"Decoder.transformer has layers: {hasattr(model.decoder.transformer, 'layers')}")
    if hasattr(model.decoder.transformer, 'layers'):
        try:
            layers = model.decoder.transformer.layers
            print(f"✓ model.decoder.transformer.layers works! Found {len(layers)} layers")
        except Exception as e:
            print(f"✗ model.decoder.transformer.layers failed: {e}")

# Explore MusicgenModel structure more deeply
print(f"\n=== Exploring MusicgenModel structure ===")
musicgen_model = model.decoder.model
print(f"MusicgenModel attributes:")
for attr in dir(musicgen_model):
    if not attr.startswith('_') and hasattr(musicgen_model, attr):
        attr_obj = getattr(musicgen_model, attr)
        if hasattr(attr_obj, '__class__') and 'torch.nn' in str(type(attr_obj)):
            print(f"  - {attr}: {type(attr_obj)}")

# Check if MusicgenModel has decoder
print(f"\nMusicgenModel has 'decoder' attribute: {hasattr(musicgen_model, 'decoder')}")
if hasattr(musicgen_model, 'decoder'):
    decoder_obj = getattr(musicgen_model, 'decoder')
    print(f"MusicgenModel.decoder type: {type(decoder_obj)}")
    print(f"MusicgenModel.decoder has layers: {hasattr(decoder_obj, 'layers')}")
    
    if hasattr(decoder_obj, 'layers'):
        try:
            layers = decoder_obj.layers
            print(f"✓ model.decoder.model.decoder.layers works! Found {len(layers)} layers")
            print(f"First layer type: {type(layers[0])}")
        except Exception as e:
            print(f"✗ model.decoder.model.decoder.layers failed: {e}")
    else:
        print("MusicgenModel.decoder attributes:")
        for attr in dir(decoder_obj):
            if not attr.startswith('_') and hasattr(decoder_obj, attr):
                attr_obj = getattr(decoder_obj, attr)
                if hasattr(attr_obj, '__class__') and 'torch.nn' in str(type(attr_obj)):
                    print(f"    - {attr}: {type(attr_obj)}")

# Check for other common transformer attributes
for attr_name in ['transformer', 'model', 'layers', 'blocks']:
    if hasattr(musicgen_model, attr_name):
        attr_obj = getattr(musicgen_model, attr_name)
        print(f"\nMusicgenModel.{attr_name}: {type(attr_obj)}")
        if hasattr(attr_obj, 'layers'):
            try:
                layers = attr_obj.layers
                print(f"✓ model.decoder.model.{attr_name}.layers works! Found {len(layers)} layers")
            except Exception as e:
                print(f"✗ model.decoder.model.{attr_name}.layers failed: {e}")