import os
import torch
import onnx
import tensorflow as tf
from onnx_tf.backend import prepare
from model import resnet50  # Import the model

# Define Constants
MODEL = resnet50
WEIGHTS = "slowfast_crime_detection.pth"
DUMMY_INPUT_SHAPE = (1, 3, 16, 128, 128)  # Shape of input tensor
TEMP_DIR = "temp_conversion"  # Temporary directory
CLASS_NUM = 2  # Number of output classes (e.g., crime vs. no crime)
ONNX_MODEL = os.path.join(TEMP_DIR, "model.onnx")
TF_MODEL_DIR = os.path.join(TEMP_DIR, "saved_model")
TFLITE_MODEL = os.path.join(TEMP_DIR, "model.tflite")

def pytorch_to_onnx():
    """Convert PyTorch model to ONNX."""
    print("\nStarting PyTorch to ONNX conversion...")
    os.makedirs(TEMP_DIR, exist_ok=True)
    
    print("\nInitializing model and loading weights...")
    model = MODEL(class_num=CLASS_NUM)
    state_dict = torch.load(WEIGHTS, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    
    print("\nExporting model to ONNX...")
    dummy_input = torch.randn(DUMMY_INPUT_SHAPE)
    torch.onnx.export(
        model,
        dummy_input,
        ONNX_MODEL,
        opset_version=11,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
    )
    print(f"ONNX model saved at: {ONNX_MODEL}")

def onnx_to_tensorflow():
    """Convert ONNX model to TensorFlow SavedModel format."""
    print("\nStarting ONNX to TensorFlow conversion...")
    os.makedirs(TF_MODEL_DIR, exist_ok=True)
    
    print("\nLoading ONNX model...")
    onnx_model = onnx.load(ONNX_MODEL)
    tf_rep = prepare(onnx_model)
    
    print("\nSaving TensorFlow model...")
    tf_rep.export_graph(TF_MODEL_DIR)
    print(f"TensorFlow SavedModel saved at: {TF_MODEL_DIR}")

def tensorflow_to_tflite():
    """Convert TensorFlow SavedModel to TensorFlow Lite."""
    print("\nStarting TensorFlow to TFLite conversion...")
    
    print("\nLoading TensorFlow model...")
    converter = tf.lite.TFLiteConverter.from_saved_model(TF_MODEL_DIR)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # Enable TF Select Ops fallback (bcos current model uses MaxPool3D, which is not supported in TFLite)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,   # Default TFLite ops.
        tf.lite.OpsSet.SELECT_TF_OPS       # Enable TF ops not natively supported.
    ]

    tflite_model = converter.convert()
    
    print("\nSaving TFLite model...")
    with open(TFLITE_MODEL, "wb") as f:
        f.write(tflite_model)
    print(f"TFLite model saved at: {TFLITE_MODEL}")

def main():
    pytorch_to_onnx()
    onnx_to_tensorflow()
    tensorflow_to_tflite()

if __name__ == "__main__":
    main()
