import onnx

onnx_model_path = "temp_conversion/model.onnx"  # Path to your ONNX model
model = onnx.load(onnx_model_path)

# Check the input and output tensor shapes
for input_tensor in model.graph.input:
    print(f"Input Name: {input_tensor.name}")
    shape = [dim.dim_value if dim.dim_value > 0 else "Dynamic" for dim in input_tensor.type.tensor_type.shape.dim]
    print(f"Input Shape: {shape}")

for output_tensor in model.graph.output:
    print(f"Output Name: {output_tensor.name}")
    shape = [dim.dim_value if dim.dim_value > 0 else "Dynamic" for dim in output_tensor.type.tensor_type.shape.dim]
    print(f"Output Shape: {shape}")
