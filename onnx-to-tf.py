from onnx_tf.backend import prepare

onnx_model = onnx.load("quantized_model.onnx")
tf_rep = prepare(onnx_model)
tf_rep.export_graph("model.pb")