import onnx
import caffe2.python.onnx.frontend
from caffe2.proto import caffe2_pb2

if __name__ == "__main__":

    # We need to provide type and shape of the model inputs, 
    # see above Note section for explanation
    data_type = onnx.TensorProto.FLOAT
    data_shape = (1, 3, 227, 227)
    value_info = {
        'data': (data_type, data_shape)
    }

    predict_net = caffe2_pb2.NetDef()
    with open('predict_net.pb', 'rb') as f:
        predict_net.ParseFromString(f.read())
        predict_net.name = "squeezenet"

    init_net = caffe2_pb2.NetDef()
    with open('init_net.pb', 'rb') as f:
        init_net.ParseFromString(f.read())
        init_net.name = "squeezenet"

    onnx_model = caffe2.python.onnx.frontend.caffe2_net_to_onnx_model(
        predict_net,
        init_net,
        value_info,
    )

    onnx.checker.check_model(onnx_model)

    onnx.save(onnx_model, "squeezenet.onnx")