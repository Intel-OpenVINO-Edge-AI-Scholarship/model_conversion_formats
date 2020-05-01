import onnx
import caffe2.python.onnx.frontend
from caffe2.proto import caffe2_pb2

import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predict_net', required=True)
    parser.add_argument('--init_net', required=True)
    parser.add_argument('--predict_net_name', required=True)
    parser.add_argument('--init_net_name', required=True)
    parser.add_argument('--onnx_file', required=True)

    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()

    # We need to provide type and shape of the model inputs, 
    # see above Note section for explanation
    data_type = onnx.TensorProto.FLOAT
    data_shape = (1, 3, 224, 224)
    value_info = {
        'data': (data_type, data_shape)
    }

    predict_net = caffe2_pb2.NetDef()
    with open(args.predict_net, 'rb') as f:
        predict_net.ParseFromString(f.read())
        predict_net.name = args.predict_net_name

    init_net = caffe2_pb2.NetDef()
    with open(args.init_net, 'rb') as f:
        init_net.ParseFromString(f.read())
        init_net.name = args.init_net_name

    onnx_model = caffe2.python.onnx.frontend.caffe2_net_to_onnx_model(
        predict_net,
        init_net,
        value_info,
    )

    onnx.checker.check_model(onnx_model)

    onnx.save(onnx_model, args.onnx_file)