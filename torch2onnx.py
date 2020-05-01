import torch
import argparse

def load_model(model, state_dict_key='model_state_dict', checkpoint_path='img_color.checkpoint.pth'):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint[state_dict_key])
    return model

def load(filepath):
    return torch.load(filepath)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pth_file", default=None, required=True)
    parser.add_argument("--onnx_file", default=None, required=True)
    parser.add_argument("--npy_file", default=None, required=True)
    parser.add_argument("--precision", default=None, required=True)
    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()

    if(args.precision == "float"):
        tensor = torch.from_numpy(np.load(args.npy_file)).float()
    elif(args.precision == "double"):
        tensor = torch.from_numpy(np.load(args.npy_file)).double()

    torch.onnx.export(load(args.pth_file), tensor, args.onnx_file, 
        verbose=True, input_names=['input'], output_names=['data'])