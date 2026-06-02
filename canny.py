import os
import cv2
import torch
from torch.autograd import Variable
from net_canny import Net

def canny(raw_img, output_path, use_cuda=False):
    """
    Generates a thresholded Canny edge image from `raw_img` and saves to `output_path`.
    """
    # Convert image to float32 and HWC -> CHW
    img = torch.from_numpy(raw_img.transpose((2, 0, 1)))
    batch = torch.stack([img]).float() / 255.0  # normalize here

    # Initialize the model
    net = Net(threshold=3.0, use_cuda=use_cuda)
    net.cuda()
    net.eval()

    data = Variable(batch)
    data = data.cuda()

    _, _, _, _, thresholded, _ = net(data)

    # Get edge image from tensor, scale to 0-255 and save using cv2
    edge_img = (thresholded.data.cpu().numpy()[0, 0] > 0.0).astype('uint8') * 255
    cv2.imwrite(output_path, edge_img)

if __name__ == '__main__':
    input_folder = '/nfs/wattrel/data/md0/kung/Cognitive-Inspired-View-Selection/modelnet_32_60_1_19'
    output_folder = '/nfs/wattrel/data/md0/kung/Cognitive-Inspired-View-Selection/modelnet_edge_1_19'

    os.makedirs(output_folder, exist_ok=True)

    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith((".png")):

                input_path = os.path.join(root, file)
                img = cv2.imread(input_path)
                if img is None:
                    print(f"Warning: Failed to read {input_path}")
                    continue

                # Convert BGR to RGB before passing to model
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                rel_path = os.path.relpath(input_path, input_folder)
                save_path = os.path.join(output_folder, rel_path)
                os.makedirs(os.path.dirname(save_path), exist_ok=True)

                # Run Canny and save edge image
                if not os.path.exists(save_path):
                    print('file', file)
                    canny(img_rgb, save_path, use_cuda=True)
