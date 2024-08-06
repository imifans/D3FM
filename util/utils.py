import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.measure import shannon_entropy
import os
import yaml
import cv2
import numpy as np
import matplotlib.pyplot as plt

def norm_sample(img):
    temp_img= img.detach().cpu().squeeze().numpy()
    temp_img=np.transpose(temp_img, (1,2,0))
    temp_img=cv2.cvtColor(temp_img,cv2.COLOR_RGB2YCrCb)[:,:,0]
    temp_img=(temp_img-np.min(temp_img))/(np.max(temp_img)-np.min(temp_img))
    temp_img=((temp_img)*255).astype('uint8')
    return temp_img

def calculate_ssim(imageA, imageB):
    assert imageA.shape == imageA.shape
    grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
    score, _ = ssim(grayA, grayB, full=True)
    return score

def calculate_qabf(imageA, imageB):
    return 0.0

def calculate_vif(imageA, imageB):
    return 0.0
    
# MI (Mutual Information)
def calculate_mi(imageA, imageB):
    hist_2d, _, _ = np.histogram2d(imageA.ravel(), imageB.ravel(), bins=20)
    pxy = hist_2d / float(np.sum(hist_2d))
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)
    px_py = px[:, None] * py[None, :]
    nzs = pxy > 0
    return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))

def calculate_sd(image):
    return np.std(image)

def calculate_en(image):
    return shannon_entropy(image)
def image_read(path, mode='RGB'):
    img_BGR = cv2.imread(path).astype('float32')
    assert mode == 'RGB' or mode == 'GRAY' or mode == 'YCrCb', 'mode error'
    if mode == 'RGB':
        img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB) # OpenCV 中用于将图像的颜色空间从 BGR（蓝-绿-红）转换为 RGB（红-绿-蓝）的函数
    elif mode == 'GRAY':  
        img = np.round(cv2.cvtColor(img_BGR, cv2.COLOR_BGR2GRAY))
    elif mode == 'YCrCb':
        img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2YCrCb)
    return img
def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def norm_img(sample):
    sample=(sample-np.min(sample))/(np.max(sample)-np.min(sample))
    sample=((sample)*255).astype(np.uint8)
    return sample

def tensor_to_numpy(tensor):
    # Ensure tensor is on CPU before converting to numpy array
    return tensor.detach().cpu().numpy()

def plot(a,b,c):
    # 创建子图
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # 显示原始RGB图像
    axs[0].imshow(a)
    axs[1].imshow(b)
    axs[2].imshow(c)
    # 显示图像
    plt.show()
    
def adjust_image_shape(img, scale):
    h, w = img.shape[:2]
    h = h - h % scale
    w = w - w % scale
    return img[:h, :w, :]

def gray_to_rgb(gray_img):
    # 将灰度图像的单通道数据复制到三个通道中
    return np.stack([gray_img] * 3, axis=-1)

def evaluate(f,vi,ir):
    ssim_value1 = calculate_ssim(f, vi)
    ssim_value2 = calculate_ssim(f, ir)
    qabf_value1 = calculate_qabf(f, vi)
    qabf_value2 = calculate_qabf(f, ir)
    vif_value1 = calculate_vif(f, vi)
    vif_value2 = calculate_vif(f, ir)
    mi_value1 = calculate_mi(f, vi)
    mi_value2 = calculate_mi(f, ir)
    sd_value = calculate_sd(f)
    en_value = calculate_en(f)

    print(f"SSIM_1: {ssim_value1:.3f}, SSIM_2: {ssim_value2:.3f}, total: {(ssim_value1+ssim_value2):.3f}")
    # print(f"Qabf1: {qabf_value1}, Qabf2: {qabf_value2}")
    # print(f"VIF1: {vif_value1}, VIF2: {vif_value2}")
    print(f"MI1: {mi_value1:.3f}, MI2: {mi_value2:.3f}, total: {(mi_value2+mi_value1):.3f}")
    print(f"SD: {sd_value:.3f}")
    print(f"EN: {en_value:.3f}")
    
def evaluate_imgs(f_img_folder, vi_img_folder, ir_img_folder, mode):
    # 获取所有图片文件名（假设所有文件夹中的图片名称是一样的）
    vi_images = [f for f in os.listdir(vi_img_folder) if f.endswith('.jpg')]
    for img_name in sorted(vi_images):
        evaluate_imgs_by_name(f_img_folder, vi_img_folder, ir_img_folder, mode, img_name)            
    
def evaluate_imgs_by_name(f_img_folder, vi_img_folder, ir_img_folder, mode, img_name):
    # 构建完整路径
    vi_path = os.path.join(vi_img_folder, img_name)
    ir_path = os.path.join(ir_img_folder, img_name)
    f_name = img_name.replace('.jpg', '.png')  # 替换扩展名
    f_path = os.path.join(f_img_folder, f_name)

    # 读取图像
    vi = image_read(vi_path, mode=mode)[np.newaxis, np.newaxis, ...] / 255.0
    ir = image_read(ir_path, mode=mode)[np.newaxis, np.newaxis, ...] / 255.0
    f = image_read(f_path, mode=mode)[np.newaxis, np.newaxis, ...] / 255.0

    # 压缩维度
    vi = np.squeeze(vi)
    ir = np.squeeze(ir)
    f = np.squeeze(f)
    
    if(mode == "GRAY"):
        vi = gray_to_rgb(vi)
        ir = gray_to_rgb(ir)
        f = gray_to_rgb(f)
    
    scale = 32
    ir = adjust_image_shape(ir,scale)
    vi = adjust_image_shape(vi,scale)
    f = adjust_image_shape(f,scale)
    assert ir.shape == vi.shape
    inf_img = ir.squeeze()
    vis_img = vi.squeeze()
    f_img = f.squeeze()
    print(f"--------------------{img_name}--------------------")
    evaluate(f_img, vis_img, inf_img)
    plot(vis_img,inf_img,f_img)
