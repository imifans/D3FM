import numpy as np
import os
import torch
import torch.fft

# import cv2

def EM_Initial(IR):
    device = IR.device

    k1 = torch.tensor([[1,-1]]).to(device)
    k2 = torch.tensor([[1],[-1]]).to(device)

    fft_k1 = psf2otf(k1, np.shape(IR)).to(device)
    fft_k2 = psf2otf(k2, np.shape(IR)).to(device)
    fft_k1sq = torch.conj(fft_k1)*fft_k1.to(device)
    fft_k2sq = torch.conj(fft_k2)*fft_k2.to(device)

    C  = torch.ones_like(IR).to(device)
    D = torch.ones_like(IR).to(device)
    F2 = torch.zeros_like(IR).to(device)
    F1 = torch.zeros_like(IR).to(device)
    H  = torch.zeros_like(IR).to(device)
    tau_a = 1
    tau_b = 1
    HP={"C":C, "D":D, "F2":F2, "F1":F1, "H":H, "tau_a":tau_a, "tau_b":tau_b,
    "fft_k1":fft_k1, "fft_k2":fft_k2, "fft_k1sq":fft_k1sq, "fft_k2sq":fft_k2sq}
    return HP

def EM_onestep(f_pre, I, V, HyperP, lamb=0.5, rho=0.01): 
    device = f_pre.device

    fft_k1 = HyperP["fft_k1"].to(device)
    fft_k2 = HyperP["fft_k2"].to(device)
    fft_k1sq = HyperP["fft_k1sq"].to(device)
    fft_k2sq = HyperP["fft_k2sq"].to(device)
    C  = HyperP["C"].to(device)
    D  = HyperP["D"].to(device)
    F2 = HyperP["F2"].to(device)
    F1 = HyperP["F1"].to(device)
    H  = HyperP["H"].to(device)
    tau_a = HyperP["tau_a"]#.to(device)
    tau_b = HyperP["tau_b"]#.to(device)

    LAMBDA =  lamb

    Y = I - V # 自然图像与红外图像的差异
    X = f_pre - V # 融合图像与自然图像的差异
    #e-step
    D = torch.sqrt(2/tau_b/(X**2+1e-6))
    C = torch.sqrt(2/tau_a/((Y-X+1e-6)**2))
    D[D>2*C] = 2*C[D>2*C]
    RHO =rho # .5*(C+D)
    
    tau_b = 1./(D+1e-10)+tau_b/2
    tau_b = torch.mean(tau_b)
    tau_a = 1./(C+1e-10)+tau_a/2
    tau_a = torch.mean(tau_a)

    # m-step

    H = prox_tv(Y-X,F1,F2, fft_k1, fft_k2, fft_k1sq, fft_k2sq)

    a1=torch.zeros_like(H)
    a1[:,:,:,:-1]=H[:,:,:,:-1]-H[:,:,:,1:]
    a1[:,:,:,-1]=H[:,:,:,-1]
    F1 = (RHO/(2*LAMBDA+RHO))*a1

    a2=torch.zeros_like(H)
    a2[:,:,:-1,:]=H[:,:,:-1,:]-H[:,:,1:,:]
    a2[:,:,-1,:]=H[:,:,-1,:]
    F2 = (RHO/(2*LAMBDA+RHO))*a2
    X = (2*C*Y+RHO*(Y-H))/(2*C+2*D+RHO)

    #
    F = I - X # X使得 min x ∥y − x∥1 + φ∥x∥1.成立

    return F,{"C":C, "D":D, "F2":F2, "F1":F1, "H":H, "tau_a":tau_a, "tau_b":tau_b,
    "fft_k1":fft_k1, "fft_k2":fft_k2, "fft_k1sq":fft_k1sq, "fft_k2sq":fft_k2sq}

def EM_onestep_2(f_pre, I, V, HyperP, lamb=0.5, rho=0.01): 
    device = f_pre.device

    fft_k1 = HyperP["fft_k1"].to(device)
    fft_k2 = HyperP["fft_k2"].to(device)
    fft_k1sq = HyperP["fft_k1sq"].to(device)
    fft_k2sq = HyperP["fft_k2sq"].to(device)
    C  = HyperP["C"].to(device)
    D  = HyperP["D"].to(device)
    F2 = HyperP["F2"].to(device)
    F1 = HyperP["F1"].to(device)
    H  = HyperP["H"].to(device)
    tau_a = HyperP["tau_a"]#.to(device)
    tau_b = HyperP["tau_b"]#.to(device)

    LAMBDA =  lamb

    Y = I - V # 自然图像与红外图像的差异
    X = f_pre - V # 融合图像与自然图像的差异
    
    #e-step
    D = torch.sqrt(2/tau_b/(X**2+1e-6)) # E(1/m)
    C = torch.sqrt(2/tau_a/((Y-X+1e-6)**2)) # E(1/n)
    
    D[D>2*C] = 2*C[D>2*C]
    RHO =rho # .5*(C+D)
    
    tau_b = 1./(D+1e-10)+tau_b/2 # 用求出来的值，叠加在原来的值的一般上，减小更小幅度。
    tau_b = torch.mean(tau_b)
    tau_a = 1./(C+1e-10)+tau_a/2
    tau_a = torch.mean(tau_a)

    # m-step

    H = prox_tv(Y-X,F1,F2, fft_k1, fft_k2, fft_k1sq, fft_k2sq)

    a1=torch.zeros_like(H)
    a1[:,:,:,:-1]=H[:,:,:,:-1]-H[:,:,:,1:]
    a1[:,:,:,-1]=H[:,:,:,-1]
    F1 = (RHO/(2*LAMBDA+RHO))*a1

    a2=torch.zeros_like(H)
    a2[:,:,:-1,:]=H[:,:,:-1,:]-H[:,:,1:,:]
    a2[:,:,-1,:]=H[:,:,-1,:]
    F2 = (RHO/(2*LAMBDA+RHO))*a2
    X = (2*C*Y+RHO*(Y-H))/(2*C+2*D+RHO)

    #
    F = X+V  # X使得 min x ∥y − x∥1 + φ∥x∥1.成立

    return F,{"C":C, "D":D, "F2":F2, "F1":F1, "H":H, "tau_a":tau_a, "tau_b":tau_b,
    "fft_k1":fft_k1, "fft_k2":fft_k2, "fft_k1sq":fft_k1sq, "fft_k2sq":fft_k2sq}
    
def EM_f(f, I, V, epsilon=1e-8):
    """
    使用 EM 算法更新图像 f 使其与图像 I 和 V 更加相似。

    参数:
    - f: 初始图像 (batch_size, channels, height, width)
    - I: 红外图像 (batch_size, channels, height, width)
    - V: 可见光图像 (batch_size, channels, height, width)
    - epsilon: 小常数，防止除以零 (默认值: 1e-8)

    返回:
    - 更新后的图像 f
    """
    def cal_latent(f, x, epsilon):
        """
        计算潜在变量。

        参数:
        - f: 当前图像 (batch_size, channels, height, width)
        - x: 给定图像 (batch_size, channels, height, width)
        - epsilon: 小常数，防止除以零 (默认值: 1e-8)

        返回:
        - 潜在变量 (batch_size, channels, height, width)
        """
        diff = f - x
        squared_diff = diff**2
        return 1 / (squared_diff + epsilon)
    
    def cal_variance(image):
        # 计算图像的均值
        mean = np.mean(image)
        # 计算方差
        variance = np.mean((image - mean) ** 2)
        return variance
       
    def cal_variance_rgb(rgb):
        # 移除 batch 维度
        rgb_img = rgb.squeeze(0)
        
        # 确保图像张量是浮点型，以便计算方差
        rgb_img = rgb_img.float()
        
        # 计算每个通道的方差
        var_per_channel = rgb_img.var(dim=(1, 2), unbiased=False)  # (3,) 张量
        
        # 计算所有通道方差的均值
        mean_var = var_per_channel.mean()
        return var_per_channel

    # 计算潜在变量 m 和 n
    #m = cal_latent(f, V, epsilon)
    #n = cal_latent(f, I, epsilon)
    m = cal_variance_rgb(V) / 2
    n = cal_variance_rgb(I) / 2

    # 使用 m 和 n 更新图像 f  # [1,3,288,489] *[1,3]
    m = m.view(1, 3, 1, 1)  # 将其扩展为 [1, 3, 1, 1]
    n = n.view(1, 3, 1, 1)  # 将其扩展为 [1, 3, 1, 1]

    f = (m * I + n * V) / (m + n + epsilon)

    # 计算更新前图像的均值和标准差
    mean_f = torch.mean(f)
    std_f = torch.std(f)

    # 归一化图像
    f = (f - mean_f) / (std_f + epsilon)

    return f



def prox_tv(X, F1, F2, fft_k1, fft_k2, fft_k1sq, fft_k2sq):
    fft_X = torch.fft.fft2(X)
    fft_F1 = torch.fft.fft2(F1)
    fft_F2 = torch.fft.fft2(F2)

    k = fft_X + torch.conj(fft_k1) * fft_F1 + torch.conj(fft_k2) * fft_F2
    k = k/(1 + fft_k1sq + fft_k2sq) #公式25  更新的是K
    k = torch.real(torch.fft.ifft2(k))
    return k

def psf2otf(psf, outSize):
    psfSize = torch.tensor(psf.shape)
    outSize = torch.tensor(outSize[-2:])
    padSize = outSize - psfSize
    psf=torch.nn.functional.pad(psf,(0, padSize[1],0, padSize[0]), 'constant')

    for i in range(len(psfSize)):
        psf = torch.roll(psf, -int(psfSize[i] / 2), i)
    otf = torch.fft.fftn(psf)
    nElem = torch.prod(psfSize)
    nOps = 0
    for k in range(len(psfSize)):
        nffts = nElem / psfSize[k]
        nOps = nOps + psfSize[k] * torch.log2(psfSize[k]) * nffts
    if torch.max(torch.abs(torch.imag(otf))) / torch.max(torch.abs(otf)) <= nOps * torch.finfo(torch.float32).eps:
        otf = torch.real(otf)
    return otf
    
