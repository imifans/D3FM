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

def EM_onestep_2(f_pre, I, V, HyperP, psi=0.5, eta=0.01): 
    device = f_pre.device

    fft_k1 = HyperP["fft_k1"].to(device)
    fft_k2 = HyperP["fft_k2"].to(device)
    fft_k1sq = HyperP["fft_k1sq"].to(device)
    fft_k2sq = HyperP["fft_k2sq"].to(device)
    E_1_over_nij  = HyperP["C"].to(device)
    E_1_over_mij  = HyperP["D"].to(device)
    F2 = HyperP["F2"].to(device)
    F1 = HyperP["F1"].to(device)
    K  = HyperP["H"].to(device)
    
    gamma = HyperP["tau_a"]#.to(device)  @γ gamma
    rho = HyperP["tau_b"]#.to(device)  @ρ rho
    PSI =  psi
    ETA =  eta 
       
    Y = f_pre - I
    X = f_pre - V
    
    #e-step
    E_1_over_mij = torch.sqrt(2/rho/(X**2+1e-6)) # E(1/m_ij) 
    E_1_over_nij = torch.sqrt(2/gamma/(Y**2+1e-6)) # E(1/n_ij)
    E_1_over_mij[ E_1_over_mij > 2 * E_1_over_nij ] = 2 * E_1_over_nij[ E_1_over_mij > 2 * E_1_over_nij ] # 防止 D 的值超过某个特定的阈值 2*C

    # 更新ρ和γ，预留一半是防止更新过快 1e-10是噪声扰动
    rho = torch.mean(1. / (E_1_over_mij + 1e-10)) + rho / 2
    gamma = torch.mean(1. / (E_1_over_nij + 1e-10)) + gamma / 2

    # m-step  更新 x_0
    for s in range(1):
        # Update k  公式25
        K = prox_tv(Y, F1, F2, fft_k1, fft_k2, fft_k1sq, fft_k2sq) # 公式25
        
        # Update u  公式27
        a1=torch.zeros_like(K)
        a1[:,:,:,:-1]=K[:,:,:,:-1]-K[:,:,:,1:]
        a1[:,:,:,-1]=K[:,:,:,-1]
        F1 = (ETA/(2*PSI+ETA))*a1
        
        # Update x  公式29
        nabla_k=torch.zeros_like(K) # ∇k
        nabla_k[:,:,:-1,:]=K[:,:,:-1,:]-K[:,:,1:,:]
        nabla_k[:,:,-1,:]=K[:,:,-1,:]
        F2 = ( ETA / ( 2 * PSI + ETA ) ) * nabla_k
        
        #x = (2m^2 ⊙ y + ηk) ⊘ (2m^2 + 2n^2 + η)
        X = (2 * E_1_over_nij * Y + ETA * (I-V-K)) / (2 * E_1_over_nij + 2 * E_1_over_mij + ETA)

    
    F = I - X # = I - (f_pre - V) = I + V - f_pre

    return F,{"C":E_1_over_nij, "D":E_1_over_mij, "F2":F2, "F1":F1, "H":K, "tau_a":gamma, "tau_b":rho,
    "fft_k1":fft_k1, "fft_k2":fft_k2, "fft_k1sq":fft_k1sq, "fft_k2sq":fft_k2sq}

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
    
