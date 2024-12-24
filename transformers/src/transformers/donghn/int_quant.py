import numpy as np
import torch
from torch.nn import functional as F
from torch.autograd import Function
from torch import nn
from torch.autograd import Function, Variable

from .observers import ObserverBase, NormalMinMaxObserver, MovingAvgMinMaxObserver, NineNineObserver
from .int_cfg import opt, QInfo

import pickle

global s_count
global c_count
global b_int_count
global o_count
global f_count
s_count = 0
c_count = 0
b_int_count = 0
o_count = 0
f_count = 0

# Debug function for logging
def debug_log(message, **kwargs):
    print(f"[DEBUG] {message}")
    for key, value in kwargs.items():
        print(f"  - {key}: {value}")

class O2AQuant(Function):
    @staticmethod
    def forward(ctx, x, dt, n, gr):
        raw_x = torch.reshape(x, (-1,))
        org_len = len(raw_x)
        if org_len % gr:
            vacant_num = qinfo.o2ag - org_len % gr
            raw_x = F.pad(raw_x, (0, vacant_num), 'constant', 0)

        raw_x = torch.reshape(raw_x, (-1, gr))
        max_dim1, _ = raw_x.max(dim=1)

        if dt == 'weight':
            xth = 2**(n-1)
        else:
            xth = 2**n

        outlier_id = (max_dim1 >= xth)
        round_value = 2**(8-n)
        raw_x[outlier_id] = torch.round(raw_x[outlier_id] / round_value) * round_value

        raw_x = torch.reshape(raw_x, (-1,))
        x = torch.reshape(raw_x[:org_len], x.shape)

        return x
        

class MulO2AQuant(Function):
    @staticmethod
    def forward(ctx, x, dt, n, gr):
        raw_x = torch.reshape(x, (-1,))
        org_len = len(raw_x)
        if org_len % gr:
            vacant_num = qinfo.o2ag - org_len % gr
            raw_x = F.pad(raw_x, (0, vacant_num), 'constant', 0)

        raw_x = torch.reshape(raw_x, (-1, gr))
        max_dim1, _ = raw_x.max(dim=1)

        for b in range(n, 8):
            if dt == 'weight':
                mul_xth = 2**(b-1)
            else:
                mul_xth = 2**b
            round_value = 2**(b+1-n)
            outlier_id = torch.bitwise_and(max_dim1 >= mul_xth, max_dim1 < mul_xth*2)
            raw_x[outlier_id] = torch.round(raw_x[outlier_id] / round_value) * round_value

        raw_x = torch.reshape(raw_x, (-1,))
        x = torch.reshape(raw_x[:org_len], x.shape)

        return x

# original FP Quant code

def FQuantF(x, N, E, nosub=False):
    M_nm = N-1-E
    bias = (2**(E-1)-1)
    max_e = (2**E-2) - bias
    min_e_nm = 1 - bias
    min_e_un = min_e_nm - M_nm
    
    max_val = 2 ** max_e
    if nosub:
        min_val = 2 ** min_e_nm
    else:
        min_val = 2 ** min_e_un

    sign = torch.sign(x)
    x = torch.abs(x)
    x[x < min_val] = 0.0
    x = torch.clamp(x, max=max_val)

    non_zero_id = (x > 0)
    exp = torch.floor(torch.log2(x[non_zero_id]))
    fract = x[non_zero_id] / (2 ** exp) - 1

    M = torch.zeros_like(fract)
    if nosub:
        M = M_nm
    else:
        nm_id = exp>=min_e_nm
        um_id = exp<min_e_nm
        M[nm_id] = M_nm
        M[um_id] = (exp[um_id] - min_e_un)
        

    fract = torch.round(fract * (2 ** M)) / (2**M)
    x[non_zero_id] = sign[non_zero_id] * (2 ** exp) * (1 + fract)

    return x


# outlier FP Quant code
'''
def FQuantF(x, N, E, nosub=False):
    M_nm = N - 1 - E
    bias = (2**(E - 1) - 1)
    max_e = (2**E - 1) - bias
    min_e_nm = 1 - bias
    min_e_un = min_e_nm - M_nm
    
    max_val = 2 ** max_e
    if nosub:
        min_val = 2 ** min_e_nm
    else:
        min_val = 2 ** min_e_un
    #print("Max_e:", max_e)
    #print("min_e_nm:", min_e_nm)
    #print("min_e_un:", min_e_un)
    #print("------------------------------------------------------------")

    sign = torch.sign(x)
    x = torch.abs(x)
    x[x < min_val] = 0.0
    max_exceeding_mask = (x > max_val)  # max값을 초과하는 값들을 식별

    # 양자화할 값들만 처리
    x_to_quantize = x.clone()
    x_to_quantize[max_exceeding_mask] = 0  # max값을 초과하는 값들은 양자화에서 제외

    non_zero_id = (x_to_quantize > 0)
    exp = torch.floor(torch.log2(x_to_quantize[non_zero_id]))
    fract = x_to_quantize[non_zero_id] / (2 ** exp) - 1

    M = torch.zeros_like(fract)
    if nosub:
        M = M_nm
    else:
        nm_id  = exp >= min_e_nm
        um_id = exp < min_e_nm
        M[nm_id] = M_nm
        M[um_id] = (exp[um_id] - min_e_un)

    fract = torch.round(fract * (2 ** M)) / (2 ** M)
    x_to_quantize[non_zero_id] = sign[non_zero_id] * (2 ** exp) * (1 + fract)
    
    # 원래 값을 유지해야 하는 값들을 처리
    x_result = x.clone()
    x_result[~max_exceeding_mask] = x_to_quantize[~max_exceeding_mask]
    
    #x[non_zero_id] = x_result

    return x_result
'''


def PQuantF(x, N, E):
    W = N - 2
    useed = (2 ** (2 ** E))
    step_e = 2 ** E
    max_k = W - 1
    min_k = -W
    max_val = useed ** max_k
    min_val = useed ** min_k

    sign = torch.sign(x)
    x = torch.abs(x)    
    x[x < min_val] = 0.0
    x = torch.clamp(x, max=max_val)

    non_zero_id = x > 0
    exp_fp = torch.floor(torch.log2(x[non_zero_id]))
    fract_fp = x[non_zero_id] / (2 ** exp_fp) - 1
    k = torch.floor(exp_fp / step_e)
    exp_ps = exp_fp - k * step_e
        
    R = torch.where(k >= 0, k + 1, -k)
    M = torch.max(torch.zeros(1).type(torch.cuda.FloatTensor), W - R - E)

    fract_ps = torch.round(fract_fp * (2 ** M)) * (2 ** (-M))
    x[non_zero_id] = sign[non_zero_id]*(useed ** k) * (2 ** exp_ps) * (1 + fract_ps)

    return x

def NQuantF(x, N, E):
    W = N - 2
    step_e = (2 ** E - 1)
    min_e = - np.ceil(W / E)*step_e
    max_e = - min_e
    max_val = 2 ** (max_e)
    min_val = 2 ** (min_e)

    sign = torch.sign(x)
    x = torch.abs(x)
    x[x < min_val] = 0.0
    x = torch.clamp(x, max=max_val)

    non_zero_id = x>0
    exp_fp = torch.floor(torch.log2(x[non_zero_id]))
    fract_fp = x[non_zero_id] / (2 ** exp_fp) - 1
    k = torch.floor(exp_fp / step_e)

    R = torch.where(k >= 0, (k + 1) * E, -k * E)
    M = torch.max(torch.zeros(1).type(torch.cuda.FloatTensor), W - R)

    fract_ps = torch.round(fract_fp * (2**M))/(2**M)

    x[non_zero_id] = sign[non_zero_id]*(2 ** exp_fp) * (1 + fract_ps)

    return x

# Outlier 아닌 부분과 outlier인 부분 나눠서 양자화 하는 함수 
def IntQuantl(x, S, N):
    level = 2**(N - 1) - 1
    
    ###out = torch.round(x * S)
    ###out = torch.clamp(out, max=level)
    
    scaling_int = x * S
    
    global s_count
    s_count += 1
    '''
    if s_count <= 113:
        x_f111 = scaling_int.flatten()
        x_list111 = x_f111.tolist()
        with open("./llama3.2_1b_outlier/max_65.00/outlier_fp4_e2_exp2_fix3/after_scale_act/after_scale_act_{}.pkl".format(s_count), "wb") as f:
            pickle.dump(x_list111, f)
    '''
    max_exceeding_mask = (scaling_int >= level)  # max값을 초과하는 값들을 식별
    
    x_to_quantize = scaling_int.clone()
    #print("Before", x_to_quantize[max_exceeding_mask])
    x_to_quantize[max_exceeding_mask] = 0  # max값을 초과하는 값들은 양자화에서 제외
    #print("After", x_to_quantize[max_exceeding_mask])
    x_to_quantize = torch.round(x_to_quantize)
    
    out = scaling_int.clone()
    out[~max_exceeding_mask] = x_to_quantize[~max_exceeding_mask]
    
    # 기존의 outlier를 양자화 하는 코드(이 코드 사용)
    
    outlier_data = out[max_exceeding_mask]
    #print("outlier_data:", outlier_data)
    original_max = torch.max(scaling_int)  
    original_min = 2**4
    outlier_data = IntQuantF(original_max, outlier_data, N=4)
    #outlier_data = FlotQuantF(outlier_data, original_max, original_min, N=4, E=3) # Original code
    #outlier_data = FlotQuantF(outlier_data, original_max, level, N=4, E=2) 
    #outlier_data = MersitQuantF(outlier_data, original_max, N=5, E=2)
    #outlier_data = PositQuantF(outlier_data, original_max, N=4, E=1)
    
    #print("IntQuantF:", IntQunatF(original_max, outlier_data))
    out[max_exceeding_mask] = outlier_data
    #print("outlier quant data:", out[max_exceeding_mask])
    #print("--------------------------------------------------------")
    
    
    return out

"""
def IntQunatF(max_val, outlier_data, N=6):
    level = 2**(N - 1) - 1
    #int_scale = level / max_val
    int_scale = level / max_val # original code
    ###int_scale = level / (max_val - scale_max)
    #print("level:", level)
    #print("max_val:", max_val)
    x_int_scaled = outlier_data * int_scale
    #print("outlier_data:", outlier_data)
    #print("---------------------------------------------------")
    
    global b_int_count
    b_int_count += 1
    '''
    if b_int_count <= 100:
        x_o3 = x_int_scaled.flatten()
        x_list_o3 = x_o3.tolist()
        with open("./mobilellm_1b_file/int5/max_65.00/outlier_int5/outlier_plot/after_scale_act/after_scale_act_{}.pkl".format(b_int_count), "wb") as f:
            pickle.dump(x_list_o3, f)
    '''
    
    out = torch.round(x_int_scaled)
    #print("x_int_scaled:", x_int_scaled)
    #print("After_Round:", out)
    #print("---------------------------------------")
    out = torch.clamp(out, max=level)
    '''
    if b_int_count <= 100:
        x_o4 = out.flatten()
        x_list_o4 = x_o4.tolist()
        with open("./mobilellm_1b_file/int5/max_65.00/outlier_int5/outlier_plot/after_quant_act/after_quant_act_{}.pkl".format(b_int_count), "wb") as f:
            pickle.dump(x_list_o4, f)
    '''
    
    out = out / int_scale
    '''
    x_o40 = out.flatten()
    x_list_o40 = x_o40.tolist()
    with open("./efficientnet_outlier/int5/max_50.00/outlier_int5/outlier_scaling/act_quant_2/dequant_act/dequant_act_{}.pkl".format(b_int_count), "wb") as f:
        pickle.dump(x_list_o40, f)
    '''
    '''
    fwrite = open("./outlier_int5_fp6_3_rmse.csv", 'a+')
    rmse_loss = torch.sqrt(((out-outlier_data)**2).mean()).item()
    ####fwrite.write('%s, %f\n' % (g_var.fm_name, rmse_loss))
    s_rmse = str(rmse_loss) + '\n'
    fwrite.write(s_rmse)
    fwrite.close()
    '''
    
    return out
"""

def IntQuantF(max_val, outlier_data, N=6, use_log_quant=True, alpha=5.0, p=1.0):
    """
    INT Quantization with optional Logarithmic Quantization for outlier data.

    Parameters:
        max_val (float): Maximum value in the original data.
        outlier_data (torch.Tensor): Input tensor for quantization.
        N (int): Total number of bits for quantization.
        use_log_quant (bool): Whether to apply logarithmic quantization.

    Returns:
        torch.Tensor: Quantized and restored tensor.
    """
    level = 2**(N - 1) - 1  # Maximum quantization level
    epsilon = 1e-5  # Stability constant for log scaling

    if use_log_quant:
        '''
        # Logarithmic Quantization
        log_outlier_data = torch.log2(torch.clamp(outlier_data, min=epsilon))  # Log-transform input
        log_max_val = torch.log2(torch.tensor(max_val))  # Log2 of the maximum value for scaling
        log_scale = level / log_max_val  # Compute scale factor for log domain

        # Scale the log-transformed data
        scaled_data = log_outlier_data * log_scale
        '''

        # Step 1: Logarithmic Transformation with range adjustment
        log_outlier_data = torch.log2(torch.clamp(outlier_data, min=epsilon)) * alpha  # Apply scaling

        '''
        if b_int_count <= 113:
            x_log_scaled = log_outlier_data.flatten()
            x_list_log_scaled = x_log_scaled.tolist()
            with open("./llama3.2_1b_log/outlier_int4/outlier_scaling/after_log_alpha/after_log_alpha_{}.pkl".format(b_int_count), "wb") as f:
                pickle.dump(x_list_log_scaled, f)
        '''

        #log_outlier_data = torch.pow(log_outlier_data, p)  # Non-linear expansion if p != 1
        log_max_val = torch.log2(torch.tensor(max_val)) * alpha  # Log2 of max_val with scaling
        log_scale = level / log_max_val  # Compute scale factor for log domain

        # Scale the log-transformed data
        scaled_data = log_outlier_data * log_scale

        global b_int_count
        b_int_count += 1
        '''
        if b_int_count <= 113:
            x_log_scaled = scaled_data.flatten()
            x_list_log_scaled = x_log_scaled.tolist()
            with open("./llama3.2_1b_log/outlier_int4/outlier_scaling/after_scale_act/after_scale_act_{}.pkl".format(b_int_count), "wb") as f:
                pickle.dump(x_list_log_scaled, f)
        '''

        # Quantization
        quantized_data = torch.round(scaled_data)  # Round to nearest integer
        quantized_data = torch.clamp(quantized_data, max=level)  # Clamp to quantization range

        '''
        if b_int_count <= 113:
            x_quantized = quantized_data.flatten()
            x_list_quantized = x_quantized.tolist()
            with open("llama3.2_1b_log/outlier_int4/outlier_scaling/after_quant_act/after_quant_act_{}.pkl".format(b_int_count), "wb") as f:
                pickle.dump(x_list_quantized, f)
        '''

        # Restore to log domain
        restored_log_data = quantized_data / log_scale  # Rescale back to log domain
        '''
        if b_int_count <= 113:
            x_quantized1 = restored_log_data.flatten()
            x_list_quantized1 = x_quantized1.tolist()
            with open("llama3.2_1b_log/outlier_int4/outlier_scaling/dequant_act/dequant_act_{}.pkl".format(b_int_count), "wb") as f:
                pickle.dump(x_list_quantized1, f)
        '''
        # Inverse-log to return to original domain
        #restored_data = torch.pow(2, restored_log_data)  # Apply inverse log2 transformation
        restored_data = torch.pow(2, restored_log_data / alpha)
        '''
        if b_int_count <= 113:
            x_quantized11 = restored_data.flatten()
            x_list_quantized11 = x_quantized11.tolist()
            with open("llama3.2_1b_log/outlier_int4/outlier_scaling/inverse_log/inverse_log_{}.pkl".format(b_int_count), "wb") as f:
                pickle.dump(x_list_quantized11, f)
        '''
    else:
        # Standard INT Quantization
        int_scale = level / max_val  # Compute scale factor for INT quantization
        x_int_scaled = outlier_data * int_scale

        #global b_int_count
        #b_int_count += 1
        '''
        if b_int_count <= 100:
            x_o3 = x_int_scaled.flatten()
            x_list_o3 = x_o3.tolist()
            with open("./int_quant/after_scale_act/after_scale_act_{}.pkl".format(b_int_count), "wb") as f:
                pickle.dump(x_list_o3, f)
        '''

        # Quantization
        out = torch.round(x_int_scaled)  # Round to nearest integer
        out = torch.clamp(out, max=level)  # Clamp to quantization range

        '''
        if b_int_count <= 100:
            x_o4 = out.flatten()
            x_list_o4 = x_o4.tolist()
            with open("./int_quant/after_quant_act/after_quant_act_{}.pkl".format(b_int_count), "wb") as f:
                pickle.dump(x_list_o4, f)
        '''

        # Restore to original scale
        restored_data = out / int_scale

    '''
    if b_int_count <= 100:
        x_restored = restored_data.flatten()
        x_list_restored = x_restored.tolist()
        with open("./quant_restored/dequant_act/dequant_act_{}.pkl".format(b_int_count), "wb") as f:
            pickle.dump(x_list_restored, f)
    '''
    
    return restored_data




# 기존의 outlier를 FP로 양자화 하는 코드(original code)
"""
def FlotQuantF(x, original_max, original_min, N=6, E=3):
    M_nm = N-1-E
    bias = (2**(E-1)-1)
    max_e = (2**E-1) - bias 
    #max_e = ((2 ** E - 1) - bias) * 2   # 지수를 2배로 늘리기 위해 2를 곱함
    #min_e_nm = 1 - bias # original code
    #min_e_un = min_e_nm - M_nm # original code
    min_e_nm = (1 - bias) * 2  # 음수 범위 두 배로 확장
    #min_e_nm = (1 - bias) - 2  # 음수 범위 두 배로 확장(exp가 0이되는 경우)
    #min_e_nm = (1 - bias) * 2  # 음수 범위 두 배로 확장(0이 )
    min_e_un = (min_e_nm - M_nm) * 2  # 서브노멀 최소 지수도 두 배로 확장
    #print("min_e_nm:", min_e_nm)
    #print("min_e_un:", min_e_un)
    #print("max_e:", max_e)
    
    max_val = 2 ** max_e
    min_val = 2 ** min_e_un

    quant_r = 2**((2**E-1) - bias)
    fp_scale = quant_r/original_max # 원래 코드
    ###fp_scale = quant_r / (original_max - original_min)  # 스케일 조정
    ###x_s = (x - original_min) * fp_scale  # 입력 데이터를 quantization 범위에 맞게 이동 및 스케일링
    
    global o_count
    o_count += 1
    '''
    if o_count <= 113:
        x_f = x.flatten()
        x_list = x_f.tolist()
        with open("./llama3.2_1b_outlier/max_65.00/outlier_fp4_e2_exp2/outlier_scaling/before_quant_act/before_quant_act_{}.pkl".format(o_count), "wb") as f:
            pickle.dump(x_list, f)
    '''
    
    x_s = x * fp_scale # 원래 코드
    '''
    if o_count <= 113:
        x_f1 = x_s.flatten() 
        x_list1 = x_f1.tolist()
        with open("./llama3.2_1b_outlier/max_65.00/outlier_fp4_e2_exp2/outlier_scaling/after_scale_act/after_scale_act_{}.pkl".format(o_count), "wb") as f:
            pickle.dump(x_list1, f)
    '''
    
    sign = torch.sign(x_s)
    x_s = torch.abs(x_s)
    x_s[x_s < min_val] = 0.0 # Original code
    #x_s[x_s < min_val] = min_val
    #x_s[(x_s < min_val) & (x_s != 0)] = min_val
    #current_min = x_s.min()  # x_s의 현재 최솟값을 구합니다.
    #x_s[(x_s < min_val) & (x_s != 0)] = current_min  # min_val보다 작은 값은 모두 current_min으로 치환합니다.
    #x_min = original_min * fp_scale
    #x_s[(x_s < min_val) & (x_s != 0)] = x_min  

    x_s = torch.clamp(x_s, max=max_val)
    
    non_zero_id = (x_s > 0)
    exp = torch.floor(torch.log2(x_s[non_zero_id])) # 원래 코드

    # 여기서 모든 exponent 값에 대해 2배로 확장
    #exp = exp * 2

    # 음수 exponent 값에 대해 2배로 확장
    negative_exp_mask = exp < 0  # 음수 지수 선택
    exp[negative_exp_mask] = exp[negative_exp_mask] * 2  # 음수 지수를 2배로 확장
    
    fract = x_s[non_zero_id] / (2 ** exp) - 1
    
    M = torch.zeros_like(fract)
    #M = M_nm
    nm_id  = exp >= min_e_nm
    um_id = exp < min_e_nm
    M[nm_id] = M_nm
    M[um_id] = (exp[um_id] - min_e_un)
    
    fract = torch.round(fract * (2 ** M)) / (2**M)
    
    x_s[non_zero_id] = sign[non_zero_id] * (2 ** exp) * (1 + fract)
    '''
    if o_count <= 113:
        x_f11 = x_s.flatten()
        x_list11 = x_f11.tolist()
        with open("./llama3.2_1b_outlier/max_65.00/outlier_fp4_e2_exp2/outlier_scaling/after_quant_act/after_quant_act_{}.pkl".format(o_count), "wb") as f:
            pickle.dump(x_list11, f)
    '''
    
    x_s = x_s / fp_scale
    '''
    if o_count <= 113:
        x_f111 = x_s.flatten()
        x_list111 = x_f111.tolist()
        with open("./llama3.2_1b_outlier/max_65.00/outlier_fp8_e4/outlier_scaling/dequant_act/dequant_act_{}.pkl".format(o_count), "wb") as f:
            pickle.dump(x_list111, f)
    '''
    
    return x_s
"""

def FlotQuantF(x, original_max, original_min, N=6, E=3, use_log_quant=True):
    """
    Floating Point Quantization with optional Logarithmic Quantization and scaling factor.

    Parameters:
        x (torch.Tensor): Input tensor.
        original_max (float): Maximum value in the original range.
        original_min (float): Minimum value in the original range.
        N (int): Total number of bits.
        E (int): Number of bits for exponent.
        use_log_quant (bool): Whether to apply logarithmic quantization.

    Returns:
        torch.Tensor: Quantized and restored tensor.
    """
    M_nm = N - 1 - E  # Mantissa bit-width
    bias = (2**(E - 1) - 1)
    max_e = (2**E - 1) - bias  # Maximum exponent
    min_e_nm = (1 - bias) * 2  # Normalized minimum exponent
    min_e_un = (min_e_nm - M_nm) * 2  # Subnormal minimum exponent
    
    max_val = 2 ** max_e
    min_val = 2 ** min_e_un

    quant_r = 2**((2**E - 1) - bias)
    fp_scale = quant_r / original_max  # Scaling factor for normalization

    global o_count
    o_count += 1

    # Step 1: Apply Logarithmic Transformation (if enabled)
    if use_log_quant:
        # Log transformation
        epsilon = 1e-5  # Stability for log2
        x_log = torch.log2(torch.clamp(x, min=epsilon))  # Log-transform input

        '''
        if o_count <= 113:
            x_f = x.flatten()
            x_list = x_f.tolist()
            with open("./llama3.2_1b_log/outlier_scaling/outlier_fp4_e2/before_quant_act/before_quant_act_{}.pkl".format(o_count), "wb") as f:
                pickle.dump(x_list, f)

        if o_count <= 113:
            x_f1 = x_log.flatten()
            x_list1 = x_f1.tolist()
            with open("./llama3.2_1b_log/outlier_scaling/outlier_fp4_e2/after_log/after_log_{}.pkl".format(o_count), "wb") as f:
                pickle.dump(x_list1, f)
        '''

        # Log-transformed maximum value
        log_original_max = torch.log2(torch.tensor(original_max))

        # Scaling factor
        fp_scale = quant_r / log_original_max


        x_log = x_log * fp_scale  # Apply scaling factor to normalize range
        '''
        if o_count <= 113:
            x_f11 = x_log.flatten()
            x_list11 = x_f11.tolist()
            with open("./llama3.2_1b_log/outlier_scaling/outlier_fp4_e2/after_scale_act_fix/after_scale_act_fix_{}.pkl".format(o_count), "wb") as f:
                pickle.dump(x_list11, f)
        '''
        # Subnormal handling
        x_log[x_log < min_val] = 0.0  # Values below minimum are set to 0
        x_log = torch.clamp(x_log, max=max_val)  # Limit to maximum value

        # Quantization in log domain
        sign = torch.sign(x_log)  # Sign extraction
        x_log = torch.abs(x_log)
        non_zero_id = (x_log > 0)  # Non-zero values mask
        exp = torch.floor(x_log[non_zero_id])  # Compute exponent
        fract = x_log[non_zero_id] / (2 ** exp) - 1  # Compute mantissa

        # Mantissa quantization
        M = torch.zeros_like(fract)  # Mantissa bit allocation
        nm_id = exp >= min_e_nm  # Normalized range
        um_id = exp < min_e_nm  # Subnormal range
        M[nm_id] = M_nm
        M[um_id] = exp[um_id] - min_e_un  # Adjust for subnormal
        fract = torch.round(fract * (2 ** M)) / (2 ** M)  # Quantize mantissa

        # Reconstruct quantized values
        x_quantized = sign[non_zero_id] * (2 ** exp) * (1 + fract)
        x_log[non_zero_id] = x_quantized  # Update quantized values
        '''
        if o_count <= 113:
            x_f111 = x_log.flatten()
            x_list111 = x_f111.tolist()
            with open("./llama3.2_1b_log/outlier_scaling/outlier_fp4_e2/after_quant_act_fix/after_quant_act_fix_{}.pkl".format(o_count), "wb") as f:
                pickle.dump(x_list111, f)
        '''

        # Restore to original domain using inverse-log
        x_s = torch.pow(2, x_log / fp_scale)  # Restore original values (division by fp_scale added)
        '''
        if o_count <= 113:
            x_f1111 = x_s.flatten()
            x_list1111 = x_f1111.tolist()
            with open("./llama3.2_1b_log/outlier_scaling/outlier_fp4_e2/inverse_log/inverse_log_{}.pkl".format(o_count), "wb") as f:
                pickle.dump(x_list1111, f)
        '''
    else:
        # Step 2: Standard Floating Point Quantization (Original Logic)
        x_s = x * fp_scale  # Normalize input
        sign = torch.sign(x_s)
        x_s = torch.abs(x_s)
        x_s[x_s < min_val] = 0.0  # Subnormal handling
        x_s = torch.clamp(x_s, max=max_val)  # Clamp to max range
        
        non_zero_id = (x_s > 0)
        exp = torch.floor(torch.log2(x_s[non_zero_id]))  # Compute exponent

        # Expand negative exponents
        negative_exp_mask = exp < 0
        exp[negative_exp_mask] = exp[negative_exp_mask] * 2  # Double negative exponents
        
        # Compute mantissa
        fract = x_s[non_zero_id] / (2 ** exp) - 1
        M = torch.zeros_like(fract)
        nm_id = exp >= min_e_nm
        um_id = exp < min_e_nm
        M[nm_id] = M_nm
        M[um_id] = exp[um_id] - min_e_un
        fract = torch.round(fract * (2 ** M)) / (2 ** M)

        # Reconstruct quantized values
        x_s[non_zero_id] = sign[non_zero_id] * (2 ** exp) * (1 + fract)
        x_s = x_s / fp_scale  # Restore original scale

    return x_s




def MersitQuantF(x, original_max, N=5, E=2):

    
    W = N - 2
    step_e = (2 ** E - 1)
    min_e = - np.ceil(W / E)*step_e
    max_e = - min_e
    max_val = 2 ** (max_e)
    min_val = 2 ** (min_e)
    ###print("max_e:", max_e)
    ###print("min_e:", min_e)
    
    q_range = 2**((2**E-1)) 
    m_scale = q_range / original_max
    x = x * m_scale

    sign = torch.sign(x)
    x = torch.abs(x)
    x[x < min_val] = 0.0
    x = torch.clamp(x, max=max_val)

    non_zero_id = x>0
    exp_fp = torch.floor(torch.log2(x[non_zero_id]))
    fract_fp = x[non_zero_id] / (2 ** exp_fp) - 1
    k = torch.floor(exp_fp / step_e)

    R = torch.where(k >= 0, (k + 1) * E, -k * E)
    M = torch.max(torch.zeros(1).type(torch.cuda.FloatTensor), W - R)

    fract_ps = torch.round(fract_fp * (2**M))/(2**M)

    x[non_zero_id] = sign[non_zero_id]*(2 ** exp_fp) * (1 + fract_ps)

    x = x / m_scale

    return x

def PositQuantF(x, original_max, N=5, E=2):
    W = N - 2
    useed = (2 ** (2 ** E))
    step_e = 2 ** E
    max_k = W - 1
    min_k = -W
    max_val = useed ** max_k
    min_val = useed ** min_k

    quant_r = 2**((2**E))
    p_scale = quant_r / original_max
    x = x * p_scale
    
    sign = torch.sign(x)
    x = torch.abs(x)    
    #x[x < min_val] = 0.0 # Original code
    x[x < min_val] = min_val
    x = torch.clamp(x, max=max_val)

    non_zero_id = x > 0
    exp_fp = torch.floor(torch.log2(x[non_zero_id]))
    fract_fp = x[non_zero_id] / (2 ** exp_fp) - 1
    k = torch.floor(exp_fp / step_e)
    exp_ps = exp_fp - k * step_e
        
    R = torch.where(k >= 0, k + 1, -k)
    M = torch.max(torch.zeros(1).type(torch.cuda.FloatTensor), W - R - E)

    fract_ps = torch.round(fract_fp * (2 ** M)) * (2 ** (-M))
    x[non_zero_id] = sign[non_zero_id]*(useed ** k) * (2 ** exp_ps) * (1 + fract_ps)
    
    x = x / p_scale

    return x


#Float Quantization Fucntion*****************************************************
class FQuant(Function):
    @staticmethod
    def forward(ctx, x, N, E, nosub):
        return FQuantF(x, N, E, nosub)

# New Poist Quantization Class***************************************************
class NQuant(Function):
    @staticmethod
    def forward(ctx, x, N, E):
        return NQuantF(x, N, E)

#Posit Quantization Class *******************************************************
class PQuant(Function):
    @staticmethod
    def forward(ctx, x, N, E):
        return PQuantF(x, N, E)

# Original INT Quant code
'''
class IntQuant(Function):
    @staticmethod
    def forward(ctx, x, S, N):
        #debug_log("Entering IntQuant forward", S=S, N=N)
        level = 2**(N - 1) - 1
        scaling_int = x * S
        ###out = torch.round(x * S)

        global s_count
        s_count += 1
        
        if s_count <= 113:
            x_f111 = scaling_int.flatten()
            x_list111 = x_f111.tolist()
            with open("./llama3.2_1b_file/w4a4/activation/after_scale_act/after_scale_act_{}.pkl".format(s_count), "wb") as f:
                pickle.dump(x_list111, f)
        

        out = torch.round(scaling_int)
        out = torch.clamp(out, max=level)
        #debug_log("Exiting IntQuant forward", out_min=out.min().item(), out_max=out.max().item())
        return out
'''

# Outlier 양자화 하는 코드 INT Quant code

class IntQuant(Function):
    @staticmethod
    def forward(ctx, x, S, N):
        
        return IntQuantl(x, S, N)


class quantizer(nn.Module):
    def __init__(self, channels, qinfo):
        super(quantizer, self).__init__()

        self.qinfo = qinfo
        '''
        if self.qinfo.data == 'weight':
            self.observer = NormalMinMaxObserver(channels)
        else:
            self.observer = NormalMinMaxObserver(channels)
        '''

        # 전체 분포의 몇%를 max값으로 잡는 observer
        
        if self.qinfo.data == 'weight':
            self.observer = NineNineObserver(channels)
        else:
            self.observer = NineNineObserver(channels)
        

        self.register_buffer('scale', torch.ones_like((self.observer.max_val), dtype=torch.float32))
        self.register_buffer('zero_point', torch.zeros_like((self.observer.max_val), dtype=torch.float32))

    def update_quant_params(self):
        if self.qinfo.qm<=5:
            #debug_log("Updating quantization parameters", qinfo=self.qinfo)
            quant_range = 2**(self.qinfo.n-1) - 1
            data_range = torch.max(torch.abs(self.observer.max_val), torch.abs(self.observer.min_val))
            self.scale = quant_range/data_range
            if self.qinfo.data == 'act':
                print("data_range:", data_range)

        elif self.qinfo.qm==6:
            bias = 2**(self.qinfo.e-1)-1
            quant_range = 2**((2**self.qinfo.e-2) - bias)
            data_range = torch.max(torch.abs(self.observer.max_val), torch.abs(self.observer.min_val))
            self.scale = quant_range/data_range #2**(-torch.log2(data_range))#data_range/data_range
            if self.qinfo.data == 'act':
                print("data_range:", data_range)


        elif self.qinfo.qm==7:        
            quant_range = 2**((2**self.qinfo.e)) #*(self.qinfo.n-2-self.qinfo.e))
            data_range = torch.max(torch.abs(self.observer.max_val), torch.abs(self.observer.min_val))
            self.scale = quant_range/data_range #2**(-torch.log2(data_range))#data_range/data_range
        
        elif self.qinfo.qm==8:
            quant_range = 2**((2**self.qinfo.e-1)) #(((self.qinfo.n-2)//self.qinfo.e)-1))
            #print(quant_range)
            data_range = torch.max(torch.abs(self.observer.max_val), torch.abs(self.observer.min_val))
            self.scale = quant_range/data_range #2**(-torch.log2(data_range))#data_range/data_range
            
        self.zero_point = torch.zeros_like(self.scale)
        #debug_log("Updated quantization parameters", scale=self.scale, zero_point=self.zero_point)

    def truncate(self, x, bitwidth):
        dev_factor = 2 ** (8-bitwidth)
        x = torch.round(x / dev_factor) * dev_factor
        return x

    def clipping(self, x, sign, bitwidth):       
        if sign == enums.signed:
            maxa = 2**(bitwidth-1)
        else:
            maxa = 2**bitwidth       
        x = torch.clamp(x, max=maxa-1)
        return x

    def forward(self, x):
        if self.qinfo.phase == 0: #train
            return x
        elif self.qinfo.phase == 1:
            self.observer(x)
            return x
        elif self.qinfo.phase == 2:
            if self.qinfo.qm<=5:
                #debug_log("Quantizing input", x_min=x.min().item(), x_max=x.max().item())
                
                global c_count

                if self.qinfo.data == 'act':
                    c_count += 1
                    ###print(c_count)
                
                
                '''    
                if self.qinfo.data == 'weight':
                    c_count += 1
                    ###print(c_count)
                '''
                
                '''
                if self.qinfo.data == 'act' and c_count <= 113:
                    x_f = x.flatten()
                    x_list = x_f.tolist()
                    with open("./llama3.2_1b_outlier/max_65.00/outlier_int8/before_quant_act/before_quant_act_{}.pkl".format(c_count), "wb") as f:
                        pickle.dump(x_list, f)
                    print("-------------------------------------------------------")
                    print("self.qinfo.n: ", self.qinfo.n)
                    print("self.qinfo.e: ", self.qinfo.e)
                    print("count: ", c_count)
                '''


                sign = torch.sign(x)
                out = torch.abs(x)
                out = IntQuant.apply(out, self.scale, self.qinfo.n)


                '''
                if self.qinfo.data == 'act' and c_count <= 113:
                    x_f1 = out.flatten()
                    x_list1 = x_f1.tolist()
                    with open("./llama3.2_1b_outlier/max_65.00/outlier_fp4_e2_exp2_fix4/after_quant_act/after_quant_act_{}.pkl".format(c_count), "wb") as f:
                        pickle.dump(x_list1, f)
                '''


                if self.qinfo.o2an > 0:
                    if self.qinfo.qm == 2: #normal O2A
                        out = O2AQuant.apply(out, self.qinfo.data, self.qinfo.o2an, self.qinfo.o2ag)
                    elif self.qinfo.qm == 3: #Multiple Level O2A
                        out = MulO2AQuant.apply(out, self.qinfo.data, self.qinfo.o2an, self.qinfo.o2ag)
                    elif self.qinfo.qm == 4: #Truncation
                        out = self.truncate(out, self.qinfo.o2an)
                    elif self.qinfo.qm == 5: #Clipping
                        out = self.clipping(out, self.qinfo.data, self.qinfo.o2an)

                out = (out*sign)/self.scale
                #debug_log("Quantized and dequantized output", out_min=out.min().item(), out_max=out.max().item())

                '''
                if self.qinfo.data == 'act' and c_count <= 113:
                    x_f111 = out.flatten()
                    x_list111 = x_f111.tolist()
                    with open("./llama3.2_1b_outlier/max_65.00/outlier_int8/dequant_act/dequant_act_{}.pkl".format(c_count), "wb") as f:
                        pickle.dump(x_list111, f)
                '''

            else:
                out = x*self.scale

                global f_count
                if self.qinfo.data == 'act':
                    f_count += 1
                    ###print(c_count)

                '''
                if self.qinfo.data == 'act' and f_count <= 113:
                    x_f1 = out.flatten()
                    x_list1 = x_f1.tolist()
                    with open("./llama3.2_1b_outlier/max_65.00/fp8_e3/after_scale_act/after_scale_act_{}.pkl".format(f_count), "wb") as f:
                        pickle.dump(x_list1, f)
                '''

                if self.qinfo.qm==6:    #FP
                    out = FQuant.apply(out, self.qinfo.n, self.qinfo.e, self.qinfo.nosub)

                    '''
                    if self.qinfo.data == 'act' and f_count <= 113:
                        x_f111 = out.flatten()
                        x_list111 = x_f111.tolist()
                        with open("./llama3.2_1b_outlier/max_65.00/fp8_e3/after_quant_act/after_quant_act_{}.pkl".format(f_count), "wb") as f:
                            pickle.dump(x_list111, f)
                    '''


                elif self.qinfo.qm==7:  #Posit
                    out = PQuant.apply(out, self.qinfo.n, self.qinfo.e)
                elif self.qinfo.qm==8:  #New
                    out = NQuant.apply(out, self.qinfo.n, self.qinfo.e)
                out = out/self.scale

                '''
                if self.qinfo.data == 'act' and f_count <= 200:
                    x_f111 = out.flatten()
                    x_list111 = x_f111.tolist()
                    with open("./mobilellm_1b_file/int5/max_65.00/outlier_fp5_e4/dequant_act/dequant_act_{}.pkl".format(f_count), "wb") as f:
                        pickle.dump(x_list111, f)
                '''

        return out

class IntQLinear(nn.Linear):
    def __init__(self, ic, oc, bias=True, is_qw=True, is_qa=True):
        super(IntQLinear, self).__init__(ic, oc, bias)
        self.is_qw = is_qw
        self.is_qa = is_qa
        self.qinfoa = QInfo(phase=opt.qphase, data='act', qm=opt.qm, n=opt.qna, e=opt.qe, o2an=opt.o2aa, o2ag=opt.o2ag, nosub=opt.qnosub)
        if is_qa:
            self.quantA = quantizer(0, qinfo=self.qinfoa)

        self.qinfow = QInfo(phase=opt.qphase, data='weight', qm=opt.qm, n=opt.qnw, e=opt.qe, o2an=opt.o2aw, o2ag=opt.o2ag, nosub=opt.qnosub)
        if is_qw:
            self.quantW = quantizer(oc, qinfo=self.qinfow)

        # Ensure self.bias remains in FP32 if bias is enabled
        if self.bias is not None:
            self.bias = torch.nn.Parameter(self.bias.detach().float())

    def forward(self, x):
        #debug_log("IntQLinear forward pass start", input_shape=x.shape)
        if self.is_qa and self.qinfoa.qm:
            aquant = self.quantA(x)
        else:
            aquant = x
        
        if self.is_qw and self.qinfow.qm:
            wquant = self.quantW(self.weight)        
        else:
            wquant = self.weight
        
        out = F.linear(aquant, wquant, self.bias)

        #debug_log("IntQLinear forward pass end", output_shape=out.shape)
        return out


class IntQEmbedding(nn.Embedding):

    def __init__(self, num, dim, pd=None, is_qw=False):
        super(IntQEmbedding, self).__init__(num, dim, pd)
        self.is_qw = is_qw

        self.qinfow = QInfo(phase=opt.qphase, data='weight', qm=opt.qm, n=16, e=opt.qe, nosub=opt.qnosub)      
        if is_qw:
            self.quantW = quantizer(0, qinfo=self.qinfow)

    def forward(self, x):
        #debug_log("IntQEmbedding forward pass start", input_shape=x.shape)

        if self.is_qw and self.qinfow.qm:
            wquant = self.quantW(self.weight)        
        else:
            wquant = self.weight
        
        out = F.embedding(x, wquant, self.padding_idx)

        #debug_log("IntQEmbedding forward pass end", output_shape=out.shape)
        return out


'''
class IntQEmbedding(nn.Embedding):
    def __init__(self, num_embeddings, embedding_dim, padding_idx=None, is_qw=True):
        # nn.Embedding의 초기화 메서드 호출 시 padding_idx 전달
        super(IntQEmbedding, self).__init__(num_embeddings, embedding_dim, padding_idx=padding_idx)
        self.is_qw = is_qw

        # QInfo 초기화 및 양자화기 설정
        self.qinfow = QInfo(phase=opt.qphase, data='weight', qm=opt.qm, n=16, e=opt.qe, nosub=opt.qnosub)
        ###self.qinfow = QInfo(phase=opt.qphase, data='weight', qm=opt.qm, n=opt.qnw, e=opt.qe, nosub=opt.qnosub)
        if is_qw:
            self.quantW = quantizer(0, qinfo=self.qinfow)

    def forward(self, x):
        # 양자화된 weight 사용
        if self.is_qw and self.qinfow.qm:
            wquant = self.quantW(self.weight)
        else:
            wquant = self.weight

        # nn.Embedding의 forward 메서드 사용
        out = F.embedding(x, wquant, self.padding_idx)
        return out
'''