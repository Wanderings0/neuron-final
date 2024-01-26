from neuron import h, gui
import numpy as np
import cv2
from scipy import ndimage

h.load_file('larkumEtAl2009_2/mosinit.hoc')

tstop = 70.0
dt = 0.1
def setup_simulation():
    h.tstop = tstop  # 设置模拟运行的总时间（例如，100毫秒）
    h.dt = dt  # 设置时间步长（例如，0.025毫秒）
    # 初始化模拟
    h.finitialize()

# 调用函数以设置模拟
setup_simulation()


# for sec in h.allsec():  # 遍历所有的段
#     print('Section name: {}, Section type: {}'.format(sec.name(), sec.hname()))
#     for seg in sec:     # 遍历段中的点位
#         print('  at position:', seg.x, 'has diameter:', seg.diam)


# 根据pixel的值来生成一个IClamp对象并返回
def stim_pixel(pixel, index, max_pixel, tstop=70, tau=10):
    index = int(pixel / (max_pixel+1e-5) * 4)
    # print(f'index={index}, pixel={pixel}, max_pixel={max_pixel}')
    location = [16,12,9,6]
    section = h.apic[location[index]]
    stim = h.IClamp(section(1-(pixel*4)/(max_pixel*(index+1))))
    stim.delay = 9 + 5 / (pixel + 0.01)
    stim.dur = tstop
    stim.amp = 0.01*np.exp(-(stim.delay)/ tau) * pixel
    # print(stim.amp)

    # 创建时间向量
    time_vector = h.Vector(np.arange(0, tstop, h.dt))
    voltage_vector = h.Vector().record(h.soma(0.5)._ref_v)
    # 输入的电流大小是随着时间和电压变化的，公式为
    # I = 0.05*exp(-t/tau)*(v-1),其中v是post-synaptic voltage
    # 注：由于我们没有post-synaptic voltage的信息，我们无法直接计算I(t,v)
    # 这里我们使用一个假设的常数电压值来生成一个示例电流序列
    # 在实际情况中，这个值需要根据模型的动态电压进行更新
    # post_synaptic_voltage = -65  # 假设的常数电压值（单位：mV）
    # current_vector = h.Vector([pixel* np.exp(-(t-15) / 0.5) for t in time_vector])

    # 将电流强度向量 "播放" 到IClamp的amp属性上
    # current_vector.play(stim._ref_amp, time_vector)

    # 查看stimd的amp属性
    print(stim.amp)

    return stim
    
    




# 假设我们有一个函数simulate_neuron_model(image)返回模拟结果
def simulate_neuron_model(image):
    #将image转化为灰度图像
    height, width = image.shape
    max_pixel = np.max(image)
    rec_v = h.Vector()
    rec_v.record(h.soma(0.5)._ref_v)
    # stim_list = []
    for i in range(height):
        for j in range(width):
            pixel = image[i, j]
            index = i * width + j
            stim = stim_pixel(pixel, index, max_pixel)
            # stim_list.append(stim)
    
    # 运行模拟
    h.run()


    # 返回记录的电压
    return np.array(rec_v)

if __name__ == '__main__':
    # 存储模拟输出和标签
    # 一个随机的图像
    image = np.random.randint(0, 256, (28, 28))
    # 计算x和y方向的梯度
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    # 计算梯度
    grad_x = ndimage.convolve(image, sobel_x)
    grad_y = ndimage.convolve(image, sobel_y)

    sobel = np.uint8(np.sqrt(grad_x ** 2 + grad_y ** 2))
    print(sobel.shape)

    # for i in range(28):
    #     for j in range(28):
    #         print(f'sobel[{i},{j}]={sobel[i,j]}')
    # 模拟细胞模型
    simulated_output = simulate_neuron_model(image)
    # plot the simulated output
    import matplotlib.pyplot as plt
    plt.plot(simulated_output)
    plt.show()

    print(simulated_output.shape)

            
            



    