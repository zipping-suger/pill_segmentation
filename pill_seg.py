import cv2
import time
import torch
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from utils import cuts_out, visualize_sam, register_shape_db, register_color_hist, \
    size_fileter, register_db_img, \
    shape_filter, pill_identify, create_color_list

# -- coding: utf-8 --

import sys
import threading
import os
import termios
from ctypes import *

sys.path.append("MVS/MvImport")
from MvCameraControl_class import *
import ctypes

libc = ctypes.CDLL("libc.so.6")  # load the C library

# Initialize SAM with Big Model ~ 380 MB
sam_checkpoint = "/home/zippingsugar/Programs/segment-anything/checkpoints/sam_vit_b_01ec64.pth"
model_type = "vit_b"
device = "cuda" if torch.cuda.is_available() else "cpu"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

# Initialize SamAutomaticMaskGenerator

# # stable parameters
# mask_generator = SamAutomaticMaskGenerator(
#     model=sam,
#     points_per_side=18,
#     pred_iou_thresh=0.95,
#     stability_score_thresh=0.96,
#     min_mask_region_area=8e2,  # Requires open-cv to run post-processing
# )

mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=18,
    pred_iou_thresh=0.95,
    stability_score_thresh=0.96,
    min_mask_region_area=8e2,  # Requires open-cv to run post-processing
)

# color list for visualization
color_list = create_color_list(15)

# register database image
img_list = register_db_img()
# register color hist
img_hist_list = register_color_hist()

# # Initialize HIKVISION Camera
SDKVersion = MvCamera.MV_CC_GetSDKVersion()
print("SDKVersion[0x%x]" % SDKVersion)

deviceList = MV_CC_DEVICE_INFO_LIST()
tlayerType = MV_GIGE_DEVICE | MV_USB_DEVICE

# ch:枚举设备 | en:Enum device
ret = MvCamera.MV_CC_EnumDevices(tlayerType, deviceList)
if ret != 0:
    print("enum devices fail! ret[0x%x]" % ret)
    sys.exit()

if deviceList.nDeviceNum == 0:
    print("find no device!")
    sys.exit()

print("Find %d devices!" % deviceList.nDeviceNum)

for i in range(0, deviceList.nDeviceNum):
    mvcc_dev_info = cast(deviceList.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
    if mvcc_dev_info.nTLayerType == MV_GIGE_DEVICE:
        print("\ngige device: [%d]" % i)
        strModeName = ""
        for per in mvcc_dev_info.SpecialInfo.stGigEInfo.chModelName:
            strModeName = strModeName + chr(per)
        print("device model name: %s" % strModeName)

        nip1 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0xff000000) >> 24)
        nip2 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x00ff0000) >> 16)
        nip3 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x0000ff00) >> 8)
        nip4 = (mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x000000ff)
        print("current ip: %d.%d.%d.%d\n" % (nip1, nip2, nip3, nip4))
    elif mvcc_dev_info.nTLayerType == MV_USB_DEVICE:
        print("\nu3v device: [%d]" % i)
        strModeName = ""
        for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chModelName:
            if per == 0:
                break
            strModeName = strModeName + chr(per)
        print("device model name: %s" % strModeName)

        strSerialNumber = ""
        for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chSerialNumber:
            if per == 0:
                break
            strSerialNumber = strSerialNumber + chr(per)
        print("user serial number: %s" % strSerialNumber)

# if sys.version >= '3':
# 	nConnectionNum = input("please input the number of the device to connect:")
# else:
# 	nConnectionNum = raw_input("please input the number of the device to connect:")
nConnectionNum = 0

if int(nConnectionNum) >= deviceList.nDeviceNum:
    print("intput error!")
    sys.exit()

# ch:创建相机实例 | en:Creat Camera Object
cam = MvCamera()

# ch:选择设备并创建句柄| en:Select device and create handle
stDeviceList = cast(deviceList.pDeviceInfo[int(nConnectionNum)], POINTER(MV_CC_DEVICE_INFO)).contents

ret = cam.MV_CC_CreateHandle(stDeviceList)
if ret != 0:
    print("create handle fail! ret[0x%x]" % ret)
    sys.exit()

# ch:打开设备 | en:Open device
ret = cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
if ret != 0:
    print("open device fail! ret[0x%x]" % ret)
    sys.exit()

# ch:探测网络最佳包大小(只对GigE相机有效) | en:Detection network optimal package size(It only works for the GigE camera)
if stDeviceList.nTLayerType == MV_GIGE_DEVICE:
    nPacketSize = cam.MV_CC_GetOptimalPacketSize()
    if int(nPacketSize) > 0:
        ret = cam.MV_CC_SetIntValue("GevSCPSPacketSize", nPacketSize)
        if ret != 0:
            print("Warning: Set Packet Size fail! ret[0x%x]" % ret)
    else:
        print("Warning: Get Packet Size fail! ret[0x%x]" % nPacketSize)

# ch:设置触发模式为off | en:Set trigger mode as off
ret = cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
if ret != 0:
    print("set trigger mode fail! ret[0x%x]" % ret)
    sys.exit()

# ch:获取数据包大小 | en:Get payload size
stParam = MVCC_INTVALUE()
memset(byref(stParam), 0, sizeof(MVCC_INTVALUE))

ret = cam.MV_CC_GetIntValue("PayloadSize", stParam)
if ret != 0:
    print("get payload size fail! ret[0x%x]" % ret)
    sys.exit()
nPayloadSize = stParam.nCurValue

# ch:开始取流 | en:Start grab image
ret = cam.MV_CC_StartGrabbing()
if ret != 0:
    print("start grabbing fail! ret[0x%x]" % ret)
    sys.exit()

data_buf = (c_ubyte * nPayloadSize)()

stDeviceList = MV_FRAME_OUT_INFO_EX()
memset(byref(stDeviceList), 0, sizeof(stDeviceList))
data_buf = (c_ubyte * nPayloadSize)()

# Load registered shape and color database
shape_db = register_shape_db()


def get_img(cam, data_buf, nPayloadSize, stDeviceList):
    ret = cam.MV_CC_GetOneFrameTimeout(byref(data_buf), nPayloadSize, stDeviceList, 1000)
    if ret == 0:
        # print("get one frame: Width[%d], Height[%d], nFrameNum[%d]" % (
        #     stDeviceList.nWidth, stDeviceList.nHeight, stDeviceList.nFrameNum))

        print("------- Frame taken -------")
        nRGBSize = stDeviceList.nWidth * stDeviceList.nHeight * 3
        stConvertParam = MV_SAVE_IMAGE_PARAM_EX()
        stConvertParam.nWidth = stDeviceList.nWidth
        stConvertParam.nHeight = stDeviceList.nHeight
        stConvertParam.pData = data_buf
        stConvertParam.nDataLen = stDeviceList.nFrameLen
        stConvertParam.enPixelType = stDeviceList.enPixelType
        stConvertParam.nImageLen = stConvertParam.nDataLen
        stConvertParam.nJpgQuality = 70
        stConvertParam.enImageType = MV_Image_Jpeg
        stConvertParam.pImageBuffer = (c_ubyte * nRGBSize)()
        stConvertParam.nBufferSize = nRGBSize
        # ret = cam.MV_CC_ConvertPixelType(stConvertParam)
        # print(stConvertParam.nImageLen)
        ret = cam.MV_CC_SaveImageEx2(stConvertParam)
        if ret != 0:
            print("convert pixel fail ! ret[0x%x]" % ret)
            del data_buf
            sys.exit()
        file_path = "raw.jpg"
        file_open = open(file_path.encode('ascii'), 'wb+')
        img_buff = (c_ubyte * stConvertParam.nImageLen)()
        libc.memcpy(byref(img_buff), stConvertParam.pImageBuffer, stConvertParam.nImageLen)
        file_open.write(img_buff)
    print("Save Image succeed!")


def on_key(event):
    global stop
    if event.key == " ":
        plt.close(fig)
    elif event.key == "q":
        stop = False


# # Load background image
# bg = Image.open("background_1.jpg")
# # Display background image
# bg.show()

# Your code here
stop = True  # initialize the loop state
while stop:
    fig = plt.figure()
    fig.canvas.manager.full_screen_toggle()  # toggle fullscreen mode
    plt.axis('off')  # hide the axis
    st = time.time()
    get_img(cam, data_buf, nPayloadSize, stDeviceList)
    image = cv2.imread("raw.jpg")
    image = cv2.resize(image, (768, 512))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    masks = mask_generator.generate(image)
    masks = size_fileter(masks, min_area=1e3, max_area=2e4)
    # cuts_out(image, masks, save_dir='raw')
    masks = shape_filter(masks, shape_db, threshold=0.1)
    cuts_out(image, masks, save_dir='cuts_out')
    masks = pill_identify(masks, cost=[0.23003319, 0.23198423, 0.17678003, 0.15274716], img_hist_list = img_hist_list, img_list=img_list) # ncc = 0.20845539,
    et = time.time()
    print("Time of processing:", (et - st))
    # print("processing...")
    plt.imshow(image)
    visualize_sam(image, masks, color_list)
    plt.axis('off')
    plt.savefig('SAM.png')
    plt.connect("key_press_event", on_key)
    print("---------  Done  ---------")
    plt.show()

# ch:停止取流 | en:Stop grab image
ret = cam.MV_CC_StopGrabbing()
if ret != 0:
    print("stop grabbing fail! ret[0x%x]" % ret)
    del data_buf
    sys.exit()

# ch:关闭设备 | Close device
ret = cam.MV_CC_CloseDevice()
if ret != 0:
    print("close deivce fail! ret[0x%x]" % ret)
    del data_buf
    sys.exit()

# ch:销毁句柄 | Destroy handle
ret = cam.MV_CC_DestroyHandle()
if ret != 0:
    print("destroy handle fail! ret[0x%x]" % ret)
    del data_buf
    sys.exit()  

del data_buf

# # Initialize realsense Camera
# camera = Camera(use_filter=False)
# im_width = 640
# im_height = 360
# _, _ = camera.get_data()  # first image is of bad quality

# while True:
#     st = time.time()
#     color_img, _ = camera.get_data()
#     crop_image = color_img[180:540, 360:920]
#     image = cv2.cvtColor(crop_image, cv2.COLOR_BGR2RGB)
#     # cv2.imwrite('raw.jpg', color_img)
#     cv2.imwrite('raw.jpg', image)
#     masks = mask_generator.generate(image)
#     masks = size_fileter(masks, min_area=5e2, max_area=2e4)
#     masks = shape_filter(masks, shape_db, threshold=0.12)
#     cuts_out(image, masks, save_dir='cuts_out')
#     masks = color_filter(masks, color_db, threshold=1.1)
#     et = time.time()
#     print("Time of processing:", (et - st))
#     plt.imshow(image)
#     visualize_sam(image, masks)
#     plt.axis('off')
#     # plt.savefig('SAM.png')
#     plt.show()
