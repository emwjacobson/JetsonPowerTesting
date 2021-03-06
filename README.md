# AGX Config

## DLAs

To set a frequence, write to `/sys/kernel/debug/bpmp/debug/clk/nafll_dla/{min,max}_rate`


## GPU

To get available frequencies:

`/sys/devices/17000000.gv11b/devfreq/17000000.gv11b/available_frequencies`

To set a frequency, write to this file

`/sys/devices/17000000.gv11b/devfreq/17000000.gv11b/{min,max}_freq`


## Testing Parameters

__Fan speed:__ `20%`

__CPU Gov:__ `Schedutil`

__CPU Min/Max freq (kHz):__ `1190400/2265600 (*1000 = Hz)`

__GPU Frequencies (Hz):__ `114750000 216750000 318750000 420750000 522750000 624750000 675750000 828750000 905250000 1032750000 1198500000 1236750000 1338750000 1377000000`

When going from HIGH to LOW frequency, use

`echo "FREQUENCY" | sudo tee /sys/devices/17000000.gv11b/devfreq/17000000.gv11b/{min,max}_freq`

When going from LOW to HIGH frequency, use

`echo "FREQUENCY" | sudo tee /sys/devices/17000000.gv11b/devfreq/17000000.gv11b/{max,min}_freq`



# Nano Config

## GPU

To get available frequencies:

`/sys/devices/gpu.0/devfreq/57000000.gpu/available_frequencies`

To set a frequency, write to this file

`/sys/devices/gpu.0/devfreq/57000000.gpu/{min,max}_freq`


## Testing parameters

__Fan speed:__ `N/A`

__CPU Gov:__ `Schedutil`

__CPU Min/Max freq (kHz):__ `102000/1479000 (*1000 = Hz)`

__GPU Frequencies (Hz):__ `76800000 153600000 230400000 307200000 384000000 460800000 537600000 614400000 691200000 768000000 844800000 921600000`

When going from HIGH to LOW frequency, use

`echo "FREQUENCY" | sudo tee /sys/devices/gpu.0/devfreq/57000000.gpu/{min,max}_freq`

When going from LOW to HIGH frequency, use

`echo "FREQUENCY" | sudo tee /sys/devices/gpu.0/devfreq/57000000.gpu/{max,min}_freq`


# Creating TensorRT engines

https://github.com/NVIDIA/TensorRT/tree/master/samples/opensource/trtexec

__inception_v4 prototxt:__ https://www.dropbox.com/s/b7masj8xdoycv2w/inception_v4.prototxt

For creating engine for the GPU

`/usr/src/tensorrt/bin/trtexec --deploy=inception_v4.prototxt --output=prob --workspace=4096 --fp16 --saveEngine=XXXXXXXX.trt --batch=1`

For creating engine for DLAs (if board has DLAs)

`/usr/src/tensorrt/bin/trtexec --deploy=inception_v4.prototxt --output=prob --workspace=4096 --fp16 --saveEngine=XXXXXXXX.trt --batch=1 --useDLACore=### --allowGPUFallback`

On the Jetson AGX, a workspace of 4096 was used. On the Jetson nano, a workspace of 1024 was used.

# Using TensorRT engines

`/usr/src/tensorrt/bin/trtexec --loadEngine=engine_gpu.trt --batch=1 --avgRuns=100 --duration=60 --streams=1`