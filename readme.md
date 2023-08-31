# Runing pyTorch Model in STM32F749

>**Model**:\
&emsp;&emsp; MNIST CNN LeNet-5 Model\
>**Toolchain:** \
&emsp;&emsp; pyTorch + TVM/LLVM + RIOT + GTK3+\
**Procedure:**\
&emsp;&emsp; (1) Use **pyTorch** to train and save model.\
&emsp;&emsp; (2) Use **TVM** to compile model and save it to C library format.\
&emsp;&emsp; (3) Write C file and Makefile to compile the model in **RIOT** OS. \
&emsp;&emsp; (4) Write UI/Terminal python app to communicate with MCU.


## Step 1: Train and Save Scripted Quantization Model

#### Solution 1: Quantization after Training
>**File:** `0101mnist.py`\
**Info:** Train mnistCNN as usual.\
**File:** `0102quantization.py`\
**Info:** Quantization after training.

```sh
# Usage:
python 0101mnist.py
python 0102quantization.py
```

```python
# `utils/model.py`
# Model for training
class MnistModel(nn.Module):
    def __init__(self):
        super(MnistModel, self).__init__()
        self.conv1 = nn.Conv2d(1,32,3,1)
        self.conv2 = nn.Conv2d(32,64,3,1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x,2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output =  F.log_softmax(x, dim=1)
        return output
```

```python
# `0102quantization.py`
# Quantization
model_fp32 = MnistModel()
state_dict = torch.load('weights/mnist_cnn.pth')
model_fp32.load_state_dict(state_dict)
model_fp32.eval()

model_fp32.qconfig = torch.quantization.get_default_qconfig('qnnpack')

model_fp32_prepared = torch.quantization.prepare(model_fp32)

model_int8 = torch.quantization.convert(model_fp32_prepared)

torch.save(model_int8.state_dict(),"weights/mnist_cnn_quant.pth")
```

#### Solution 2: Quantization Training (Ref)

>**File:** `0103mnist_QAT.py`\
**Info:** Train quantization mnistCNN directly.\

```sh
# Usage:
python 0103mnist_QAT.py
```

* Quantization model
```python
# `utils/qmodel.py`
# Model for training
class QMnistModel(nn.Module):
    def __init__(self):
        super(QMnistModel, self).__init__()
        self.quant = torch.ao.quantization.QuantStub()
        self.conv1 = nn.Conv2d(1,6,5,1,2)
        self.conv2 = nn.Conv2d(6,16,5,1)
        self.conv3 = nn.Conv2d(16,120,5,1)
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, 10)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.maxpool2d = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.dequant = torch.ao.quantization.DeQuantStub()
    
    def forward(self, x):
        x = self.quant(x)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool2d(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2d(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.relu3(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.dequant(x)
        output =  F.log_softmax(x, dim=1)
        return output
```

* Quantization training
```python
# `0103mnist_QAT.py`
# Build Quantization Model
model = QMnistModel()
model.eval()
model.qconfig = torch.ao.quantization.get_default_qconfig('qnnpack')
model_fp32_fused = torch.ao.quantization.fuse_modules(model, [['conv1', 'relu1'],['conv2', 'relu2'],['fc1', 'relu3']])
model_fp32_prepared = torch.ao.quantization.prepare_qat(model_fp32_fused.train())

# After Training, transform to 8bits
model_fp32_prepared.eval()
model_int8 = torch.ao.quantization.convert(model_fp32_prepared)

# Just-in-time compilation, scripted model
input_shape = [1,1,28,28]
input_data = torch.randn(input_shape)
scripted_model = torch.jit.trace(model_int8, input_data).eval()

scripted_model.save('weights/qmnist_lenet5_scripted_int8.pth')
```

## Step 2: Compiled with TVM to Generate tar Library

>**File:** `0201tvm_no_optim.py`\
**Info:** Load scripted model, compile with tvm, saved in tar library.

```sh
# Usage:
python 0201tvm_no_optim.py
```

```python
# Input shape for pyTorch model
shape_dict = {'input0': [1,1,28,28]}
model = tvmc.load('weights/qmnist_lenet5_scripted_int8.pth', shape_dict=shape_dict)
```

Then, we get the `mnistCNN.tar` under the directory `./pkg/mnistCNN/`

## Step 3: Compile with RIOT OS

#### 3.1 Writing Testing Code for MCU

>**File:** `0301mcu_mnist.c`\
**Info:** Relay model compiling with RIOT.

```c
// Very important to include input and output.
#include <tvmgen_default.h> 

// Define input output format, learn from tvmgen_default.h
static float input[784];
static float output[10];
struct tvmgen_default_inputs default_inputs = {.input0 = &input[0],};
struct tvmgen_default_outputs default_outputs = {.output = &output,};

...

// Image is sent pixel by pixel, then the value is normalized
// Refer to https://pytorch.org/vision/stable/generated/torchvision.transforms.ToTensor.html
for(int i=0; i < size; ++i){
    scanf("%d", &t);
    input[i] = (float)(t*1.0/255);
}

...

// Runing testing, gets output
tvmgen_default_run(&default_inputs, &default_outputs);
```

#### 3.2 Prepare the RIOT OS

>**File 1:** `./Makefile`\
**File 2:** `./pkg/mnistCNN/Makefile`\
**File 3:** `./pkg/mnistCNN/Makefile.include`\
**File 4:** `./RIOT/makefiles/utvm.inc.mk`\
**File 5:** `./RIOT/makefiles/utvm/Makefile.utvm`\
**Info:** Makefile to compile with RIOT OS.

* Download RIOT from github

```shell
cd $ROOT

git clone https://github.com/RIOT-OS/RIOT.git 
```

* Create `Makefile` in `ROOT` Directory

```Makefile
RIOTBASE= ./RIOT

BOARD ?= stm32f746g-disco
APPLICATION = MNIST

EXTERNAL_PKG_DIRS += pkg

USEPKG += mnistCNN 
USEMODULE += stdin

include $(RIOTBASE)/Makefile.include

CFLAGS += -Wno-strict-prototypes 
CFLAGS += -Wno-missing-include-dirs

override BINARY := $(ELFFILE)

```

The rest four makefiles could be directly downloaded and put them into the right place.

#### 3.3 Compiling and Flashing

```sh
cd $ROOT

make flash
```


## Step 4. Testing with Terminal or UI Software

#### 4.1 With Terminal
>**File 1:** `0401pc_terminal_sendimage.py`\
**Info:** Test AI Model in MCU with Terminal.

```sh
# Usage:
python 0401pc_terminal_sendimage.py
```

```python
# `0401pc_terminal_sendimage.py`
# The serial port should be adjusted according to the situation
ser = setSerial("/dev/ttyACM0")
```


#### 4.1 With UI Software
>**File 1:** `0402_pc_ui_sendImage.py`\
**Info:** Test AI Model in MCU with User Interface.


```sh
# Usage:
python 0402_pc_ui_sendImage.py
```

