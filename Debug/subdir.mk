################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../rayT.cu 

CU_DEPS += \
./rayT.d 

OBJS += \
./rayT.o 


# Each subdirectory must supply rules for building sources it contributes
%.o: ../%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-10.2/bin/nvcc -G -g -O0 -gencode arch=compute_50,code=sm_50  -odir "." -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-10.2/bin/nvcc -G -g -O0 --compile --relocatable-device-code=false -gencode arch=compute_50,code=compute_50 -gencode arch=compute_50,code=sm_50  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


