#!/bin/bash
set -e

if [ -z "$APS" ]; then
  echo 'APS environment variable is not set. Please run `pixi shell` first.'
  exit 1
fi

cd $APS/examples/aspdac26-tutorial/csrc/inference

mkdir -p $APS/build/examples/aspdac26-tutorial
mkdir -p $APS/build/examples/aspdac26-tutorial/inference_model

python3 bin2header.py stories15Mq.bin $APS/build/examples/aspdac26-tutorial/inference_model/model_weights.h
python3 tokenizer2header.py tokenizer.bin $APS/build/examples/aspdac26-tutorial/inference_model/tokenizer_data.h

riscv32-unknown-elf-gcc -std=gnu99 -O3 -Wall -Wextra -fno-common -fno-builtin-printf -march=rv32imaf_zicsr_zifencei -mabi=ilp32f -mcmodel=medany -specs=htif_nano.specs -static -T htif.ld -I$MEGGINCLUDE -I$APS/build/examples/aspdac26-tutorial/inference_model -o $APS/build/examples/aspdac26-tutorial/inference.riscv inference.c tinyalloc/tinyalloc.c -lm

riscv32-unknown-elf-gcc -std=gnu99 -O3 -Wall -Wextra -fno-common -fno-builtin-printf -march=rv32imaf_zicsr_zifencei -mabi=ilp32f -mcmodel=medany -specs=htif_nano.specs -static -T htif.ld -I$MEGGINCLUDE -I$APS/build/examples/aspdac26-tutorial/inference_model -o $APS/build/examples/aspdac26-tutorial/inference_native.riscv inference_native.c tinyalloc/tinyalloc.c -lm

cd -