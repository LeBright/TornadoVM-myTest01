The little record of using TornadoVM.  
Backend are OpenCL and PTX.  
Intel i5-13400F  
RTX4060Ti 8G CUDA12.8  
Win11  
Using Instructions(After set TornadoVM environment)
```
mvn clean compile  
mvn clean package  
tornado                                                               \
    --threadInfo                                                      \
    --printKernel                                                     \
    -cp target\MatMul-1.0.jar MatMul.MatMul                           \
    --jvm="-Ds0.t0.device=1:0 -Ds1.t0.device=1:0 -Ds2.t0.device=1:0"  \
    > result.txt
```