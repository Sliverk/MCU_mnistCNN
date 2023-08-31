#include <stdio.h>

#include <tvmgen_default.h>
// #include "model_io_vars.h"

static float input[784];
static float output[10];
struct tvmgen_default_inputs default_inputs = {.input0 = &input[0],};
struct tvmgen_default_outputs default_outputs = {.output = &output,};


int main(void)
{
    int size = 0;

    while(1){
        // printf("Starting\n");
        scanf("%d", &size);
        printf("%d\n", size);

        int t = 0;
        for(int i=0; i < size; ++i){
            scanf("%d", &t);
            input[i] = (float)(t*1.0/255);
        }
        tvmgen_default_run(&default_inputs, &default_outputs);
        int max = -99999;
        int loc = 0;
        for(int i=0; i < 10; ++i){
            if(max < output[i]){
                max = output[i];
                loc = i;
            }
        }
        printf("%d\n", loc);
        
    }
    return 0;
}