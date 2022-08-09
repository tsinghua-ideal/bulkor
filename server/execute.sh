#!/bin/bash
for lognum in 31 32
do
    let tmp=${lognum}-20-1
    let k=2**tmp
    sed -i "s/[0-9]\{1,4\} << 10/${k} << 10/g" app/src/main.rs enclave/src/lib.rs
    sed -i "s/[0-9]\{1,2\}u32-/${lognum}u32-/g" app/src/storage_ocalls/mod.rs enclave/src/oram_storage/mod.rs
    for ratio in 16 8 4 2
    do
        sed -i "s/${lognum}u32-[0-9]\{1,2\}u/${lognum}u32-${ratio}u/g" app/src/storage_ocalls/mod.rs enclave/src/oram_storage/mod.rs
        make
        cd bin
        nocache ./app > insert_log_${k}_${ratio}_${num_threads}.txt
        cd ..
        sed -i "s/exercise_oram_consecutive(n /\/\/exercise_oram_consecutive(n /g" app/src/main.rs
        sed -i "s/true/false/g" enclave/src/custom_counter.rs
        for num_threads in 1 4 8 16 
        do
            sed -i "s/NUM_THREADS: usize = [0-9]\{1,4\}/NUM_THREADS: usize = ${num_threads}/g" app/src/storage_ocalls/shuffle_manager.rs enclave/src/oram_storage/shuffle_manager.rs
            make
            cd bin
            nocache ./app > bulk_log_${k}_${ratio}_${num_threads}.txt
            cd ..
        done
        cd bin 
        rm d* l* m* p* s* tr*
        cd ..
        sed -i "s/\/\/exercise_oram_consecutive(n /exercise_oram_consecutive(n /g" app/src/main.rs
        sed -i "s/false/true/g" enclave/src/custom_counter.rs
    done
done