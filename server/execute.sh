#!/bin/bash
lognum=25
let tmp=${lognum}-20-1
let k=2**tmp
sed -i "s/[0-9]\{1,4\} << 10/${k} << 10/g" app/src/main.rs enclave/src/lib.rs
sed -i "s/[0-9]\{1,2\}u32-/${lognum}u32-/g" app/src/storage_ocalls/mod.rs
for ratio in 16 8 4 2 1
do
    sed -i "s/${lognum}u32-[0-9]\{1,2\}u/${lognum}u32-${ratio}u/g" app/src/storage_ocalls/mod.rs
    make
    cd bin
    nocache ./app > insert_log_${k}_${ratio}.txt
    cd ..
    sed -i "s/exercise_oram_consecutive(n /\/\/exercise_oram_consecutive(n /g" app/src/main.rs
    sed -i "s/true/false/g" enclave/src/custom_counter.rs
    make
    cd bin
    nocache ./app > bulk_log_${k}_${ratio}.txt
    rm d* l* m* p* s* tr*
    cd ..
    sed -i "s/\/\/exercise_oram_consecutive(n /exercise_oram_consecutive(n /g" app/src/main.rs
    sed -i "s/false/true/g" enclave/src/custom_counter.rs
done