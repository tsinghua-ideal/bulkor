Bulkor
===

## Overview

Bulkor designs and implements the bulk loading for Path ORAM. 

## Setup Environment

*Requirement*: Ubuntu 18.04, Intel SGX SDK 2.14, and Rust toolchain

`baseline_1` is now named `main`

If you already have a local clone, you can update it by running the following commands.
```
git branch -m baseline_1 main
git fetch origin
git branch -u origin/main main
git remote set-head origin -a
```

```
cd server
```

### For 1024B block evaluation

```
./execute.sh
```

### For 64B block evaluation 

```
git apply 1024B_to_64B.patch
./execute.sh
```

### For fully in-enclave evaluation

change the value of `TREETOP_CACHING_THRESHOLD_LOG2` in `server/enclave/src/oram_storage/mod.rs` from `24` to `24u32-1u32.log2()`
then `./execute.sh`

## Implementation

Our implementation extracts and reassembles the codes in a Rust re-implementation of ZeroTrace (https://github.com/mobilecoinfoundation/mc-oblivious), and additionally adds the bulk loading procedure and support of disks and crash consistency. The original Rust implementation of ZeroTrace does not include disk support, and the disk part in the C++ implementation of ZeroTrace is broken (https://github.com/sshsshy/ZeroTrace#other-notes).
