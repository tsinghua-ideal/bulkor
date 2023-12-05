Bulkor
===

## Overview

Bulkor designs and implements the bulk loading for Path ORAM. 

## Setup Environment

*Requirement*: Ubuntu 18.04, Intel SGX SDK 2.14, and Rust toolchain

```
cd server
make
cd bin
./app
```

## Implementation

Our implementation extracts and reassembles the codes in a Rust re-implementation of ZeroTrace (https://github.com/mobilecoinfoundation/mc-oblivious), and additionally adds the bulk loading procedure and support of disks and crash consistency. The original Rust implementation of ZeroTrace does not include disk support, and the disk part in the C++ implementation of ZeroTrace is broken (https://github.com/sshsshy/ZeroTrace#other-notes).
