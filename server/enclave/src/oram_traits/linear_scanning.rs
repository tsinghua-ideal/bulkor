// Copyright (c) 2018-2021 The MobileCoin Foundation

//! This module defines a naive, linear-scanning ORAM

use super::*;

pub struct LinearScanningORAM<ValueSize: ArrayLength<u8>> {
    data: Vec<(A64Bytes<ValueSize>, u64)>, //(data, counter)
}

impl<ValueSize: ArrayLength<u8>> LinearScanningORAM<ValueSize> {
    pub fn new(size: u64) -> Self {
        Self {
            data: vec![(Default::default(), 0); size as usize],
        }
    }
}

impl<ValueSize: ArrayLength<u8>> ORAM<ValueSize> for LinearScanningORAM<ValueSize> {
    fn len(&self) -> u64 {
        self.data.len() as u64
    }
    fn access<T, F: FnOnce(&mut A64Bytes<ValueSize>, &mut u64) -> T>(
        &mut self,
        query: u64,
        f: F,
    ) -> T {
        let mut temp: A64Bytes<ValueSize> = Default::default();
        let mut counter = 0;
        for idx in 0..self.data.len() {
            temp.cmov((idx as u64).ct_eq(&query), &self.data[idx].0);
            counter.cmov((idx as u64).ct_eq(&query), &self.data[idx].1);
        }
        let result = f(&mut temp, &mut counter);
        for idx in 0..self.data.len() {
            self.data[idx].0.cmov((idx as u64).ct_eq(&query), &temp);
            self.data[idx].1.cmov((idx as u64).ct_eq(&query), &counter);
        }
        result
    }

    fn persist(&mut self, lifetime_id: u64, new_snapshot_id: u64, volatile: bool) {
        unimplemented!()
    }
}
