use sgx_types::*;

#[no_mangle]
pub unsafe extern "C" fn sbrk_o(increment: usize) -> *mut c_void {
    libc::sbrk(increment as intptr_t)
}
