#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use crate::{WBLH, WideBits};

#[cfg(target_arch = "x86_64")]
impl WideBits {
    #[target_feature(enable = "avx2")]
    unsafe fn not_avx2_words_to(src: *const u64, dst: *mut u64, word_len: usize) {
        let chunks = word_len / 4;
        let simd_words = chunks * 4;
        let all_ones = _mm256_set1_epi64x(-1);

        for i in 0..chunks {
            let off = i * 4;
            let x = unsafe { _mm256_loadu_si256(src.add(off) as *const __m256i) };
            let y = _mm256_xor_si256(x, all_ones);
            unsafe { _mm256_storeu_si256(dst.add(off) as *mut __m256i, y) };
        }

        for i in simd_words..word_len {
            unsafe { *dst.add(i) = !*src.add(i) };
        }
    }

    #[target_feature(enable = "avx2")]
    unsafe fn not_avx2_kernel(&self) -> (usize, Box<[u64]>) {
        let len = self.len;
        let word_len = WBLH::required_word_len(len);
        let mut data = vec![0u64; word_len];

        unsafe { Self::not_avx2_words_to(self.data.as_ptr(), data.as_mut_ptr(), word_len) };

        let mut data = data.into_boxed_slice();
        WBLH::sanitize_last_word(&mut data, len);
        (len, data)
    }

    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn not_avx2(&self) -> Self {
        let (len, data) = unsafe { self.not_avx2_kernel() };
        Self::new_unchecked(len, data)
    }

    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn not_assign_avx2(&mut self) {
        unsafe {
            Self::not_avx2_words_to(self.data.as_ptr(), self.data.as_mut_ptr(), self.data.len())
        };
        WBLH::sanitize_last_word(&mut self.data, self.len);
    }
}
