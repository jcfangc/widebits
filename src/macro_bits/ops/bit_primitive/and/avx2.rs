#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use crate::{MBLH, MacroBits};

impl MacroBits {
    /// Computes `lhs & rhs` into `dst` for `word_len` words, using AVX2 on the
    /// largest aligned prefix of 4-word chunks and scalar code for the tail.
    ///
    /// # Safety
    /// - The current CPU must support `avx2`.
    /// - `lhs` must be valid for reads of `word_len` consecutive `u64` values.
    /// - `rhs` must be valid for reads of `word_len` consecutive `u64` values.
    /// - `dst` must be valid for writes of `word_len` consecutive `u64` values.
    /// - The memory ranges referenced by `lhs`, `rhs`, and `dst` must not be
    ///   dangling for the duration of this call.
    /// - `dst` must not overlap with `lhs` or `rhs` in a way that violates
    ///   Rust's aliasing rules for mutable writes.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn and_avx2_words_to(lhs: *const u64, rhs: *const u64, dst: *mut u64, word_len: usize) {
        let chunks = word_len / 4;
        let simd_words = chunks * 4;

        for i in 0..chunks {
            let off = i * 4;
            let a = unsafe { _mm256_loadu_si256(lhs.add(off) as *const __m256i) };
            let b = unsafe { _mm256_loadu_si256(rhs.add(off) as *const __m256i) };
            let r = _mm256_and_si256(a, b);
            unsafe { _mm256_storeu_si256(dst.add(off) as *mut __m256i, r) };
        }

        for i in simd_words..word_len {
            unsafe {
                *dst.add(i) = *lhs.add(i) & *rhs.add(i);
            }
        }
    }
}

impl MacroBits {
    /// Builds the clipped bitwise-AND result buffer, where the output length is
    /// `min(lhs.len, rhs.len)`.
    ///
    /// # Safety
    /// - The current CPU must support `avx2`.
    /// - `lhs.data` and `rhs.data` must each contain at least
    ///   `MBLH::required_word_len(min(lhs.len, rhs.len))` initialized words.
    /// - `lhs` and `rhs` must satisfy the internal invariants required by
    ///   `MacroBits` and `MBLH`, including that their backing storage matches
    ///   their logical lengths.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn and_clip_avx2_kernel(lhs: &Self, rhs: &Self) -> (usize, Box<[u64]>) {
        let len = lhs.len.min(rhs.len);
        let word_len = MBLH::required_word_len(len);
        let mut data = vec![0u64; word_len];

        unsafe {
            Self::and_avx2_words_to(
                lhs.data.as_ptr(),
                rhs.data.as_ptr(),
                data.as_mut_ptr(),
                word_len,
            );
        }

        let mut data = data.into_boxed_slice();
        MBLH::sanitize_last_word(&mut data, len);
        (len, data)
    }

    /// Returns `self & rhs` with clipped length semantics.
    ///
    /// The output length is `min(self.len, rhs.len)`.
    ///
    /// # Safety
    /// - The current CPU must support `avx2`.
    /// - `self` and `rhs` must satisfy all internal `MacroBits` invariants:
    ///   their `data` buffers must be valid, initialized, and large enough for
    ///   their declared logical lengths.
    /// - `Self::new_unchecked(len, data)` must be valid for the produced
    ///   `(len, data)` pair. This function relies on
    ///   `and_clip_avx2_kernel` + `MBLH::sanitize_last_word` to preserve that.
    #[inline]
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn and_clip_avx2(&self, rhs: &Self) -> Self {
        let (len, data) = unsafe { Self::and_clip_avx2_kernel(self, rhs) };
        Self::new_unchecked(len, data)
    }

    /// Replaces `self` with `self & rhs` using clipped length semantics.
    ///
    /// The resulting length is `min(self.len, rhs.len)`.
    ///
    /// # Safety
    /// - The current CPU must support `avx2`.
    /// - `self` and `rhs` must satisfy all internal `MacroBits` invariants.
    /// - Replacing `self.len` and `self.data` with the values returned from
    ///   `and_clip_avx2_kernel` must preserve the invariants required by
    ///   `MacroBits`.
    #[inline]
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn and_clip_assign_avx2(&mut self, rhs: &Self) {
        let (len, data) = unsafe { Self::and_clip_avx2_kernel(self, rhs) };
        self.len = len;
        self.data = data;
    }
}

impl MacroBits {
    /// Builds the padded bitwise-AND result buffer, where the output length is
    /// `max(lhs.len, rhs.len)`.
    ///
    /// Words outside the common initialized prefix are left as zero, which
    /// matches pad-with-zero semantics for bitwise-AND.
    ///
    /// # Safety
    /// - The current CPU must support `avx2`.
    /// - `lhs` and `rhs` must satisfy the internal invariants required by
    ///   `MacroBits` and `MBLH`.
    /// - `lhs.data.len()` and `rhs.data.len()` must correctly describe the
    ///   number of initialized words available in each operand.
    /// - The first `common_words = min(lhs.data.len(), rhs.data.len())` words of
    ///   both inputs must be valid for reading.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn and_pad_avx2_kernel(lhs: &Self, rhs: &Self) -> (usize, Box<[u64]>) {
        let len = lhs.len.max(rhs.len);
        let word_len = MBLH::required_word_len(len);

        let lhs_words = lhs.data.len();
        let rhs_words = rhs.data.len();
        let common_words = lhs_words.min(rhs_words);

        let mut data = vec![0u64; word_len];

        unsafe {
            Self::and_avx2_words_to(
                lhs.data.as_ptr(),
                rhs.data.as_ptr(),
                data.as_mut_ptr(),
                common_words,
            );
        }

        let mut data = data.into_boxed_slice();
        MBLH::sanitize_last_word(&mut data, len);
        (len, data)
    }

    /// Returns `self & rhs` with padded length semantics.
    ///
    /// The output length is `max(self.len, rhs.len)`, and missing bits are
    /// treated as zero.
    ///
    /// # Safety
    /// - The current CPU must support `avx2`.
    /// - `self` and `rhs` must satisfy all internal `MacroBits` invariants:
    ///   their `data` buffers must be valid, initialized, and consistent with
    ///   their logical lengths.
    /// - `Self::new_unchecked(len, data)` must be valid for the produced
    ///   `(len, data)` pair.
    #[inline]
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn and_pad_avx2(&self, rhs: &Self) -> Self {
        let (len, data) = unsafe { Self::and_pad_avx2_kernel(self, rhs) };
        Self::new_unchecked(len, data)
    }

    /// Replaces `self` with `self & rhs` using padded length semantics.
    ///
    /// The resulting length is `max(self.len, rhs.len)`, and missing bits are
    /// treated as zero.
    ///
    /// # Safety
    /// - The current CPU must support `avx2`.
    /// - `self` and `rhs` must satisfy all internal `MacroBits` invariants.
    /// - Replacing `self.len` and `self.data` with the values returned from
    ///   `and_pad_avx2_kernel` must preserve the invariants required by
    ///   `MacroBits`.
    #[inline]
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn and_pad_assign_avx2(&mut self, rhs: &Self) {
        let (len, data) = unsafe { Self::and_pad_avx2_kernel(self, rhs) };
        self.len = len;
        self.data = data;
    }
}
