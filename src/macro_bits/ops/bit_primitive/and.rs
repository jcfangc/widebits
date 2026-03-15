mod avx2;
mod scalar;

use crate::MacroBits;

impl MacroBits {
    #[inline]
    pub fn and_clip(&self, rhs: &Self) -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            if std::arch::is_x86_feature_detected!("avx2") {
                // SAFETY: AVX2 use is guarded by runtime feature detection.
                return unsafe { self.and_clip_avx2(rhs) };
            }
        }

        self.and_clip_scalar(rhs)
    }

    #[inline]
    pub fn and_clip_assign(&mut self, rhs: &Self) {
        #[cfg(target_arch = "x86_64")]
        {
            if std::arch::is_x86_feature_detected!("avx2") {
                // SAFETY: AVX2 use is guarded by runtime feature detection.
                unsafe { self.and_clip_assign_avx2(rhs) };
                return;
            }
        }

        self.and_clip_assign_scalar(rhs);
    }

    #[inline]
    pub fn and_pad(&self, rhs: &Self) -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            if std::arch::is_x86_feature_detected!("avx2") {
                // SAFETY: AVX2 use is guarded by runtime feature detection.
                return unsafe { self.and_pad_avx2(rhs) };
            }
        }

        self.and_pad_scalar(rhs)
    }

    #[inline]
    pub fn and_pad_assign(&mut self, rhs: &Self) {
        #[cfg(target_arch = "x86_64")]
        {
            if std::arch::is_x86_feature_detected!("avx2") {
                // SAFETY: AVX2 use is guarded by runtime feature detection.
                unsafe { self.and_pad_assign_avx2(rhs) };
                return;
            }
        }

        self.and_pad_assign_scalar(rhs);
    }
}
