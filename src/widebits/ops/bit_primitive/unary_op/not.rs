#[cfg(target_arch = "x86_64")]
mod avx2;
mod scalar;

use crate::WideBits;

impl WideBits {
    #[inline]
    pub fn not(&self) -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            if std::arch::is_x86_feature_detected!("avx2") {
                // SAFETY: AVX2 use is guarded by runtime feature detection.
                return unsafe { self.not_avx2() };
            }
        }

        self.not_scalar()
    }

    #[inline]
    pub fn not_assign(&mut self) {
        #[cfg(target_arch = "x86_64")]
        {
            if std::arch::is_x86_feature_detected!("avx2") {
                // SAFETY: AVX2 use is guarded by runtime feature detection.
                unsafe { self.not_assign_avx2() };
                return;
            }
        }

        self.not_assign_scalar();
    }
}
