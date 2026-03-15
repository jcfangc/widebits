use crate::{MBLH, MacroBits};

impl MacroBits {
    #[inline]
    fn and_clip_scalar_kernel(lhs: &Self, rhs: &Self) -> (usize, Box<[u64]>) {
        let len = lhs.len.min(rhs.len);
        let word_len = MBLH::required_word_len(len);
        let mut data = Vec::with_capacity(word_len);

        for i in 0..word_len {
            data.push(lhs.data[i] & rhs.data[i]);
        }

        let mut data = data.into_boxed_slice();
        MBLH::sanitize_last_word(&mut data, len);
        (len, data)
    }

    #[inline]
    pub(super) fn and_clip_scalar(&self, rhs: &Self) -> Self {
        let (len, data) = Self::and_clip_scalar_kernel(self, rhs);
        Self::new_unchecked(len, data)
    }

    #[inline]
    pub(super) fn and_clip_assign_scalar(&mut self, rhs: &Self) {
        let (len, data) = Self::and_clip_scalar_kernel(self, rhs);
        self.len = len;
        self.data = data;
    }
}

impl MacroBits {
    #[inline]
    fn and_pad_scalar_kernel(lhs: &Self, rhs: &Self) -> (usize, Box<[u64]>) {
        let len = lhs.len.max(rhs.len);
        let word_len = MBLH::required_word_len(len);

        let lhs_words = lhs.data.len();
        let rhs_words = rhs.data.len();
        let common_words = lhs_words.min(rhs_words);

        let mut data = Vec::with_capacity(word_len);

        for i in 0..common_words {
            data.push(lhs.data[i] & rhs.data[i]);
        }

        data.resize(word_len, 0);

        let mut data = data.into_boxed_slice();
        MBLH::sanitize_last_word(&mut data, len);
        (len, data)
    }

    #[inline]
    pub(super) fn and_pad_scalar(&self, rhs: &Self) -> Self {
        let (len, data) = Self::and_pad_scalar_kernel(self, rhs);
        Self::new_unchecked(len, data)
    }

    #[inline]
    pub(super) fn and_pad_assign_scalar(&mut self, rhs: &Self) {
        let (len, data) = Self::and_pad_scalar_kernel(self, rhs);
        self.len = len;
        self.data = data;
    }
}
