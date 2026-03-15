use crate::{WBLH, WideBits};

impl WideBits {
    #[inline]
    pub(super) fn not_scalar(&self) -> Self {
        let len = self.len;
        let word_len = WBLH::required_word_len(len);
        let mut data = Vec::with_capacity(word_len);

        for i in 0..word_len {
            data.push(!self.data[i]);
        }

        let mut data = data.into_boxed_slice();
        WBLH::sanitize_last_word(&mut data, len);
        Self::new_unchecked(len, data)
    }

    #[inline]
    pub(super) fn not_assign_scalar(&mut self) {
        let len = self.len;
        let word_len = WBLH::required_word_len(len);

        for i in 0..word_len {
            self.data[i] = !self.data[i];
        }

        WBLH::sanitize_last_word(&mut self.data, len);
    }
}
