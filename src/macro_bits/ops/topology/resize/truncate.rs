use crate::{MBLH, MacroBits};

impl MacroBits {
    #[inline]
    pub fn truncate(&mut self, new_len: usize) {
        if new_len >= self.len {
            return;
        }

        self.len = new_len;
        let required = MBLH::required_word_len(new_len);
        self.data = self.data[..required].into();

        MBLH::sanitize_last_word(&mut self.data, new_len);
    }
}
