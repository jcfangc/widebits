use crate::MacroBits;

mod extend;
mod truncate;

impl MacroBits {
    #[inline]
    pub fn resize(&mut self, new_len: usize) {
        if new_len > self.len {
            self.extend(new_len);
        } else {
            self.truncate(new_len);
        }
    }
}
