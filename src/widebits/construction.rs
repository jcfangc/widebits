mod from_to_words;
mod ones;
mod zeros;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConstructionError {
    /// words 不足以容纳 len bit
    InsufficientWords { required: usize, provided: usize },
}
