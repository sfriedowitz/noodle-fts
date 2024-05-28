mod euler;
mod updater;

pub use euler::EulerUpdater;
pub use updater::FieldUpdater;

#[derive(thiserror::Error, Debug)]
pub enum UpdaterError {}
